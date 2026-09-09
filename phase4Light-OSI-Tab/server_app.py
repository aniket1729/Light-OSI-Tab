import os
import time
import json
import torch
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import SimpleClientManager
from task import TabularNet, Tabular1DVAE
from config import (
    RESOURCE_MODE, NUM_CLIENTS, NUM_CLASSES, META_FILE_PATH, 
    OUTPUT_DIR, MODEL_FILE_PATH, ACTIVE_CONFIG, FL_METHOD_STR,
    LATENT_DIM, SYNTHETIC_SAMPLES_PER_CLASS
)

NUM_ROUNDS = 1
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_timestamp():
    return datetime.now().strftime("%H:%M:%S")

def print_config_details():
    print("\n" + "="*60)
    print(f"[{get_timestamp()}] 📌 Main Config SERVER")
    print("-"*60)
    print(f"  ∟ Resource Constraint : {'Yes' if RESOURCE_MODE else 'No'}")
    print(f"  ∟ Method              : {FL_METHOD_STR}")
    print(f"  ∟ Dataset             : {ACTIVE_CONFIG['name']}")
    print("="*60 + "\n")

class Objective1OneShotStrategy(FedAvg):
    def __init__(self, num_features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_features = num_features

    def aggregate_fit(self, server_round, results, failures):
        start_time = time.time()
        print("\n" + "="*60)
        print(f"🚀STABILIZED DATA-FREE RECONSTRUCTION")
        print("="*60)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[{get_timestamp()}] ⚡ Server Computation Device: {device}")
        print(f"[{get_timestamp()}] 📥 Received {len(results)} client results and {len(failures)} failures.")

        vae_models = []
        client_present_classes = []

        for i, (client, fit_res) in enumerate(results):
            try:
                param_arrays = parameters_to_ndarrays(fit_res.parameters)
                local_vae = Tabular1DVAE(input_dim=self.num_features, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES)
                
                state_dict = {}
                for key, arr in zip(local_vae.state_dict().keys(), param_arrays):
                    state_dict[key] = torch.tensor(arr)
                
                local_vae.load_state_dict(state_dict)
                local_vae.eval()
                local_vae.to(device)
                vae_models.append(local_vae)

                # Retrieve present classes reported by this client
                p_classes = json.loads(fit_res.metrics.get("present_classes_json", "[]")) if fit_res.metrics else list(range(NUM_CLASSES))
                client_present_classes.append(p_classes)
            except Exception as e:
                print(f"[{get_timestamp()}] ❌ Failed to load Client {i} parameters: {e}")

        if len(vae_models) == 0:
            raise RuntimeError("❌ CRITICAL: No client VAE models were successfully aggregated by the server!")

        # 1. Dual-Temperature Latent Generation
        gen_start_time = time.time()
        print(f"[{get_timestamp()}] ⚡ Generating class-balanced synthetic dataset...")
        synth_x_list, synth_y_list = [], []
        temperatures = [0.80, 1.00]

        with torch.no_grad():
            for c in range(NUM_CLASSES):
                # Find decoders that actually saw Class c during local training
                eligible_vaes = [
                    vae_models[idx] for idx, p_classes in enumerate(client_present_classes) if c in p_classes
                ]
                
                if len(eligible_vaes) == 0:
                    print(f"⚠️ Warning: No client holds class {c}. Falling back to all decoders.")
                    eligible_vaes = list(vae_models)

                num_vaes = max(1, len(eligible_vaes))
                base_samples_per_vae = max(1, SYNTHETIC_SAMPLES_PER_CLASS // num_vaes)

                for vae in eligible_vaes:
                    for temp in temperatures:
                        sub_samples = base_samples_per_vae // len(temperatures)
                        z_samples = torch.randn(sub_samples, LATENT_DIM, device=device) * temp
                        y_samples = torch.full((sub_samples,), c, dtype=torch.long, device=device)
                        
                        reconstructed_x = vae.decode(z_samples, y_samples)
                        synth_x_list.append(reconstructed_x.cpu())
                        synth_y_list.append(y_samples.cpu())

        synth_X = torch.cat(synth_x_list, dim=0)
        synth_Y = torch.cat(synth_y_list, dim=0)
        gen_duration = time.time() - gen_start_time
        print(f"[{get_timestamp()}] ✅ Reconstructed {len(synth_X)} samples in {gen_duration:.2f}s.")

        # 2. Downstream Classifier Training (Light Label Smoothing)
        clf_start_time = time.time()
        epochs = 20
        print(f"\n[{get_timestamp()}] 🎓 Training TabularNet ({epochs} Epochs)...")

        SERVER_BATCH_SIZE = 256
        synth_loader = DataLoader(TensorDataset(synth_X, synth_Y), batch_size=SERVER_BATCH_SIZE, shuffle=True)
        global_net = TabularNet(self.num_features, NUM_CLASSES).to(device)
        optimizer = torch.optim.AdamW(global_net.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

        print(f"\n[{get_timestamp()}] 🎓 Training TabularNet ({epochs} Epochs, Batch Size: {SERVER_BATCH_SIZE})...")
        global_net.train()
        train_start = time.time()

        for epoch in range(1, epochs + 1):
            ep_start = time.time()
            running_loss = 0.0
            total_batches = 0

            for x_b, y_b in synth_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                out = global_net(x_b)
                loss = criterion(out, y_b)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                total_batches += 1

            scheduler.step()
            ep_time = time.time() - ep_start
            total_elapsed = time.time() - train_start
            avg_loss = running_loss / total_batches

            if epoch == 1 or epoch % 1 == 0 or epoch == epochs:
                print(f"   ∟ Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.4f} - Epoch Time: {ep_time:.2f}s - Elapsed: {total_elapsed:.1f}s")

        clf_duration = time.time() - clf_start_time
        print(f"[{get_timestamp()}] ✅ Classifier training finished in {clf_duration:.2f}s.")

        # Save model
        torch.save(global_net.cpu().state_dict(), MODEL_FILE_PATH)
        print(f"[{get_timestamp()}] 📝 Final Global Model saved to {MODEL_FILE_PATH}\n")

        final_params = [val.cpu().numpy() for val in global_net.state_dict().values()]
        return ndarrays_to_parameters(final_params), {}

def server_fn(context):
    print_config_details()
    meta_info = torch.load(META_FILE_PATH, weights_only=True)
    num_features = meta_info["num_features"]

    net = TabularNet(num_features, NUM_CLASSES)
    initial_params = ndarrays_to_parameters([val.cpu().numpy() for val in net.state_dict().values()])

    strategy = Objective1OneShotStrategy(
        num_features=num_features,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        initial_parameters=initial_params
    )
    return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=NUM_ROUNDS), client_manager=SimpleClientManager())

app = ServerApp(server_fn=server_fn)
