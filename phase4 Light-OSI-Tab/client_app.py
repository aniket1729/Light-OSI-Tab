import os
import time
import json
import torch
from datetime import datetime
from flwr.client import NumPyClient, ClientApp
from task import (
    Tabular1DVAE, vae_loss_function, load_data, 
    CLIENT_RESOURCE_PROFILES, calculate_network_latency
)
from config import (
    RESOURCE_MODE, FL_METHOD_STR, ACTIVE_CONFIG, DATA_DIR, NUM_CLASSES, META_FILE_PATH, LATENT_DIM
)

LOCAL_VAE_EPOCHS = 100
LEARNING_RATE = 0.001

def get_timestamp():
    return datetime.now().strftime("%H:%M:%S")

def print_config_details(cid: int):
    print("\n" + "="*60)
    print(f"[{get_timestamp()}] 📌 Main Config CLIENT {cid+1:02d}")
    print("-"*60)
    print(f"  ∟ Resource Constraint : {'Yes' if RESOURCE_MODE else 'No'}")
    print(f"  ∟ Method              : {FL_METHOD_STR}")
    print(f"  ∟ Dataset             : {ACTIVE_CONFIG['name']}")
    print("="*60 + "\n")

class Objective1OneShotClient(NumPyClient):
    def __init__(self, trainloader, valloader, num_features: int, cid: int):
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.num_features = num_features
        self.cid = cid
        
        profile_id = cid if RESOURCE_MODE else 0
        self.profile = CLIENT_RESOURCE_PROFILES.get(profile_id, CLIENT_RESOURCE_PROFILES[0])
        self.vae = Tabular1DVAE(input_dim=num_features, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES)

        # Inspect local dataset to identify available classes
        self.present_classes = []
        all_y = torch.cat([y_b for _, y_b in self.trainloader], dim=0)
        
        for c in range(NUM_CLASSES):
            if (all_y == c).sum().item() > 0:
                self.present_classes.append(int(c))
                
        print(f"[{get_timestamp()}] 📌 Client {self.cid+1:02d} local class presence: {self.present_classes}")

    def fit(self, parameters, config):
        start_time = time.time()
        print(f"[{get_timestamp()}] ⚡ Client {self.cid+1:02d}: Training local VAE for {LOCAL_VAE_EPOCHS} epochs...")
        
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=LEARNING_RATE)
        self.vae.train()
        
        for epoch in range(LOCAL_VAE_EPOCHS):
            for x_batch, y_batch in self.trainloader:
                optimizer.zero_grad()
                recon_batch, mu, logvar = self.vae(x_batch, y_batch)
                loss = vae_loss_function(recon_batch, x_batch, mu, logvar)
                loss.backward()
                optimizer.step()

        duration = time.time() - start_time
        print(f"[{get_timestamp()}] ✅ Client {self.cid+1:02d}: VAE local training completed in {duration:.2f}s.")

        vae_weights = [val.detach().cpu().numpy().astype("float32") for val in self.vae.state_dict().values()]

        if RESOURCE_MODE:
            param_bytes = sum(v.element_size() * v.nelement() for v in self.vae.state_dict().values())
            time.sleep(calculate_network_latency(self.cid, payload_bytes=param_bytes))

        # Include present classes in metrics
        metrics = {
            "present_classes_json": json.dumps(self.present_classes)
        }

        return vae_weights, len(self.trainloader.dataset), metrics

    def evaluate(self, parameters, config):
        return 0.0, len(self.valloader.dataset), {"accuracy": 0.0}

def client_fn(context):
    cid = int(context.node_config.get("cid", 0))
    profile = CLIENT_RESOURCE_PROFILES.get(cid, CLIENT_RESOURCE_PROFILES[0])

    print_config_details(cid)

    data_path = os.path.join(DATA_DIR, f"client_{cid + 1:02d}.pt")    
    trainloader, valloader = load_data(data_path, batch_size=profile["batch_size"])

    meta_info = torch.load(META_FILE_PATH, weights_only=True)
    return Objective1OneShotClient(trainloader, valloader, num_features=meta_info["num_features"], cid=cid).to_client()

app = ClientApp(client_fn=client_fn)
