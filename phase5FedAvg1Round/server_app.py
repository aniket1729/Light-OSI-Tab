import os
import time
import torch
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import SimpleClientManager
from task import TabularNet
from config import (
    RESOURCE_MODE, NUM_CLIENTS, NUM_CLASSES, META_FILE_PATH, DATA_DIR, EVAL_FILE,
    OUTPUT_DIR, MODEL_FILE_PATH, ACTIVE_CONFIG, FL_METHOD_STR
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

class FedAvgOneShotStrategy(FedAvg):
    def __init__(self, num_features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_features = num_features
        self.start_time = time.time()

    def aggregate_fit(self, server_round, results, failures):
        print("\n" + "="*60)
        print(f"[{get_timestamp()}] 🚀 SERVER AGGREGATION & GLOBAL MODEL EVALUATION")
        print("="*60)
        agg_start = time.time()

        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Convert aggregated parameters to PyTorch state_dict
            ndarrays = parameters_to_ndarrays(aggregated_parameters)
            global_net = TabularNet(input_dim=self.num_features, num_classes=NUM_CLASSES)
            params_dict = zip(global_net.state_dict().keys(), ndarrays)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            global_net.load_state_dict(state_dict)

            # Save global aggregated model to output/best_model.pth
            torch.save(state_dict, MODEL_FILE_PATH)
            print(f"[{get_timestamp()}] 📝 Saved final global aggregated model to {MODEL_FILE_PATH}")

            # Measure Upstream Payload & Bandwidth
            param_bytes = sum(v.element_size() * v.nelement() for v in state_dict.values())
            single_client_mb = param_bytes / (1024 * 1024)
            total_upstream_mb = single_client_mb * len(results)
            est_total_bandwidth_mb = total_upstream_mb * 2.0  # Downstream + Upstream

            # Integrated Unseen Test Set Evaluation (global_test.pt)
            test_file_path = os.path.join(DATA_DIR, EVAL_FILE)
            test_acc, macro_f1 = 0.0, 0.0

            if os.path.exists(test_file_path):
                test_data = torch.load(test_file_path, weights_only=True)
                X_test = test_data["x"].float()
                y_test = test_data["y"].long()

                global_net.eval()
                with torch.no_grad():
                    outputs = global_net(X_test)
                    _, preds = torch.max(outputs, 1)

                y_true = y_test.numpy()
                y_pred = preds.numpy()

                test_acc = accuracy_score(y_true, y_pred) * 100.0
                report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                macro_f1 = report["macro avg"]["f1-score"]

            total_duration_min = (time.time() - self.start_time) / 60.0

            # Print Standardized Final Summary Box
            print("\n" + "="*60)
            print("📊 FINAL EXPERIMENTAL SUMMARY")
            print("------------------------------------------------------------")
            print(f"  ∟ Resource Constraint   : {'Yes' if RESOURCE_MODE else 'No'}")
            print(f"  ∟ Method Name           : {FL_METHOD_STR}")
            print(f"  ∟ Dataset Name          : {ACTIVE_CONFIG['name']}")
            print(f"  ∟ Communication Rounds  : {NUM_ROUNDS}")
            print(f"  ∟ Client Count          : {len(results)}")
            print(f"  ∟ Upstream Payload      : {total_upstream_mb:.2f} MB")
            print(f"  ∟ Est. Total Bandwidth  : {est_total_bandwidth_mb:.2f} MB")
            print(f"  ∟ Total Training Time   : {total_duration_min:.2f} mins")
            print(f"  ∟ Global Test Accuracy  : {test_acc:.2f}%")
            print(f"  ∟ Macro F1-Score        : {macro_f1:.4f}")
            print("="*60 + "\n")

        return aggregated_parameters, metrics

def server_fn(context):
    print_config_details()
    meta_info = torch.load(META_FILE_PATH, weights_only=True)
    num_features = meta_info["num_features"]

    net = TabularNet(num_features, NUM_CLASSES)
    initial_params = ndarrays_to_parameters([val.cpu().numpy() for val in net.state_dict().values()])

    strategy = FedAvgOneShotStrategy(
        num_features=num_features,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        initial_parameters=initial_params
    )
    return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=NUM_ROUNDS), client_manager=SimpleClientManager())

app = ServerApp(server_fn=server_fn)
