import os
import time
import torch
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import SimpleClientManager
from task import TabularNet, debug_print
from config import RESOURCE_MODE, NUM_CLIENTS, NUM_CLASSES, META_FILE_PATH, OUTPUT_DIR, MODEL_FILE_PATH, ACTIVE_CONFIG

# ==========================================
# ⚙️ EXPERIMENT CONTROL PANEL (CONSTANTS)
# ==========================================
NUM_ROUNDS = 20             # Total FL rounds (z)
EVAL_EVERY_N_ROUNDS = 1     # =2 Only evaluate every 2nd round (1, 3, 5...)
# ==========================================
# NUM_CLIENTS = 0             # N = 10 clients (Minimum to start a round)    # MIN_AVAILABLE_CLIENTS
# ==========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

def weighted_average(metrics):
    accuracies = [m["accuracy"] * num_examples for num_examples, m in metrics]
    examples = [num_examples for num_examples, m in metrics]
    
    for num_examples, m in metrics:
        acc = m["accuracy"]
        print(f"===> Client Report -> Accuracy: {acc:.4f} | Samples: {num_examples}")
    
    return {"accuracy": sum(accuracies) / sum(examples)}

class CustomFedAvg(FedAvg):
    def __init__(self, num_features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_features = num_features
        self.latest_weights = None
        self.total_upstream_bytes = 0
        self.start_time = None

    def aggregate_fit(self, server_round, results, failures):
        if server_round == 1:
            self.start_time = time.time()
            print("\n" + "="*60)
            print("🚀 FEDERATED LEARNING EXPERIMENT STARTED")
            print("="*60)

        ### Print how many clients sent data
        print(f"\n📦 [ROUND {server_round}/{NUM_ROUNDS}] ===> Received updates from {len(results)} clients.")

        # Calculate size of weights transferred (Baseline Upstream Payload tracking)
        round_bytes = 0
        for i, (client, fit_res) in enumerate(results):
            # Calculate size of weights in Kilobytes (approximation)
            param_bytes = sum(t.nbytes for t in parameters_to_ndarrays(fit_res.parameters))
            round_bytes += param_bytes
            print(f"   ∟ ===> Client {i+1} sent: {param_bytes / 1024:.2f} KB")
        ### ###
        self.total_upstream_bytes += round_bytes
        round_mb = round_bytes / (1024 * 1024)
        print(f"   ∟ Upstream Traffic in Round {server_round}: {round_mb:.4f} MB i.e. {round_bytes / 1024:.2f} KB")

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        # Save model at the final round
        if aggregated_parameters is not None and server_round == NUM_ROUNDS:
            print("" + "-"*60)
            print(f"📝 [FINAL ROUND {server_round}] Saving final model weights to {MODEL_FILE_PATH}...")
            self.latest_weights = aggregated_parameters
            self.save_model_weights(self.latest_weights)
            print("-"*60 + "")

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        # 1. Perform standard aggregation
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        #""" if not metrics or "accuracy" not in metrics:
        #"""    return loss, metrics

        if metrics and "accuracy" in metrics:
          current_acc = metrics["accuracy"]
          print(f"📢 [Round {server_round}] ===> Global Accuracy: {current_acc:.4f}")

        # Final experiment summary
        if server_round == NUM_ROUNDS:
            # Print Final Experiment Summary
            elapsed_time = time.time() - self.start_time

            # Total Comm = Upstream + Downstream (Symmetric push/pull approximation)
            total_comm_mb = (self.total_upstream_bytes * 2) / (1024 * 1024)

            dataset_name = ACTIVE_CONFIG["name"]
            print("\n" + "="*60)
            print("📊 FINAL TELEMETRY SUMMARY FOR PAPER SECTION 5")
            print("="*60)
            print(f"  ∟ Dataset                        : {dataset_name}")
            print(f"  L Resource Constraint            : {'Yes' if RESOURCE_MODE else 'No'}")
            print(f"  ∟ Total Communication Rounds (z) : {NUM_ROUNDS}")
            print(f"  ∟ Active Edge Clients (N)        : {NUM_CLIENTS}")
            print(f"  ∟ Total Upstream Payload         : {self.total_upstream_bytes / (1024*1024):.4f} MB")
            print(f"  ∟ Est. Total Bandwidth           : {total_comm_mb:.4f} MB    <==== Est. Total Communication (Up+Down, symmetric)")
            print(f"  ∟ Total Execution Time           : {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} mins)")
            if metrics and "accuracy" in metrics:
                print(f"  ∟ Final Global Accuracy          : {metrics['accuracy']:.4f}")
            print("="*60 + "\n")

        return loss, metrics

    def save_model_weights(self, results):
        """Helper to physically save the parameters from the strategy."""
        # Note: In a production Strategy, we usually save parameters 
        # inside aggregate_fit. This is a simplified version for your local test.
        # It's better to save after aggregate_fit to ensure we have the math.
        #pass
        print(f"📝 Saving model at {MODEL_FILE_PATH} num_features= {self.num_features}")
        ndarrays = parameters_to_ndarrays(self.latest_weights)
        #""" net = TabularNet()
        net = TabularNet(self.num_features, NUM_CLASSES)
        params_dict = zip(net.state_dict().keys(), ndarrays)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        net.load_state_dict(state_dict)
        torch.save(net.state_dict(), MODEL_FILE_PATH)

    def configure_evaluate(self, server_round, parameters, client_manager):
        # 2. 🔋 POWER SAVING LOGIC: Skip evaluation if not the Nth round
        if server_round % EVAL_EVERY_N_ROUNDS != 0:
            print(f"🔋 [ROUND {server_round}] Skipping Evaluation to save Client Battery.")
            return []
        return super().configure_evaluate(server_round, parameters, client_manager)

class DetailedClientManager(SimpleClientManager):
    """Custom manager to log every time a client hits the server"""
    def register(self, client) -> bool:
        # Step 1: Perform the actual registration first
        res = super().register(client)

        # Step 2: Only print if registration was successful
        if res:
            # avoid calling self.num_available() here to prevent recursion
            print(f"--- 🛡️ [NEW CLIENT DETECTED] ---")
            print(f"Status: Client added to Wait-List | Client ID: {client.cid}") # cid is the Unique ID Flower gives each client
            print(f"----------------------------------\n")
        return res

def server_fn(context):
    #""" NUM_CLIENTS = context.run_config["num-clients"]    From TOML (old approach)
    dataset_name = ACTIVE_CONFIG["name"]
    print(f"==== Starting ServerApp Dataset= '{dataset_name}' ====")

    # model initialization script
    meta_info = torch.load(META_FILE_PATH)
    num_features = meta_info["num_features"]  # e.g., dynamically resolved to 8, 12, or 15
    debug_print(f"---- server: num_features= {num_features}, type = {type(num_features)}, NUM_CLASSES= {NUM_CLASSES}, type = {type(NUM_CLASSES)}\n")

    #""" net = TabularNet()
    net = TabularNet(num_features, NUM_CLASSES)
    ndarrays = [val.cpu().numpy() for val in net.state_dict().values()]
    initial_params = ndarrays_to_parameters(ndarrays)

    # Initialize the custom manager
    client_manager = DetailedClientManager()

    strategy = CustomFedAvg(
        num_features=num_features,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        initial_parameters=initial_params,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    
    config = ServerConfig(num_rounds=NUM_ROUNDS)
    return ServerAppComponents(strategy=strategy, config=config, client_manager=client_manager)

app = ServerApp(server_fn=server_fn)
