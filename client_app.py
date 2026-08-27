import os
import time
import torch
from flwr.client import NumPyClient, ClientApp
from task import TabularNet, load_data, CLIENT_RESOURCE_PROFILES, calculate_network_latency, debug_print
from config import RESOURCE_MODE, DATA_DIR, CLIENT_FILE_PREFIX, NUM_CLASSES, META_FILE_PATH, ACTIVE_CONFIG

# ==========================================
# ⚙️ CLIENT CONTROL PANEL (HYPERPARAMETERS)
# ==========================================
LOCAL_EPOCHS = 10         # Number of passes over local data per round. 3 or 1 or 5
LEARNING_RATE = 0.001     # 0.001 Adam Optimizer, 0.05 Step size for SGD
# ==========================================


class SimpleClient(NumPyClient):
    def __init__(self, trainloader, valloader, net, cid: int):
        self.trainloader = trainloader
        self.valloader = valloader
        self.net = net
        self.cid = cid
        
        #""" profile_id = 0 if RESOURCE_MODE == 0 else cid
        profile_id = cid if RESOURCE_MODE else 0
        self.profile = CLIENT_RESOURCE_PROFILES.get(profile_id, CLIENT_RESOURCE_PROFILES[0])

    def fit(self, parameters, config):
        start_time = time.time()
        print(f"🏃 [CLIENT {self.cid+1} | {self.profile['name']}] ===> Training started on {len(self.trainloader.dataset)} local samples.")
        
        # 1. Update local model with server parameters
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.net.load_state_dict(state_dict, strict=True)

        # Use Macro for Learning Rate
        optimizer = torch.optim.Adam(self.net.parameters(), lr=LEARNING_RATE)    #torch.optim.SGD(self.net.parameters(), lr=LEARNING_RATE)
        criterion = torch.nn.CrossEntropyLoss()

        # 2. Local Training Loop with Multiple Epochs
        self.net.train()
        for epoch in range(LOCAL_EPOCHS): # <--- Added Local Epochs logic
            correct, total = 0, 0
            for x_batch, y_batch in self.trainloader:
                optimizer.zero_grad()
                outputs = self.net(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
            
            epoch_acc = correct / total

            # Simulate System / Computation Delay per epoch
            cpu_delay = self.profile["cpu_delay"]
            time.sleep(cpu_delay)
            debug_print(f"---- Simulated system delay: {cpu_delay:.3f} sec")

            print(f"Epoch {epoch+1}/{LOCAL_EPOCHS} - ===> LOCAL-TRAINING Accuracy: {epoch_acc:.4f}")

        print(f"   ∟ Client {self.cid+1} Local Epochs Completed. Last Epoch Acc: {epoch_acc:.4f}")

        # 3. Simulate Network Latency Delay
        if RESOURCE_MODE:
            param_bytes = sum(v.element_size() * v.nelement() for v in self.net.state_dict().values())    # calculate model size in bytes
            simulated_delay = calculate_network_latency(self.cid, payload_bytes=param_bytes)    # (model_size × 2) / bandwidth
            time.sleep(simulated_delay) # Hold process to simulate physical delay
            debug_print(f"---- Simulated network delay: {simulated_delay:.3f} sec")

        total_wall_clock = (time.time() - start_time)
        debug_print(f"---- total_wall_clock= {total_wall_clock}")

        # 4. Return updated weights, local dataset size and telemetry metrics back to server
        weights = [val.cpu().numpy() for val in self.net.state_dict().values()]
        return weights, len(self.trainloader.dataset), {
            "accuracy": float(epoch_acc),
            "wall_clock_time": float(total_wall_clock),
            "profile_name": str(self.profile["name"])
        }

    def evaluate(self, parameters, config):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.net.load_state_dict(state_dict, strict=True)

        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        
        self.net.eval()
        with torch.no_grad():
            for x_batch, y_batch in self.valloader:
                outputs = self.net(x_batch)
                loss += criterion(outputs, y_batch).item()
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        accuracy = correct / total if total > 0 else 0.0
        print(f"===> LOCAL-EVALUATE Accuracy: {accuracy:.4f}")
        return float(loss / len(self.valloader)), len(self.valloader.dataset), {"accuracy": float(accuracy)}

def client_fn(context):
    # 1. Read explicit node configs passed from bash. (cid default to 0)
    cid = int(context.node_config.get("cid", 0))
    data_path = context.node_config.get("data-path", f"./data/client_01.pt")

    # 2. Format filename to match 'client_01.pt', 'client_02.pt', etc. (1-indexed filename)
    filename = f"client_{cid + 1:02d}.pt"
    data_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(data_path):
        print(f"❌ Error: Data folder '{DATA_DIR}' or Data file '{data_path}' not exists. Please reverify prepare_tabular_data.py execution first.")
        return
    
    dataset_name = ACTIVE_CONFIG["name"]
    print(f"==== Starting ClientApp Dataset= '{dataset_name}' ====")

    # 3. Lookup profile to pass client-specific batch_size to Load Data
    # profile = CLIENT_RESOURCE_PROFILES[cid]
    profile = CLIENT_RESOURCE_PROFILES.get(cid, CLIENT_RESOURCE_PROFILES[0])

    # 4. Load dataset and initialize client
    trainloader, valloader = load_data(data_path, batch_size=profile["batch_size"])

    # model initialization script
    meta_info = torch.load(META_FILE_PATH)
    NUM_FEATURES = meta_info["num_features"]  # e.g., dynamically resolved to 8, 12, or 15
    debug_print(f"---- client: NUM_FEATURES= {NUM_FEATURES}, type = {type(NUM_FEATURES)}, NUM_CLASSES= {NUM_CLASSES}, type = {type(NUM_CLASSES)}\n")

    #""" net = TabularNet()
    net = TabularNet(NUM_FEATURES, NUM_CLASSES)
    return SimpleClient(trainloader, valloader, net, cid=cid).to_client()

app = ClientApp(client_fn=client_fn)
