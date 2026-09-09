import os
import time
import torch
from datetime import datetime
from flwr.client import NumPyClient, ClientApp
from task import (
    TabularNet, load_data, CLIENT_RESOURCE_PROFILES, calculate_network_latency
)
from config import (
    RESOURCE_MODE, FL_METHOD_STR, ACTIVE_CONFIG, DATA_DIR, 
    NUM_CLASSES, META_FILE_PATH, LOCAL_EPOCHS
)

LEARNING_RATE = 0.001

def get_timestamp():
    return datetime.now().strftime("%H:%M:%S")

def print_config_details(cid: int, profile_name: str):
    print("\n" + "="*60)
    print(f"[{get_timestamp()}] 📌 Main Config CLIENT {cid+1:02d}")
    print("-"*60)
    print(f"  ∟ Resource Constraint : {'Yes' if RESOURCE_MODE else 'No'}")
    print(f"  ∟ Method              : {FL_METHOD_STR}")
    print(f"  ∟ Dataset             : {ACTIVE_CONFIG['name']}")
    print(f"  ∟ Client Profile      : {profile_name}")
    print("="*60 + "\n")

class FedAvgOneShotClient(NumPyClient):
    def __init__(self, trainloader, valloader, num_features: int, cid: int):
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.num_features = num_features
        self.cid = cid
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        profile_id = cid if RESOURCE_MODE else 0
        self.profile = CLIENT_RESOURCE_PROFILES.get(profile_id, CLIENT_RESOURCE_PROFILES[0])
        self.model = TabularNet(input_dim=num_features, num_classes=NUM_CLASSES).to(self.device)

        # Inspect local dataset class presence
        self.present_classes = []
        all_y = torch.cat([y_b for _, y_b in self.trainloader], dim=0)
        for c in range(NUM_CLASSES):
            if (all_y == c).sum().item() > 0:
                self.present_classes.append(int(c))
                
        print(f"[{get_timestamp()}] 📌 Client {self.cid+1:02d} local class presence: {self.present_classes}")

    def fit(self, parameters, config):
        start_time = time.time()
        print(f"[{get_timestamp()}] ⚡ Client {self.cid+1:02d}: Training local TabularNet for {LOCAL_EPOCHS} epochs...")
        
        if parameters and len(parameters) > 0:
            state_dict = zip(self.model.state_dict().keys(), parameters)
            self.model.load_state_dict({k: torch.tensor(v) for k, v in state_dict})

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        self.model.train()

        for epoch in range(LOCAL_EPOCHS):
            for x_batch, y_batch in self.trainloader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                out = self.model(x_batch)
                loss = criterion(out, y_batch)
                loss.backward()
                optimizer.step()

        duration = time.time() - start_time
        print(f"[{get_timestamp()}] ✅ Client {self.cid+1:02d}: Training completed in {duration:.2f}s ({duration/60:.2f}m).")

        # Convert state_dict to list of NumPy arrays for parameter transport
        model_weights = [val.detach().cpu().numpy().astype("float32") for val in self.model.state_dict().values()]

        # Compute payload size for network delay simulation
        if RESOURCE_MODE:
            param_bytes = sum(v.element_size() * v.nelement() for v in self.model.state_dict().values())
            time.sleep(calculate_network_latency(self.cid, payload_bytes=param_bytes))

        return model_weights, len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        return 0.0, len(self.valloader.dataset), {"accuracy": 0.0}

def client_fn(context):
    cid = int(context.node_config.get("cid", 0))
    profile_id = cid if RESOURCE_MODE else 0
    profile = CLIENT_RESOURCE_PROFILES.get(profile_id, CLIENT_RESOURCE_PROFILES[0])

    print_config_details(cid, profile["name"])

    data_path = os.path.join(DATA_DIR, f"client_{cid + 1:02d}.pt")    
    trainloader, valloader = load_data(data_path, batch_size=profile["batch_size"])

    meta_info = torch.load(META_FILE_PATH, weights_only=True)
    return FedAvgOneShotClient(trainloader, valloader, num_features=meta_info["num_features"], cid=cid).to_client()

app = ClientApp(client_fn=client_fn)
