import torch
from flwr.client import NumPyClient, ClientApp
from task import TabularNet, load_data

# ==========================================
# ⚙️ CLIENT CONTROL PANEL (HYPERPARAMETERS)
# ==========================================
DEFAULT_DATA_PATH = "./data/default.pt"
LOCAL_EPOCHS = 10         # Number of passes over local data per round. 3 or 1 or 5
LEARNING_RATE = 0.001     # 0.001 Adam Optimizer, 0.05 Step size for SGD
BATCH_SIZE = 32           # Images per training batch
# ==========================================

class SimpleClient(NumPyClient):
    def __init__(self, trainloader, valloader, net):
        self.trainloader = trainloader
        self.valloader = valloader
        self.net = net

    def fit(self, parameters, config):
        # 1. Print local data info
        print(f"🏃 ===> Training started on {len(self.trainloader.dataset)} local samples.")
        
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
            print(f"Epoch {epoch+1}/{LOCAL_EPOCHS} - ===> LOCAL-TRAINING Accuracy: {epoch_acc:.4f}")

        # 3. Return updated weights and local dataset size
        weights = [val.cpu().numpy() for val in self.net.state_dict().values()]
        return weights, len(self.trainloader.dataset), {"accuracy": float(epoch_acc)}

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
    # 1. Check for path in terminal config, else use Macro default
    data_path = context.node_config.get("data-path", DEFAULT_DATA_PATH)
    
    # 2. Load data (Batch size could be added to task.load_data if needed)
    trainloader, valloader = load_data(data_path)
    
    net = TabularNet()
    return SimpleClient(trainloader, valloader, net).to_client()

app = ClientApp(client_fn=client_fn)
