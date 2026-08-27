# task.py (Tabular Baseline Version with Heterogeneous System Support)
import os
import re
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from config import RESOURCE_MODE, DEBUG

# ==========================================
# ⚙️ TABULAR TASK CONTROL PANEL
# ==========================================
#""" NUM_FEATURES = 27         # Number of input tabular columns (Adjust to CSV)
#""" NUM_CLASSES = 3           # Number of target output classes
DEFAULT_BATCH_SIZE = 32   # Default Fallback Batch Size
TRAINING_DATA_SPLIT = 0.8
# ==========================================

# =======================================================
# ⚙️ HARDWARE RESOURCE PROFILES (10 Heterogeneous CLIENTS)
# =======================================================
if not RESOURCE_MODE:
    CLIENT_RESOURCE_PROFILES = {
        0: {"name": "Default",    "batch_size": DEFAULT_BATCH_SIZE, "cpu_delay": 0.0, "bandwidth_kbps": 0}, # Default
        # "bandwidth_kbps": 0  # Disabled: no network simulation in default mode
    }
else:
    # Hardware Resource Profiles for 10 Heterogeneous Clients
    CLIENT_RESOURCE_PROFILES = {
        0: {"name": "High-End Node",    "batch_size": 128, "cpu_delay": 0.0, "bandwidth_kbps": 100000}, # Fiber / Wi-Fi 6    # Memory: high,  CPU: fast, Network: fast
        1: {"name": "Compute-Bound",   "batch_size": 128, "cpu_delay": 0.3, "bandwidth_kbps": 100000}, # Slow CPU            # Memory: high,  CPU: slow, Network: fast
        2: {"name": "Network-Bound",   "batch_size": 128, "cpu_delay": 0.0, "bandwidth_kbps": 128}, # 2G/3G Network          # Memory: high,  CPU: fast, Network: slow
        3: {"name": "Memory-Bound",    "batch_size": 16,  "cpu_delay": 0.0, "bandwidth_kbps": 100000}, # Low RAM             # Memory: small, CPU: fast, Network: fast
        4: {"name": "Compute+Network", "batch_size": 128, "cpu_delay": 0.3, "bandwidth_kbps": 128},                          # Memory: high,  CPU: slow, Network: slow
        5: {"name": "Network+Memory",  "batch_size": 16,  "cpu_delay": 0.0, "bandwidth_kbps": 128},                          # Memory: small, CPU: fast, Network: slow
        6: {"name": "Compute+Memory",  "batch_size": 16,  "cpu_delay": 0.3, "bandwidth_kbps": 100000},                       # Memory: small, CPU: slow, Network: fast
        7: {"name": "Constrained Edge", "batch_size": 16,  "cpu_delay": 0.5, "bandwidth_kbps": 64}, # Extreme limit          # Memory: small, CPU: slow, Network: extremely slow
        8: {"name": "Mid-Tier Device PHONE", "batch_size": 64,  "cpu_delay": 0.1, "bandwidth_kbps": 10000},                  # Memory: moderate, CPU: moderate, Network: moderate
        9: {"name": "Mid-Tier Device TABLET", "batch_size": 64,  "cpu_delay": 0.1, "bandwidth_kbps": 10000},                 # Memory: moderate, CPU: moderate, Network: moderate
    }

def calculate_network_latency(cid: int, payload_bytes: int = 155000) -> float:
    """Calculates latency based on CPU speed (compute slowdown) and Network Bandwidth (upload/download latency)."""
    profile = CLIENT_RESOURCE_PROFILES.get(cid, CLIENT_RESOURCE_PROFILES[0])

    # 1. Bandwidth Delay (Upload + Download Latency)
    if profile["bandwidth_kbps"] > 0:
        bandwidth_bytes_per_sec = (profile["bandwidth_kbps"] * 1024) / 8
        network_delay = (payload_bytes * 2) / bandwidth_bytes_per_sec # Upstream + Downstream
    else:
        network_delay = 0.0

    # 2. CPU Delay per epoch step
    cpu_delay = profile["cpu_delay"]

    return network_delay + cpu_delay
# =======================================================

def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)
# =======================================================


    #"""Simple MLP Baseline for Continuous & Encoded Tabular Streams."""

"""#
class TabularNet(nn.Module):
    def __init__(self, num_features=NUM_FEATURES, num_classes=NUM_CLASSES):
        super(TabularNet, self).__init__()
        self.fc1 = nn.Linear(num_features, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
"""


# Fully generic PyTorch model that adapts to ANY tabular dataset
class TabularNet(nn.Module):
    """Dynamic MLP Baseline for Continuous & Encoded Tabular Streams."""
    def __init__(self, input_dim: int, num_classes: int):
    #""" def __init__(self, input_dim=NUM_FEATURES, num_classes=NUM_CLASSES):
        super(TabularNet, self).__init__()
        debug_print(f"---- task: input_dim= {input_dim} num_classes={num_classes}\n")
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def load_data(data_path: str, batch_size: int = DEFAULT_BATCH_SIZE):
    start_time = time.time()
    """Loads localized .pt tabular dataset using client-specific batch size."""
    print(f"📂 [DISK] Loading local dataset: {data_path} (Batch Size: {batch_size})...")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset path {data_path} not found.")

    data = torch.load(data_path, weights_only=True)
    
    # Ensure inputs are 2D tensors [batch_size, num_features]
    x_data = data["x"].float()
    y_data = data["y"].long()
    
    full_dataset = TensorDataset(x_data, y_data)
    
    train_size = int(TRAINING_DATA_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    print(f"📦 [DATA LOADED] Training samples= {train_size}, Validation samples= {val_size}.")
    
    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_ds, batch_size=batch_size)
    
    end_time = time.time()
    print(f"✅ [RAM] Ready in {end_time - start_time:.2f} seconds. Samples -> Train: {train_size}, Val: {val_size}") # Track loading speed

    return trainloader, valloader
# =======================================================

