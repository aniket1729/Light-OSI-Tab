# task.py
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from config import RESOURCE_MODE, NUM_CLASSES

DEFAULT_BATCH_SIZE = 32
TRAINING_DATA_SPLIT = 0.8

if not RESOURCE_MODE:
    CLIENT_RESOURCE_PROFILES = {
        0: {"name": "Uniform Edge Node", "batch_size": DEFAULT_BATCH_SIZE, "cpu_delay": 0.0, "bandwidth_kbps": 0},
    }
else:
    CLIENT_RESOURCE_PROFILES = {
        0: {"name": "High-End Node",         "batch_size": 128, "cpu_delay": 0.0, "bandwidth_kbps": 100000},
        1: {"name": "Compute-Bound",        "batch_size": 128, "cpu_delay": 0.3, "bandwidth_kbps": 100000},
        2: {"name": "Network-Bound",        "batch_size": 128, "cpu_delay": 0.0, "bandwidth_kbps": 128},
        3: {"name": "Memory-Bound",         "batch_size": 16,  "cpu_delay": 0.0, "bandwidth_kbps": 100000},
        4: {"name": "Compute+Network",      "batch_size": 128, "cpu_delay": 0.3, "bandwidth_kbps": 128},
        5: {"name": "Network+Memory",       "batch_size": 16,  "cpu_delay": 0.0, "bandwidth_kbps": 128},
        6: {"name": "Compute+Memory",       "batch_size": 16,  "cpu_delay": 0.3, "bandwidth_kbps": 100000},
        7: {"name": "Constrained Edge",     "batch_size": 16,  "cpu_delay": 0.5, "bandwidth_kbps": 64},
        8: {"name": "Mid-Tier Device PHONE", "batch_size": 64,  "cpu_delay": 0.1, "bandwidth_kbps": 10000},
        9: {"name": "Mid-Tier Device TABLET","batch_size": 64,  "cpu_delay": 0.1, "bandwidth_kbps": 10000},
    }

def calculate_network_latency(cid: int, payload_bytes: int) -> float:
    profile_id = cid if RESOURCE_MODE else 0
    profile = CLIENT_RESOURCE_PROFILES.get(profile_id, CLIENT_RESOURCE_PROFILES[0])
    if profile["bandwidth_kbps"] > 0:
        bandwidth_bytes_per_sec = (profile["bandwidth_kbps"] * 1024) / 8
        return (payload_bytes * 2) / bandwidth_bytes_per_sec
    return 0.0

class TabularNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = NUM_CLASSES):
        super(TabularNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.ln1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 64)
        self.ln2 = nn.LayerNorm(64)
        self.out = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.ln1(self.fc1(x)))
        h = self.dropout(h)
        h = self.act(self.ln2(self.fc2(h)))
        return self.out(h)

def debug_print(msg: str):
    print(msg)

def load_data(data_path: str, batch_size: int = DEFAULT_BATCH_SIZE):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset path {data_path} not found.")

    data = torch.load(data_path, weights_only=True)
    x_data = data["x"].float()
    y_data = data["y"].long()

    # Standardize features locally across continuous columns
    mean = x_data.mean(dim=0, keepdim=True)
    std = x_data.std(dim=0, keepdim=True) + 1e-7
    x_data = (x_data - mean) / std

    full_dataset = TensorDataset(x_data, y_data)
    train_size = int(TRAINING_DATA_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_ds, batch_size=batch_size)
    return trainloader, valloader
