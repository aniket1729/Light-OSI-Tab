# task.py (Tabular Baseline Version)
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.utils.data import DataLoader, TensorDataset
#import pandas as pd
#import numpy as np

# ==========================================
# ⚙️ TABULAR TASK CONTROL PANEL
# ==========================================
NUM_FEATURES = 27         # Number of input tabular columns (Adjust to your CSV)
NUM_CLASSES = 3           # Number of target output classes
#NUM_CLASSES = 10         # 0-9 digits
BATCH_SIZE = 32           # How many images per batch
#INPUT_CHANNELS = 1       # 1 for Grayscale (MNIST)
TRAINING_DATA_SPLIT = 0.8 #
# ==========================================

class TabularNet(nn.Module):
    """Simple MLP Baseline for Continuous & Encoded Tabular Streams."""
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

def load_data(data_path: str):
    start_time = time.time()
    """Loads localized .pt tabular dictionary {"x": FloatTensor, "y": LongTensor}."""
    print(f"📂 [DISK] Loading local dataset: {data_path}...")
    data = torch.load(data_path, weights_only=True)
    
    # Ensure inputs are 2D tensors [batch_size, num_features]
    x_data = data["x"].float()
    y_data = data["y"].long()
    
    full_dataset = TensorDataset(x_data, y_data)
    
    train_size = int(TRAINING_DATA_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    print(f"📦 [DATA LOADED] Training samples= {train_size}, Validation samples= {val_size}.")
    
    trainloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    
    end_time = time.time()
    print(f"✅ [RAM] Ready in {end_time - start_time:.2f} seconds.") # Track loading speed

    return trainloader, valloader
