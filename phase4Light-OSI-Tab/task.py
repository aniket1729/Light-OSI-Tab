# task.py
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from config import RESOURCE_MODE, LATENT_DIM, NUM_CLASSES

DEFAULT_BATCH_SIZE = 32
TRAINING_DATA_SPLIT = 0.8

if not RESOURCE_MODE:
    CLIENT_RESOURCE_PROFILES = {
        0: {"name": "Default", "batch_size": DEFAULT_BATCH_SIZE, "cpu_delay": 0.0, "bandwidth_kbps": 0},
    }
else:
    CLIENT_RESOURCE_PROFILES = {
        0: {"name": "High-End Node",    "batch_size": 128, "cpu_delay": 0.0, "bandwidth_kbps": 100000},
        1: {"name": "Compute-Bound",   "batch_size": 128, "cpu_delay": 0.3, "bandwidth_kbps": 100000},
        2: {"name": "Network-Bound",   "batch_size": 128, "cpu_delay": 0.0, "bandwidth_kbps": 128},
        3: {"name": "Memory-Bound",    "batch_size": 16,  "cpu_delay": 0.0, "bandwidth_kbps": 100000},
        4: {"name": "Compute+Network", "batch_size": 128, "cpu_delay": 0.3, "bandwidth_kbps": 128},
        5: {"name": "Network+Memory",  "batch_size": 16,  "cpu_delay": 0.0, "bandwidth_kbps": 128},
        6: {"name": "Compute+Memory",  "batch_size": 16,  "cpu_delay": 0.3, "bandwidth_kbps": 100000},
        7: {"name": "Constrained Edge", "batch_size": 16,  "cpu_delay": 0.5, "bandwidth_kbps": 64},
        8: {"name": "Mid-Tier Device PHONE", "batch_size": 64,  "cpu_delay": 0.1, "bandwidth_kbps": 10000},
        9: {"name": "Mid-Tier Device TABLET", "batch_size": 64,  "cpu_delay": 0.1, "bandwidth_kbps": 10000},
    }

def calculate_network_latency(cid: int, payload_bytes: int = 155000) -> float:
    profile = CLIENT_RESOURCE_PROFILES.get(cid, CLIENT_RESOURCE_PROFILES[0])
    if profile["bandwidth_kbps"] > 0:
        bandwidth_bytes_per_sec = (profile["bandwidth_kbps"] * 1024) / 8
        return (payload_bytes * 2) / bandwidth_bytes_per_sec
    return 0.0

class TabularNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
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

class Tabular1DVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = LATENT_DIM, num_classes: int = NUM_CLASSES):
        super(Tabular1DVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Class embedding layer to scale class signal up to feature magnitude
        self.label_emb = nn.Embedding(num_classes, 16)

        # Encoder with LayerNorm (stable across non-IID partitions)
        self.encoder_fc = nn.Sequential(
            nn.Linear(input_dim + 16, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.SiLU()
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        
        # Decoder with LayerNorm (prevents server evaluation skew)
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim + 16, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, input_dim)
        )

    def encode(self, x, y):
        c = self.label_emb(y)
        inputs = torch.cat([x, c], dim=1)
        h = self.encoder_fc(inputs)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        c = self.label_emb(y)
        inputs = torch.cat([z, c], dim=1)
        return self.decoder_fc(inputs)

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, y)
        return recon_x, mu, logvar

def vae_loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction='mean')
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    # Slightly higher KL annealing factor for distinct cluster separation
    return MSE + (0.02 * KLD)

def load_data(data_path: str, batch_size: int = DEFAULT_BATCH_SIZE):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset path {data_path} not found.")

    data = torch.load(data_path, weights_only=True)
    x_data = data["x"].float()
    y_data = data["y"].long()

    # Standardize features across continuous columns
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
