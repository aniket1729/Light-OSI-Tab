import torch, os

DATASET_NAME = "AgriYield"
DATA_DIR = "./data_crop"
DATASET_NAME = "TrafficVol"
DATA_DIR = "./data_traffic"
DATASET_NAME = "Greenhouse"
DATA_DIR = "./data_greenhouse"

print(f"--- For {DATASET_NAME}, Checking Client Dataset Class Distributions ---")
for i in range(1, 11):
    path = os.path.join(DATA_DIR, f"client_{i:02d}.pt")
    if os.path.exists(path):
        d = torch.load(path, weights_only=True)
        y = d["y"]
        counts = [int((y == c).sum()) for c in range(3)]
        print(f"Client {i:02d}: Class counts = {counts} | Total = {len(y)}")
