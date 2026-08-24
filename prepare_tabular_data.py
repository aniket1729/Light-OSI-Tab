import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import tomllib

# ==========================================
# ⚙️ CONFIGURATION & CONTROL PANEL
# ==========================================
CSV_FILE_PATH = "./temp_data/Smart_Farming_Crop_Yield_2024.csv"
DATA_DIR = "./data/"
TARGET_TOTAL_ROWS = 5000   # Upsampled for 10-20 client scalability
ALPHA = 0.5                # Dirichlet parameter (Lower = More Non-IID skew)
GLOBAL_TEST_RATIO = 0.15   # 15% reserved for final server inference (model_predict.py)
NUM_YIELD_CLASSES = 3      # Low (0), Medium (1), High (2)
# ==========================================
with open("pyproject.toml", "rb") as f:
    config = tomllib.load(f)
NUM_CLIENTS = config["tool"]["flwr"]["app"]["config"]["num-clients"]
# ==========================================

os.makedirs(DATA_DIR, exist_ok=True)

def preprocess_and_upsample_smart_farming_data(csv_path, target_rows):
    """Loads CSV, verifies missing values, encodes features, statistically upsamples to target_rows, reports exact shapes, and returns standardized arrays."""
    df = pd.read_csv(csv_path)

    # 0. Check raw missing values
    print("🔍 Checking raw missing values per column:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "   ∟ No missing values found!")
    print("\n")

    # 1. Fill missing values in categorical columns
    df["irrigation_type"] = df["irrigation_type"].fillna("Unknown")
    df["crop_disease_status"] = df["crop_disease_status"].fillna("None")

    # 2. Extract feature columns (Drop identifiers, dates, and target)
    # Drop non-predictive metadata, string identifiers, and raw date strings
    drop_cols = ["farm_id", "sensor_id", "sowing_date", "harvest_date", "timestamp", "yield_kg_per_hectare"]    # Dropped from X because it is converted into target y
    feature_df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    # 3. One-Hot Encode remaining object/categorical columns with modern pandas compatibility (region, irrigation_type, fertilizer_type, crop_disease_status, etc.)
    #feature_df = pd.get_dummies(feature_df, columns=feature_df.select_dtypes(include=['object']).columns, drop_first=True)
    # Modern Pandas compatibility for categorical column selection
    cat_cols = feature_df.select_dtypes(include=['object', 'string', 'category']).columns
    feature_df = pd.get_dummies(feature_df, columns=cat_cols, drop_first=True)

    # 4. Verify post-processing missing values
    print(f"🔍 Post-processing missing values check: {feature_df.isnull().sum().sum()} NaNs remaining.")
    
    # 5. Standardize continuous numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df.values).astype(np.float32)
    
    # 6. Discretize target column 'yield_kg_per_hectare' into 3 target classes (0: Low, 1: Med, 2: High)
    y = pd.qcut(df["yield_kg_per_hectare"], q=NUM_YIELD_CLASSES, labels=[0, 1, 2]).astype(int).values

    # 7. Statistical Upsampling to reach target_rows
    if len(X_scaled) < target_rows:
        X_scaled, y = resample(X_scaled, y, n_samples=target_rows, random_state=42, stratify=y)
        # Add tiny Gaussian noise to continuous features to avoid exact duplicate rows
        noise = np.random.normal(0, 0.01, X_scaled.shape).astype(np.float32)
        X_scaled += noise

    num_samples, num_features = X_scaled.shape
    # Note the total number of features created (e.g., if one-hot encoding expands your columns to 18 features, set NUM_FEATURES = 18 in task.py).

    # 📢 IMPORTANT FOR TASK.PY
    print("\n" + "="*50)
    print("📊 DATASET DIMENSIONS SUMMARY FOR task.py")
    print(f"   ∟ Total Samples (Rows) : {num_samples}")
    print(f"   ∟ NUM_FEATURES (Cols)  : {num_features}  <-- Set NUM_FEATURES in task.py!")
    print(f"   ∟ Target Classes       : {NUM_YIELD_CLASSES}  <-- Set NUM_CLASSES in task.py!")
    print("="*50 + "\n")

    return X_scaled, y, num_features

def save_pt_file(x, y, name):
    """Utility to format and save x, y tensors into a single .pt file."""
    tensor_data = {
        "x": torch.tensor(x, dtype=torch.float32),
        "y": torch.tensor(y, dtype=torch.long)
    }
    file_path = os.path.join(DATA_DIR, name)
    torch.save(tensor_data, file_path)
    print(f"✅ Saved: {name} ({x.shape[0]} samples)")

def dirichlet_non_iid_split(y, num_clients, alpha):
    """Partitions data indices using Dirichlet Distribution Dir(alpha)."""
    min_size = 0
    client_indices = [[] for _ in range(num_clients)]
    
    while min_size < 10:  # Ensure every client receives at least 10 samples
        client_indices = [[] for _ in range(num_clients)]
        for k in range(NUM_YIELD_CLASSES):
            idx_k = np.where(y == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([p * (len(idx_j) < len(y) / num_clients) for p, idx_j in zip(proportions, client_indices)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            client_indices_split = np.split(idx_k, proportions)
            for i, idx in enumerate(client_indices_split):
                client_indices[i].extend(idx)
        min_size = min([len(idx) for idx in client_indices])
        
    return client_indices

def main():
    print("\n")
    if not os.path.exists(CSV_FILE_PATH):
        raise FileNotFoundError(f"Dataset not found at {CSV_FILE_PATH}. Please place your Kaggle CSV there.")

    X, y, num_features = preprocess_and_upsample_smart_farming_data(CSV_FILE_PATH, TARGET_TOTAL_ROWS)
    print(f"✅ Data Preprocessed: {len(X)} rows with {num_features} input features.")

    # 1. Split Global Test Set
    num_samples = len(X)
    indices = np.random.permutation(num_samples)
    split_idx = int(num_samples * GLOBAL_TEST_RATIO)
    
    global_test_idx = indices[:split_idx]
    client_pool_idx = indices[split_idx:]
    print(f"✅ Split sizes: GlobalTest= {len(global_test_idx)} ClientPool= {len(client_pool_idx)}.")

    # Save Global Test Set for model_predict.py
    save_pt_file(X[global_test_idx], y[global_test_idx], "global_test.pt")
    print(f"🎯 Created Global Server Test Set: global_test.pt ({len(global_test_idx)} rows)")

    # 2. Partition Client Pool into Non-IID Shards using Dirichlet Distribution
    y_client_pool = y[client_pool_idx]
    dirichlet_splits = dirichlet_non_iid_split(y_client_pool, NUM_CLIENTS, ALPHA)

    print(f"\n--- Generating {NUM_CLIENTS} Non-IID Client Partitions (Dirichlet alpha={ALPHA}) ---")
    for i, relative_indices in enumerate(dirichlet_splits):
        actual_indices = client_pool_idx[relative_indices]
        save_pt_file(X[actual_indices], y[actual_indices], f"client_{i+1:02d}.pt")
        
        # Label distribution report
        client_y = y[actual_indices]
        counts = [np.sum(client_y == c) for c in range(NUM_YIELD_CLASSES)]
        print(f"  ∟ Created Non-IID client_{i+1:02d}.pt: {len(actual_indices)} rows | Class Counts (0,1,2): {counts}")

    print(f"\n✨ Dataset prepared. All .pt files successfully output to '{DATA_DIR}'.")

if __name__ == "__main__":
    main()
