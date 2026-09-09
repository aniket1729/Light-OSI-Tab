import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ==========================================
# ⚙️ CONFIGURATION & CONTROL PANEL
# ==========================================
CSV_FILE_PATH = "./temp_data/Smart_Farming_Crop_Yield_2024.csv"
DATA_DIR = "./data"
NUM_SHARDS = 3            # Number of IID client shards
GLOBAL_TEST_RATIO = 0.15  # 15% reserved exclusively for final server inference (model_predict.py)
NUM_YIELD_CLASSES = 3     # Low (0), Medium (1), High (2)

# Non-IID Groups matching the MNIST crop_groups approach
CROP_GROUPS = [
    (["Wheat", "Rice"], "WheatRice"),
    (["Maize", "Cotton"], "MaizeCotton"),
    (["Soybean"], "Soybean")
]
# ==========================================

os.makedirs(DATA_DIR, exist_ok=True)

def preprocess_smart_farming_data(csv_path):
    """Loads CSV, verifies missing values, encodes features, reports exact shapes, and returns standardized arrays."""
    df = pd.read_csv(csv_path)

    # 🔍 DEBUG PRINT 1: Check raw missing values in original CSV
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
    
    # Keep track of crop_type for non-IID grouping, then drop or encode (One-Hot Encode remaining categoricals)
    crop_types = df["crop_type"].values
    
    # One-Hot Encode remaining object/categorical columns with modern pandas compatibility (region, irrigation_type, fertilizer_type, crop_disease_status, etc.)
    #feature_df = pd.get_dummies(feature_df, columns=feature_df.select_dtypes(include=['object']).columns, drop_first=True)
    # Modern Pandas compatibility for categorical column selection
    cat_cols = feature_df.select_dtypes(include=['object', 'string', 'category']).columns
    feature_df = pd.get_dummies(feature_df, columns=cat_cols, drop_first=True)
    
    # 🔍 DEBUG PRINT 2: Verify post-processing missing values
    print(f"🔍 Post-processing missing values check: {feature_df.isnull().sum().sum()} NaNs remaining.")

    # 3. Standardize continuous numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df.values).astype(np.float32)

    # 4. Discretize target column 'yield_kg_per_hectare' into 3 target classes (0: Low, 1: Med, 2: High)
    y = pd.qcut(df["yield_kg_per_hectare"], q=NUM_YIELD_CLASSES, labels=[0, 1, 2]).astype(int).values

    num_samples, num_features = X_scaled.shape
    # Note the total number of features created (e.g., if one-hot encoding expands your columns to 18 features, set NUM_FEATURES = 18 in task.py).

    # 📢 IMPORTANT FOR TASK.PY
    print("\n" + "="*50)
    print("📊 DATASET DIMENSIONS SUMMARY FOR task.py")
    print(f"   ∟ Total Samples (Rows) : {num_samples}")
    print(f"   ∟ NUM_FEATURES (Cols)  : {num_features}  <-- Set this in task.py!")
    print(f"   ∟ Target Classes       : {NUM_YIELD_CLASSES}  <-- Set NUM_CLASSES in task.py!")
    print("="*50 + "\n")

    return X_scaled, y, crop_types, num_features

def save_pt_file(x, y, name):
    """Utility to format and save x, y tensors into a single .pt file."""
    tensor_data = {
        "x": torch.tensor(x, dtype=torch.float32),
        "y": torch.tensor(y, dtype=torch.long)
    }
    file_path = os.path.join(DATA_DIR, name)
    torch.save(tensor_data, file_path)
    print(f"✅ Saved: {name} ({x.shape[0]} samples)")

def main():
    print("\n")
    if not os.path.exists(CSV_FILE_PATH):
        raise FileNotFoundError(f"Dataset not found at {CSV_FILE_PATH}. Please place your Kaggle CSV there.")

    X, y, crop_types, num_features = preprocess_smart_farming_data(CSV_FILE_PATH)
    print(f"✅ Data Preprocessed: {len(X)} rows with {num_features} input features.")

    # ==========================================
    # STEP 1: Reserve Global Server Test Set (Unseen Data for Final Evaluation)
    # ==========================================
    num_samples = len(X)
    indices = np.random.permutation(num_samples)
    split_idx = int(num_samples * GLOBAL_TEST_RATIO)

    global_test_idx, client_pool_idx = indices[:split_idx], indices[split_idx:]
    print(f"✅ Split sizes: GlobalTest= {len(global_test_idx)} ClientPool= {len(client_pool_idx)}.")

    # Save Global Test Set for model_predict.py
    save_pt_file(X[global_test_idx], y[global_test_idx], "global_test.pt")
    print(f"🎯 Created Global Server Test Set: global_test.pt ({len(global_test_idx)} rows)")

    # ==========================================
    # STEP 2: Generate IID Client Partitions (set01.pt, set02.pt, set03.pt)
    # ==========================================
    print("\n--- Generating IID Client Partitions ---")
    shard_size = len(client_pool_idx) // NUM_SHARDS
    print(f"✅ Shard sizes: Shards={NUM_SHARDS} for ClientSize= {shard_size}.")


    for i in range(NUM_SHARDS):
        # Assign remainder samples to the last client so no rows are lost
        if i == NUM_SHARDS - 1:
            sh_indices = client_pool_idx[i * shard_size :]
        else:
            sh_indices = client_pool_idx[i * shard_size : (i + 1) * shard_size]
        save_pt_file(X[sh_indices], y[sh_indices], f"set{i+1:02d}.pt")
        print(f"  ∟ Created IID Client File: set{i+1:02d}.pt ({len(sh_indices)} rows)")

    # ==========================================
    # STEP 3: Generate Non-IID Crop Group Partitions (typWheatRice.pt, typMaizeCotton.pt, etc.)
    # ==========================================
    print("\n--- Generating Non-IID Crop Group Partitions ---")
    for crops, suffix in CROP_GROUPS:
        # Mask matching crops
        crop_mask = np.isin(crop_types, crops)
        
        # Intersect with client pool indices (so global_test remains unseen)
        group_indices = np.intersect1d(client_pool_idx, np.where(crop_mask)[0])

        save_pt_file(X[group_indices], y[group_indices], f"typ{suffix}.pt")
        print(f"  ∟ Created Non-IID Client File: typ{suffix}.pt ({len(group_indices)} rows)")

    print(f"\n✨ All .pt files successfully output to '{DATA_DIR}'.")
    print(f"\n✨ Data prep completed. Clients receive whole partition files, while task.py splits them 80:20 internally!")

if __name__ == "__main__":
    main()
