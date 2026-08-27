import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import tomllib
from config import NUM_CLIENTS, NUM_CLASSES, CSV_FILE_PATH, DATA_DIR, CLIENT_FILE_PREFIX, EVAL_FILE, META_FILE_PATH, ACTIVE_CONFIG
from task import debug_print

# ==========================================
# ⚙️ CONFIGURATION & CONTROL PANEL
# ==========================================
MIN_TARGET_ROWS = 5000     # Upsampled for 10-20 client scalability
ALPHA = 0.5                # Dirichlet parameter (Lower = More Non-IID skew)
GLOBAL_TEST_RATIO = 0.15   # 15% reserved for final server inference (model_predict.py)
# ==========================================
# [tool.flwr.app.config]
# num-clients = 10
#with open("pyproject.toml", "rb") as f:
#    config = tomllib.load(f)
#NUM_CLIENTS = config["tool"]["flwr"]["app"]["config"]["num-clients"]
# ==========================================

os.makedirs(DATA_DIR, exist_ok=True)

def preprocess_dataset(csv_path, min_target_rows):
    """Loads CSV, handles dates/missing values, encodes features, upsamples, and returns standardized arrays."""

    target_column_name = ACTIVE_CONFIG["target_column"]

    df = pd.read_csv(csv_path, keep_default_na=False)
    #"""#
    if "irrigation_type" in df.columns:
        debug_print("**** col: irrigation_type")
        debug_print(f"Unique values = {df["irrigation_type"].unique()}.")
        debug_print(f"Missing values = {df[["irrigation_type"]].isna().sum()}.")
        debug_print(f"Value counts = {df["irrigation_type"].value_counts(dropna=False)}.")
        debug_print("--------")
    
    if "crop_disease_status" in df.columns:
        debug_print("**** col: crop_disease_status")
        debug_print(f"Unique values = {df["crop_disease_status"].unique()}.")
        debug_print(f"Missing values = {df[["crop_disease_status"]].isna().sum()}.")
        debug_print(f"Value counts = {df["crop_disease_status"].value_counts(dropna=False)}.")
        debug_print("--------")
    
    if "holiday" in df.columns:
        debug_print("**** col: holiday")
        debug_print(f"Unique values = {df["holiday"].unique()}.")
        debug_print(f"Missing values = {df[["holiday"]].isna().sum()}.")
        debug_print(f"Value counts = {df["holiday"].value_counts(dropna=False)}.")
        debug_print("--------")
    #"""
    # 0. Check raw missing values
    print("🔍 Checking raw missing values per column:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "   ∟ No missing values found!")
    #""" print(".\n")

    # 1. Dataset-specific column handling & Feature Engineering for Time-Series / Date Columns
    if "date_time" in df.columns: # Metro Interstate Traffic Dataset
        df["date_time"] = pd.to_datetime(df["date_time"])
        df["hour"] = df["date_time"].dt.hour
        df["day_of_week"] = df["date_time"].dt.dayofweek
        df["month"] = df["date_time"].dt.month

    """# Fill missing values in categorical columns
    if "irrigation_type" in df.columns:
        df["irrigation_type"] = df["irrigation_type"].fillna("Unknown")
    if "crop_disease_status" in df.columns:
        df["crop_disease_status"] = df["crop_disease_status"].fillna("None")
    """
    # Generic categorical fill Missing Value for ANY dataset
    cat_cols = df.select_dtypes(include=['object', 'string', 'category']).columns
    df[cat_cols] = df[cat_cols].fillna("None")

    # 2. Extract feature columns (Drop identifiers, raw dates, and target)
    # Drop non-predictive metadata, string identifiers, and raw date strings
    drop_cols = ACTIVE_CONFIG["drop_columns"] + [target_column_name]    # Dropped from X because it is converted into target y
    feature_df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # 3. One-Hot Encode remaining object/categorical columns with modern pandas compatibility (region, irrigation_type, fertilizer_type, crop_disease_status, etc.)
    #feature_df = pd.get_dummies(feature_df, columns=feature_df.select_dtypes(include=['object']).columns, drop_first=True)
    # Modern Pandas compatibility for categorical column selection
    cat_cols_remaining = feature_df.select_dtypes(include=['object', 'string', 'category']).columns
    #--- feature_df = pd.get_dummies(feature_df, columns=cat_cols_remaining, drop_first=True)
    feature_df = pd.get_dummies(feature_df, columns=cat_cols_remaining, drop_first=True, dtype=np.float32)

    # 4. Verify post-processing missing values
    print(f"🔍 Post-processing missing values check: {feature_df.isnull().sum().sum()} NaNs remaining.")
    
    # 5. Standardize continuous numerical features
    scaler = StandardScaler()
    #--- X_scaled = scaler.fit_transform(feature_df.values).astype(np.float32)
    X_scaled = scaler.fit_transform(feature_df).astype(np.float32)

    # 6. Discretize target column into NUM_CLASSES Equi-depth target classes (i.e. 3 => 0: Low, 1: Med, 2: High)
    y = pd.qcut(df[target_column_name], q=NUM_CLASSES, labels=list(range(NUM_CLASSES))).astype(int).values

    # 7. Statistical Upsampling (if small dataset)
    if len(X_scaled) < min_target_rows:
        X_scaled, y = resample(X_scaled, y, n_samples=min_target_rows, random_state=42, stratify=y)
        # Add tiny Gaussian noise to continuous features to avoid exact duplicate rows
        noise = np.random.normal(0, 0.01, X_scaled.shape).astype(np.float32)
        X_scaled += noise

    num_samples, num_features = X_scaled.shape
    # Note the total number of features created (e.g., if one-hot encoding expands your columns to 18 features).

    meta_info = {
        "dataset_name": ACTIVE_CONFIG["name"],
        #"num_features": X_train.shape[1],  # Dynamically extracted after encoding!
        "num_features": num_features,
        #"num_classes": len(np.unique(y)),
        "num_classes": NUM_CLASSES,
        "target_column": target_column_name,
        "num_samples": num_samples,
    }
    torch.save(meta_info, META_FILE_PATH)

    # 📢 IMPORTANT FOR TASK.PY
    print("\n" + "="*50)
    print("📊 DATASET DIMENSIONS SUMMARY FOR task.py")
    print(f"   ∟ Total Samples (Rows) : {num_samples}")
    print(f"   ∟ NUM_FEATURES (Cols)  : {num_features}")
    print(f"   ∟ Target Classes       : {NUM_CLASSES}")
    print(f"   ∟ Unique Classes Found : {len(np.unique(y))}")
    print(f"🎯 --> META File saved at {META_FILE_PATH}")
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
        for k in range(NUM_CLASSES):
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
    #""" print("\n")
    dataset_name = ACTIVE_CONFIG["name"]
    print(f"==== Preparing Client Dataset chunks from Dataset= '{dataset_name}' ====")
    if not os.path.exists(CSV_FILE_PATH):
        raise FileNotFoundError(f"Dataset not found at {CSV_FILE_PATH}. Please place your Kaggle CSV there.")

    X, y, num_features = preprocess_dataset(CSV_FILE_PATH, MIN_TARGET_ROWS)
    print(f"✅ Data Preprocessed: {len(X)} rows with {num_features} input features.")

    # 1. Split Global Test Set
    num_samples = len(X)
    indices = np.random.permutation(num_samples)
    split_idx = int(num_samples * GLOBAL_TEST_RATIO)
    
    global_test_idx = indices[:split_idx]
    client_pool_idx = indices[split_idx:]
    print(f"✅ Split sizes: GlobalTest= {len(global_test_idx)} ClientPool= {len(client_pool_idx)}.")

    # Save Global Test Set for model_predict.py
    save_pt_file(X[global_test_idx], y[global_test_idx], EVAL_FILE)
    print(f"  ∟ Created Global Server Test Set: {EVAL_FILE} ({len(global_test_idx)} rows)")

    # 2. Partition Client Pool into Non-IID Shards using Dirichlet Distribution
    y_client_pool = y[client_pool_idx]
    dirichlet_splits = dirichlet_non_iid_split(y_client_pool, NUM_CLIENTS, ALPHA)

    print(f"\n--- Generating {NUM_CLIENTS} Non-IID Client Partitions (Dirichlet alpha={ALPHA}) ---")
    for i, relative_indices in enumerate(dirichlet_splits):
        actual_indices = client_pool_idx[relative_indices]
        save_pt_file(X[actual_indices], y[actual_indices], f"{CLIENT_FILE_PREFIX}_{i+1:02d}.pt")
        
        # Label distribution report
        client_y = y[actual_indices]
        counts = [np.sum(client_y == c) for c in range(NUM_CLASSES)]
        print(f"  ∟ Created Non-IID {CLIENT_FILE_PREFIX}_{i+1:02d}.pt: {len(actual_indices)} rows | Class Counts (0,1,2): {counts}")

    print(f"\n✨ Dataset prepared. All .pt files successfully output to '{DATA_DIR}'.")

if __name__ == "__main__":
    main()
