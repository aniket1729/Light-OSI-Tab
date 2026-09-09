import os

# Number of FL clients
NUM_CLIENTS = 10

DEBUG = False

# Hyperparameters
LATENT_DIM = 4              # Kept for standard baseline compatibility
LOCAL_EPOCHS = 100          # Local training epochs for 1-Round FedAvg
SYNTHETIC_SAMPLES_PER_CLASS = 10000

RESOURCE_MODE = True

# FL Method Selection
FL_METHOD_STR = "FedAvg 1 Round [Baseline]" # Standard 1-Round FedAvg / Naive OSFL

# Select one Active Dataset:
DATASET_NAME = "AgriYield-3C"
# DATASET_NAME = "TrafficVol-3C"
# DATASET_NAME = "Greenhouse-3C"

DATASET_CONFIGS = {
    "AgriYield-3C": {
        "name"         : "AgriYield",
        "input_csv"    : "./temp_data/Smart_Farming_Crop_Yield_2024.csv",
        "target_column": "yield_kg_per_hectare",
        "drop_columns" : ["farm_id", "sensor_id", "sowing_date", "harvest_date", "timestamp"],
        "data_dir"     : "./data_crop/",
        "num_classes"  : 3,
        "class_names"  : ["Low Yield (0)", "Med Yield (1)", "High Yield (2)"],
    },
    "TrafficVol-3C": {
        "name"         : "TrafficVol",
        "input_csv"    : "./temp_data/Metro_Interstate_Traffic_Volume.csv",
        "target_column": "traffic_volume",
        "drop_columns" : ["date_time"],
        "data_dir"     : "./data_traffic/",
        "num_classes"  : 3,
        "class_names"  : ["Low Traffic (0)", "Med Traffic (1)", "High Traffic (2)"],
    },
    "Greenhouse-3C": {
        "name"         : "Greenhouse",
        "input_csv"    : "./temp_data/iot_plant_rl_dataset.csv",
        "target_column": "reward",
        "drop_columns" : ["episode_id", "replicate_id", "action_name"],
        "data_dir"     : "./data_greenhouse/",
        "num_classes"  : 3,
        "class_names"  : ["Low Reward (0)", "Med Reward (1)", "High Reward (2)"],
    },
}

# Active Configuration Variables
ACTIVE_CONFIG = DATASET_CONFIGS[DATASET_NAME]

CSV_FILE_PATH = ACTIVE_CONFIG["input_csv"]
DATA_DIR      = ACTIVE_CONFIG["data_dir"]
NUM_CLASSES   = ACTIVE_CONFIG["num_classes"]
CLASS_NAMES   = ACTIVE_CONFIG["class_names"]

# Derived Paths for Preprocessed Tensors
CLIENT_FILE_PREFIX = "client"
EVAL_FILE = "global_test.pt"
META_FILE_PATH = os.path.join(DATA_DIR, "meta.pt")

# Prediction Control Panel
OUTPUT_DIR = "./output/"
MODEL_FILE_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
PLOT_FILE_PATH = os.path.join(OUTPUT_DIR, "fedavg_confusion_matrix.png")
