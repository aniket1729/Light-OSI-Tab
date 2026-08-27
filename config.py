import os

RESOURCE_MODE = False
#RESOURCE_MODE = True

# Number of FL clients
NUM_CLIENTS = 10

DEBUG = False
#DEBUG = True

# ============================================================
# DATASET / EXPERIMENT CONTROL
# ============================================================
# Select one:
DATASET_NAME = "AgriYield-3C"
DATASET_NAME = "TrafficVol-3C"
#DATASET_NAME = "Greenhouse-3C"

# Dictionary mapping each dataset to its raw CSV source & serialized output folder
DATASET_CONFIGS = {
    "AgriYield-3C": {
        "name"         : "AgriYield",
        "input_csv"    : "./temp_data/Smart_Farming_Crop_Yield_2024.csv",
        "target_column": "yield_kg_per_hectare",
        "drop_columns" : ["farm_id", "sensor_id", "sowing_date", "harvest_date", "timestamp"],
        "data_dir"     : "./data_crop/",
        "num_classes"  : 3,
        "class_names"  : ["Low Yield (0)", "Med Yield (1)", "High Yield (2)"],
    },    # https://www.kaggle.com/datasets/atharvasoundankar/smart-farming-sensor-data-for-yield-prediction	https://iotdataset.com/data/smart-farming-yield-prediction-dataset

    "TrafficVol-3C": {
        "name"         : "TrafficVol",
        "input_csv"    : "./temp_data/Metro_Interstate_Traffic_Volume.csv",
        "target_column": "traffic_volume",
        "drop_columns" : ["date_time"],
        "data_dir"     : "./data_traffic/",
        "num_classes"  : 3,
        "class_names"  : ["Low Traffic (0)", "Med Traffic (1)", "High Traffic (2)"],
    },    # https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume    https://doi.org/10.24432/C5X60B

    "Greenhouse-3C": {
        "name"         : "Greenhouse",
        "input_csv"    : "./temp_data/iot_plant_rl_dataset.csv",
        "target_column": "reward",
        "drop_columns" : ["episode_id", "replicate_id", "action_name"],
        "data_dir"     : "./data_greenhouse/",
        "num_classes"  : 3,
        "class_names"  : ["Low Reward (0)", "Med Reward (1)", "High Reward (2)"],
    },    # https://www.kaggle.com/datasets/wisam1985/smart-greenhouse-iot-dataset-for-rl
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

# =====================================================================


# ==========================================
# ⚙️ PREDICTION CONTROL PANEL
# ==========================================
OUTPUT_DIR = "./output/"
MODEL_FILE_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
PLOT_FILE_PATH = os.path.join(OUTPUT_DIR, "fedavg_confusion_matrix.png")
# ==========================================
