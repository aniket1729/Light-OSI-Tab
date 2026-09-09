#!/bin/bash



LOG_DIR="./logs/"
# Create directory if it doesn't exist
mkdir -p "$LOG_DIR"
echo "Log directory is ready at: $LOG_DIR"

# Port configuration
SUPERLINK_IP="127.0.0.1:9092"

# Extract DATA_DIR dynamically from config.py
DATA_DIR=$(python3 -c "from config import DATA_DIR; print(DATA_DIR)")
NUM_CLIENTS=$(python3 -c "from config import NUM_CLIENTS; print(NUM_CLIENTS)")
CLIENT_FILE_PREFIX=$(python3 -c "from config import CLIENT_FILE_PREFIX; print(CLIENT_FILE_PREFIX)")
#""" # Extract DATA_DIR dynamically from pyproject.toml
# DATA_DIR=$(python3 -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['tool']['flwr']['app']['config']['data_dir'])")
# NUM_CLIENTS=$(python3 -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["tool"]["flwr"]["app"]["config"]["num-clients"])')

echo "=========================================================================="
echo "🚀 Launching Flower Supernodes: ${NUM_CLIENTS} using DATA_DIR: ${DATA_DIR}"
echo "=========================================================================="
#""" for i in $(seq 0 $((NUM_CLIENTS - 1))); do
for i in $(seq -f "%02g" 1 $NUM_CLIENTS); do
  API_PORT=$((9094 + 10#$i))
  DATA_FILE="${DATA_DIR}${CLIENT_FILE_PREFIX}_${i}.pt"
  LOG_FILE="${LOG_DIR}${CLIENT_FILE_PREFIX}_${i}.log"
  echo "" > "$LOG_FILE"

  echo "  ∟ Starting Supernode $i on API port $API_PORT using $DATA_FILE"

  FLWR_CLIENTAPP_IO=client_app:app flower-supernode \
    --insecure \
    --superlink $SUPERLINK_IP \
	--node-config "cid=$((10#$i - 1))" \
    --clientappio-api-address "127.0.0.1:$API_PORT" > "$LOG_FILE" 2>&1 &
done

echo "✅ All $NUM_CLIENTS Supernodes running in background! Logs saved to ${LOG_DIR}${CLIENT_FILE_PREFIX}_XX.log."
