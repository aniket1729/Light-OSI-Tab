#!/bin/bash

LOGS_DIR="./logs/"
# Create directory if it doesn't exist
mkdir -p "$LOGS_DIR"
echo "Log directory is ready at: $LOGS_DIR"

# Port configuration
SUPERLINK_IP="127.0.0.1:9092"

#NUM_CLIENTS=5
NUM_CLIENTS=$(python3 -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["tool"]["flwr"]["app"]["config"]["num-clients"])')

echo "🚀 Starting $NUM_CLIENTS Flower Supernodes..."
#for i in $(seq 0 $((NUM_CLIENTS - 1))); do
for i in $(seq -f "%02g" 1 $NUM_CLIENTS); do
  API_PORT=$((9094 + 10#$i))
  DATA_FILE="./data/client_${i}.pt"
  echo "" > "$LOGS_DIR/client_${i}.log"

  echo "  ∟ Starting Supernode $i on API port $API_PORT using $DATA_FILE"

  FLWR_CLIENTAPP_IO=client_app:app flower-supernode \
    --insecure \
    --superlink $SUPERLINK_IP \
	--node-config "cid=$((10#$i - 1))" \
    --clientappio-api-address "127.0.0.1:$API_PORT" > "$LOGS_DIR/client_${i}.log" 2>&1 &
done

echo "✅ All $NUM_CLIENTS Supernodes running in background! Logs saved to client_XX.log."
