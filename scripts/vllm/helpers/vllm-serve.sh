#!/bin/bash
# vLLM Server Startup Script for DGX Spark
# Usage: ./vllm-serve.sh <model_name> [port]

set -e

# Determine installation directory (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
MODEL="${1:-Qwen/Qwen2.5-0.5B-Instruct}"
PORT="${2:-8000}"
VLLM_DIR="$SCRIPT_DIR/vllm"
ENV_SCRIPT="$SCRIPT_DIR/vllm_env.sh"
PID_FILE="$SCRIPT_DIR/.vllm-server.pid"
LOG_FILE="$SCRIPT_DIR/vllm-server.log"

# Check if server is already running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "ERROR: vLLM server is already running (PID: $PID)"
        echo "Use ./vllm-stop.sh to stop it first"
        exit 1
    fi
fi

# Source environment
source "$ENV_SCRIPT"

echo "=" | tr '=' '-' | head -c 70 && echo
echo "Starting vLLM Server on DGX Spark"
echo "=" | tr '=' '-' | head -c 70 && echo
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Log file: $LOG_FILE"
echo "PID file: $PID_FILE"
echo "=" | tr '=' '-' | head -c 70 && echo

# Start server in background
cd "$VLLM_DIR"
nohup python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port "$PORT" \
    --gpu-memory-utilization 0.9 \
    > "$LOG_FILE" 2>&1 &

# Save PID
echo $! > "$PID_FILE"
echo "OK: Server started with PID: $(cat $PID_FILE)"
echo "OK: Waiting for server to be ready..."

# Wait for server to be ready
sleep 5
if ps -p $(cat "$PID_FILE") > /dev/null 2>&1; then
    echo "OK: Server is running!"
    echo ""
    echo "Test with: curl http://localhost:$PORT/v1/models"
    echo "View logs: tail -f $LOG_FILE"
    echo "Stop server: ./vllm-stop.sh"
else
    echo "ERROR: Server failed to start. Check logs: $LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi
