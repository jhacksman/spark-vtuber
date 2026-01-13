#!/bin/bash
# vLLM Server Startup Script for DGX Spark
# Usage: ./vllm-serve.sh <model_name> [port]

set -e

# Determine script directory and find vLLM install directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find the vLLM install directory
# If running from helpers/, look for ../../../vllm-install
# If running from install dir, use current dir
if [[ -f "$SCRIPT_DIR/vllm_env.sh" ]]; then
    # Running from install directory
    VLLM_INSTALL_DIR="$SCRIPT_DIR"
elif [[ -f "$SCRIPT_DIR/../../../vllm-install/vllm_env.sh" ]]; then
    # Running from scripts/vllm/helpers/
    VLLM_INSTALL_DIR="$(cd "$SCRIPT_DIR/../../../vllm-install" && pwd)"
else
    echo "ERROR: Cannot find vllm_env.sh"
    echo "Expected locations:"
    echo "  - $SCRIPT_DIR/vllm_env.sh (if running from install dir)"
    echo "  - $SCRIPT_DIR/../../../vllm-install/vllm_env.sh (if running from repo)"
    echo ""
    echo "Make sure vLLM is installed. Run:"
    echo "  ./scripts/vllm/install_vllm.sh --install-dir ./vllm-install"
    exit 1
fi

# Configuration
MODEL="${1:-Qwen/Qwen2.5-0.5B-Instruct}"
PORT="${2:-8000}"

# Convert relative model path to absolute path (relative to where user ran the command)
# This is needed because we cd to VLLM_DIR before running vLLM
if [[ "$MODEL" == ./* ]] || [[ "$MODEL" == ../* ]]; then
    # Save the original working directory
    ORIGINAL_PWD="$(pwd)"
    MODEL="$(cd "$ORIGINAL_PWD" && realpath "$MODEL")"
    echo "INFO: Converted relative path to absolute: $MODEL"
fi
VLLM_DIR="$VLLM_INSTALL_DIR/vllm"
ENV_SCRIPT="$VLLM_INSTALL_DIR/vllm_env.sh"
PID_FILE="$SCRIPT_DIR/.vllm-server.pid"
LOG_FILE="$SCRIPT_DIR/vllm-server.log"
MAX_WAIT_SECONDS=300  # 5 minutes max wait for model loading

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

# Clear problematic environment variables that can cause issues on Blackwell
# VLLM_FLASH_ATTN_VERSION expects an integer but some configs set it to "skip"
unset VLLM_FLASH_ATTN_VERSION

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
echo "OK: Waiting for server to be ready (this may take a few minutes for large models)..."

# Wait for server to be ready by checking the health endpoint
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT_SECONDS ]; do
    # Check if process is still running
    if ! ps -p $(cat "$PID_FILE") > /dev/null 2>&1; then
        echo ""
        echo "ERROR: Server process died. Check logs: $LOG_FILE"
        echo "Last 20 lines of log:"
        tail -20 "$LOG_FILE"
        rm -f "$PID_FILE"
        exit 1
    fi
    
    # Check if server is responding
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo ""
        echo "OK: Server is ready!"
        echo ""
        echo "Test with: curl http://localhost:$PORT/v1/models"
        echo "View logs: tail -f $LOG_FILE"
        echo "Stop server: ./vllm-stop.sh"
        exit 0
    fi
    
    # Show progress every 10 seconds
    if [ $((WAIT_COUNT % 10)) -eq 0 ] && [ $WAIT_COUNT -gt 0 ]; then
        echo "  Still waiting... (${WAIT_COUNT}s elapsed)"
    fi
    
    sleep 1
    WAIT_COUNT=$((WAIT_COUNT + 1))
done

echo ""
echo "ERROR: Server did not become ready within ${MAX_WAIT_SECONDS} seconds"
echo "The server process is still running (PID: $(cat $PID_FILE))"
echo "Check logs: tail -f $LOG_FILE"
exit 1
