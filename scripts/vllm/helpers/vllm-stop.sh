#!/bin/bash
# vLLM Server Stop Script for DGX Spark

# Determine installation directory (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PID_FILE="$SCRIPT_DIR/.vllm-server.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "No vLLM server PID file found. Server may not be running."
    exit 0
fi

PID=$(cat "$PID_FILE")

if ! ps -p $PID > /dev/null 2>&1; then
    echo "vLLM server (PID: $PID) is not running. Cleaning up PID file."
    rm -f "$PID_FILE"
    exit 0
fi

echo "Stopping vLLM server (PID: $PID)..."
kill $PID

# Wait for process to terminate
for i in {1..10}; do
    if ! ps -p $PID > /dev/null 2>&1; then
        echo "OK: Server stopped successfully"
        rm -f "$PID_FILE"
        exit 0
    fi
    sleep 1
done

# Force kill if still running
if ps -p $PID > /dev/null 2>&1; then
    echo "Server did not stop gracefully. Force killing..."
    kill -9 $PID
    sleep 1
    if ! ps -p $PID > /dev/null 2>&1; then
        echo "OK: Server force stopped"
        rm -f "$PID_FILE"
    else
        echo "ERROR: Failed to stop server"
        exit 1
    fi
fi
