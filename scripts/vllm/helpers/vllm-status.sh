#!/bin/bash
# vLLM Server Status Script for DGX Spark

# Determine installation directory (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PID_FILE="$SCRIPT_DIR/.vllm-server.pid"
LOG_FILE="$SCRIPT_DIR/vllm-server.log"

echo "=" | tr '=' '-' | head -c 70 && echo
echo "vLLM Server Status on DGX Spark"
echo "=" | tr '=' '-' | head -c 70 && echo

if [ ! -f "$PID_FILE" ]; then
    echo "Status: NOT RUNNING (no PID file found)"
    exit 0
fi

PID=$(cat "$PID_FILE")

if ! ps -p $PID > /dev/null 2>&1; then
    echo "Status: NOT RUNNING (stale PID file)"
    echo "Cleaning up PID file..."
    rm -f "$PID_FILE"
    exit 0
fi

echo "Status: RUNNING"
echo "PID: $PID"
echo "Started: $(ps -p $PID -o lstart= 2>/dev/null || echo 'Unknown')"
echo "CPU: $(ps -p $PID -o %cpu= 2>/dev/null || echo 'N/A')%"
echo "Memory: $(ps -p $PID -o %mem= 2>/dev/null || echo 'N/A')%"
echo ""

# Check if log file exists and show last few lines
if [ -f "$LOG_FILE" ]; then
    echo "Recent log entries (last 10 lines):"
    echo "-" | tr '-' '-' | head -c 70 && echo
    tail -n 10 "$LOG_FILE"
else
    echo "Log file not found: $LOG_FILE"
fi

echo ""
echo "=" | tr '=' '-' | head -c 70 && echo
