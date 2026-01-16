#!/bin/bash
################################################################################
# Start All Services Script
#
# Starts vLLM, Fish Speech, and Web Test Server in the correct order.
# Services are started in background and PIDs are tracked for cleanup.
#
# Usage: ./start-all.sh [--reference-audio PATH]
#
# Ports:
#   - vLLM:        8000
#   - Fish Speech: 8843
#   - Web Test:    8844
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PID_DIR="$REPO_ROOT/.pids"
LOG_DIR="$REPO_ROOT/logs"

# Default ports
VLLM_PORT=8000
FISH_PORT=8843
WEB_PORT=8844

# Default model path
MODEL_PATH="$REPO_ROOT/models/qwen3-30b-a3b-awq"

# Reference audio (optional)
REFERENCE_AUDIO=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --reference-audio)
            REFERENCE_AUDIO="$2"
            shift 2
            ;;
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --reference-audio PATH  Path to reference audio for voice cloning"
            echo "  --model PATH            Path to vLLM model (default: models/qwen3-30b-a3b-awq)"
            echo "  --help                  Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create directories
mkdir -p "$PID_DIR" "$LOG_DIR"

echo "========================================================================"
echo "  Starting All Services"
echo "========================================================================"
echo ""

# Check if any services are already running
check_port() {
    local port=$1
    if lsof -i :$port > /dev/null 2>&1; then
        return 0  # Port in use
    fi
    return 1  # Port free
}

if check_port $VLLM_PORT; then
    log_warning "Port $VLLM_PORT already in use (vLLM may be running)"
fi
if check_port $FISH_PORT; then
    log_warning "Port $FISH_PORT already in use (Fish Speech may be running)"
fi
if check_port $WEB_PORT; then
    log_warning "Port $WEB_PORT already in use (Web Test may be running)"
fi

################################################################################
# 1. Start vLLM Server
################################################################################
log_info "Starting vLLM server on port $VLLM_PORT..."

VLLM_INSTALL_DIR="$REPO_ROOT/vllm-install"
if [ ! -d "$VLLM_INSTALL_DIR" ]; then
    log_error "vLLM installation not found at $VLLM_INSTALL_DIR"
    log_info "Run: bash scripts/vllm/install_vllm.sh --install-dir ./vllm-install"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    log_error "Model not found at $MODEL_PATH"
    log_info "Run: bash scripts/download_models.sh"
    exit 1
fi

# Source vLLM environment and start server
(
    cd "$VLLM_INSTALL_DIR/helpers"
    source ./vllm_env.sh
    
    # The vllm-serve.sh script handles PID tracking internally
    # We'll run it in background and capture its PID
    ./vllm-serve.sh "$MODEL_PATH" $VLLM_PORT &
) &
VLLM_LAUNCHER_PID=$!
echo $VLLM_LAUNCHER_PID > "$PID_DIR/vllm-launcher.pid"

log_info "vLLM launcher started (PID: $VLLM_LAUNCHER_PID)"
log_info "Waiting for vLLM to be ready (this takes 4-5 minutes for large models)..."

# Wait for vLLM to be ready
VLLM_WAIT=0
VLLM_MAX_WAIT=720  # 12 minutes (large models on DGX Spark can take longer)
while [ $VLLM_WAIT -lt $VLLM_MAX_WAIT ]; do
    if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
        log_success "vLLM is ready on port $VLLM_PORT"
        break
    fi
    
    # Show progress every 30 seconds
    if [ $((VLLM_WAIT % 30)) -eq 0 ] && [ $VLLM_WAIT -gt 0 ]; then
        log_info "  Still waiting for vLLM... (${VLLM_WAIT}s elapsed)"
    fi
    
    sleep 5
    VLLM_WAIT=$((VLLM_WAIT + 5))
done

if [ $VLLM_WAIT -ge $VLLM_MAX_WAIT ]; then
    log_error "vLLM did not start within ${VLLM_MAX_WAIT} seconds"
    log_info "Check logs: tail -f $VLLM_INSTALL_DIR/helpers/vllm-server.log"
    exit 1
fi

################################################################################
# 2. Start Fish Speech Server
################################################################################
log_info "Starting Fish Speech server on port $FISH_PORT..."

FISH_SCRIPT="$SCRIPT_DIR/fish-speech/fish-speech-serve.sh"
if [ ! -f "$FISH_SCRIPT" ]; then
    log_error "Fish Speech serve script not found at $FISH_SCRIPT"
    exit 1
fi

# Start Fish Speech in background
nohup bash "$FISH_SCRIPT" $FISH_PORT > "$LOG_DIR/fish-speech.log" 2>&1 &
FISH_PID=$!
echo $FISH_PID > "$PID_DIR/fish-speech.pid"

log_info "Fish Speech started (PID: $FISH_PID)"
log_info "Waiting for Fish Speech to be ready..."

# Wait for Fish Speech to be ready
FISH_WAIT=0
FISH_MAX_WAIT=360  # 6 minutes (first run compiles Triton kernels)
while [ $FISH_WAIT -lt $FISH_MAX_WAIT ]; do
    if curl -s "http://localhost:$FISH_PORT/v1/health" > /dev/null 2>&1; then
        log_success "Fish Speech is ready on port $FISH_PORT"
        break
    fi
    
    # Check if process died
    if ! ps -p $FISH_PID > /dev/null 2>&1; then
        log_error "Fish Speech process died. Check logs: $LOG_DIR/fish-speech.log"
        tail -20 "$LOG_DIR/fish-speech.log"
        exit 1
    fi
    
    # Show progress every 30 seconds
    if [ $((FISH_WAIT % 30)) -eq 0 ] && [ $FISH_WAIT -gt 0 ]; then
        log_info "  Still waiting for Fish Speech... (${FISH_WAIT}s elapsed)"
    fi
    
    sleep 5
    FISH_WAIT=$((FISH_WAIT + 5))
done

if [ $FISH_WAIT -ge $FISH_MAX_WAIT ]; then
    log_error "Fish Speech did not start within ${FISH_MAX_WAIT} seconds"
    log_info "Check logs: tail -f $LOG_DIR/fish-speech.log"
    exit 1
fi

################################################################################
# 3. Start Web Test Server
################################################################################
log_info "Starting Web Test server on port $WEB_PORT..."

# Build web server command
WEB_CMD="python $SCRIPT_DIR/web-test/server.py --port $WEB_PORT --vllm-url http://localhost:$VLLM_PORT --tts-url http://localhost:$FISH_PORT"

# Add reference audio if specified
if [ -n "$REFERENCE_AUDIO" ]; then
    if [ -f "$REFERENCE_AUDIO" ]; then
        WEB_CMD="$WEB_CMD --reference-audio $REFERENCE_AUDIO"
        log_info "Using reference audio: $REFERENCE_AUDIO"
    else
        log_warning "Reference audio not found: $REFERENCE_AUDIO"
    fi
elif [ -f "$REPO_ROOT/references/default_voice.wav" ]; then
    # Use default reference audio if available
    WEB_CMD="$WEB_CMD --reference-audio $REPO_ROOT/references/default_voice.wav"
    log_info "Using default reference audio"
fi

# Start web server in background
nohup $WEB_CMD > "$LOG_DIR/web-test.log" 2>&1 &
WEB_PID=$!
echo $WEB_PID > "$PID_DIR/web-test.pid"

log_info "Web Test server started (PID: $WEB_PID)"

# Wait briefly for web server to start
sleep 2
if curl -s "http://localhost:$WEB_PORT/health" > /dev/null 2>&1; then
    log_success "Web Test server is ready on port $WEB_PORT"
else
    log_warning "Web Test server may still be starting..."
fi

################################################################################
# Summary
################################################################################
echo ""
echo "========================================================================"
log_success "All services started!"
echo "========================================================================"
echo ""
echo "Services:"
echo "  vLLM:        http://localhost:$VLLM_PORT"
echo "  Fish Speech: http://localhost:$FISH_PORT"
echo "  Web Test:    http://localhost:$WEB_PORT"
echo ""
echo "Logs:"
echo "  vLLM:        $VLLM_INSTALL_DIR/helpers/vllm-server.log"
echo "  Fish Speech: $LOG_DIR/fish-speech.log"
echo "  Web Test:    $LOG_DIR/web-test.log"
echo ""
echo "To stop all services: ./scripts/stop-all.sh"
echo ""
