#!/bin/bash
################################################################################
# Fish Speech API Server Startup Script
#
# Usage: ./fish-speech-serve.sh [PORT]
# Default port: 8843
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
PORT="${1:-8843}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Find Fish Speech installation
# Check common locations
if [ -d "$REPO_ROOT/fish-speech-install/fish-speech" ]; then
    FISH_SPEECH_DIR="$REPO_ROOT/fish-speech-install/fish-speech"
    VENV_DIR="$REPO_ROOT/fish-speech-install/.fish-speech"
elif [ -d "$REPO_ROOT/fish-speech" ]; then
    FISH_SPEECH_DIR="$REPO_ROOT/fish-speech"
    VENV_DIR="$REPO_ROOT/.fish-speech"
elif [ -d "$HOME/fish-speech-install/fish-speech" ]; then
    FISH_SPEECH_DIR="$HOME/fish-speech-install/fish-speech"
    VENV_DIR="$HOME/fish-speech-install/.fish-speech"
else
    log_error "Fish Speech installation not found!"
    log_info "Expected locations:"
    log_info "  - $REPO_ROOT/fish-speech-install/fish-speech"
    log_info "  - $REPO_ROOT/fish-speech"
    log_info "  - $HOME/fish-speech-install/fish-speech"
    log_info ""
    log_info "Run install_fish_speech.sh first to install Fish Speech"
    exit 1
fi

# Check for virtual environment
if [ ! -d "$VENV_DIR" ]; then
    log_error "Virtual environment not found at $VENV_DIR"
    log_info "Run install_fish_speech.sh first to set up the environment"
    exit 1
fi

# Check for model in multiple locations
# Priority: 1) models/openaudio-s1-mini (new name from download_models.sh)
#           2) models/fish-speech-model (old name from download_models.sh)
#           3) fish-speech/checkpoints/openaudio-s1-mini (from install_fish_speech.sh)
if [ -d "$REPO_ROOT/models/openaudio-s1-mini" ]; then
    MODEL_DIR="$REPO_ROOT/models/openaudio-s1-mini"
    MODEL_PATH="$MODEL_DIR"
elif [ -d "$REPO_ROOT/models/fish-speech-model" ]; then
    MODEL_DIR="$REPO_ROOT/models/fish-speech-model"
    MODEL_PATH="$MODEL_DIR"
elif [ -d "$FISH_SPEECH_DIR/checkpoints/openaudio-s1-mini" ]; then
    MODEL_DIR="$FISH_SPEECH_DIR/checkpoints/openaudio-s1-mini"
    MODEL_PATH="checkpoints/openaudio-s1-mini"
else
    log_error "Model not found!"
    log_info "Expected locations:"
    log_info "  - $REPO_ROOT/models/openaudio-s1-mini (from download_models.sh)"
    log_info "  - $REPO_ROOT/models/fish-speech-model (legacy location)"
    log_info "  - $FISH_SPEECH_DIR/checkpoints/openaudio-s1-mini (from install_fish_speech.sh)"
    log_info ""
    log_info "Run download_models.sh or install_fish_speech.sh to download the model"
    exit 1
fi

# Check GPU memory (Fish Speech needs ~12GB)
if command -v nvidia-smi &> /dev/null; then
    GPU_MEM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    GPU_MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    # Handle [N/A] or non-numeric values (e.g., on unified memory systems like DGX Spark)
    if [[ "$GPU_MEM_FREE" =~ ^[0-9]+$ ]] && [[ "$GPU_MEM_TOTAL" =~ ^[0-9]+$ ]]; then
        log_info "GPU Memory: ${GPU_MEM_FREE}MB free / ${GPU_MEM_TOTAL}MB total"
        if [[ "$GPU_MEM_FREE" -lt 12000 ]]; then
            log_warning "Fish Speech needs ~12GB VRAM. Only ${GPU_MEM_FREE}MB free."
            log_warning "If vLLM is running, you may need to stop it first or ensure enough memory."
        fi
    else
        log_info "GPU Memory: Unable to query (unified memory system)"
    fi
fi

# Activate virtual environment
log_info "Activating Fish Speech environment..."
source "$VENV_DIR/bin/activate"

# Set TRITON_PTXAS_PATH to use system ptxas for DGX Spark (GB10/SM 12.1a) compatibility
# The bundled ptxas in Triton doesn't support sm_121a, but the system ptxas from CUDA 13.0 does
# See: https://github.com/pytorch/pytorch/issues/163801
if [ -f "/usr/local/cuda/bin/ptxas" ]; then
    export TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"
    log_info "Using system ptxas for Triton: $TRITON_PTXAS_PATH"
fi

# Change to Fish Speech directory
cd "$FISH_SPEECH_DIR"

# Start the server
log_info "Starting Fish Speech API server on port $PORT..."
log_info "Model: $MODEL_DIR"
log_info ""
log_info "API Endpoints:"
log_info "  POST http://0.0.0.0:$PORT/v1/tts - Generate speech"
log_info "  GET  http://0.0.0.0:$PORT/v1/health - Health check"
log_info "  GET  http://0.0.0.0:$PORT/v1/references/list - List voices"
log_info ""
log_success "Server starting... (Ctrl+C to stop)"
log_info ""

# Run the API server
exec python -m tools.api_server \
    --listen "0.0.0.0:$PORT" \
    --llama-checkpoint-path "$MODEL_PATH" \
    --decoder-checkpoint-path "$MODEL_PATH/codec.pth" \
    --decoder-config-name "modded_dac_vq" \
    --compile
