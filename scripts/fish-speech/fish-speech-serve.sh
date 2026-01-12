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

# Check for model
MODEL_DIR="$FISH_SPEECH_DIR/checkpoints/openaudio-s1-mini"
if [ ! -d "$MODEL_DIR" ]; then
    log_error "Model not found at $MODEL_DIR"
    log_info "Run install_fish_speech.sh to download the model"
    exit 1
fi

# Activate virtual environment
log_info "Activating Fish Speech environment..."
source "$VENV_DIR/bin/activate"

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
    --llama-checkpoint-path "checkpoints/openaudio-s1-mini" \
    --decoder-checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth" \
    --decoder-config-name "modded_dac_vq" \
    --compile
