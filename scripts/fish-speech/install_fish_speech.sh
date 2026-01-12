#!/bin/bash
################################################################################
# Fish Speech Installation Script for NVIDIA DGX Spark (Blackwell GB10)
# Version: 1.0.0
#
# This script automates the installation of Fish Speech TTS on DGX Spark systems
# with Blackwell GB10 GPUs.
#
# Usage: ./install_fish_speech.sh [OPTIONS]
# Options:
#   --install-dir DIR    Installation directory (default: $PWD/fish-speech-install)
#   --python-version VER Python version (default: 3.12)
#   --skip-model         Skip model download
#   --help               Show this help message
################################################################################

set -e
set -o pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default configuration
INSTALL_DIR="$PWD/fish-speech-install"
PYTHON_VERSION="3.12"
SKIP_MODEL=false
FISH_SPEECH_REPO="https://github.com/fishaudio/fish-speech.git"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

################################################################################
# Helper Functions
################################################################################

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

check_command() {
    if command -v "$1" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

show_help() {
    echo "Fish Speech Installation Script for DGX Spark"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --install-dir DIR    Installation directory (default: \$PWD/fish-speech-install)"
    echo "  --python-version VER Python version (default: 3.12)"
    echo "  --skip-model         Skip model download"
    echo "  --help               Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --install-dir ./fish-speech-install"
    exit 0
}

################################################################################
# Parse Arguments
################################################################################

while [[ $# -gt 0 ]]; do
    case $1 in
        --install-dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        --python-version)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --skip-model)
            SKIP_MODEL=true
            shift
            ;;
        --help)
            show_help
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            ;;
    esac
done

# Convert to absolute path
INSTALL_DIR="$(cd "$(dirname "$INSTALL_DIR")" 2>/dev/null && pwd)/$(basename "$INSTALL_DIR")"

################################################################################
# Pre-flight Checks
################################################################################

preflight_checks() {
    print_header "Pre-flight System Checks"

    log_info "Checking system requirements..."

    # Check if running on ARM64
    ARCH=$(uname -m)
    if [[ "$ARCH" != "aarch64" ]] && [[ "$ARCH" != "arm64" ]]; then
        log_warning "This script is designed for ARM64 architecture (DGX Spark)"
        log_warning "Detected architecture: $ARCH"
    fi

    # Check for NVIDIA GPU
    if ! check_command nvidia-smi; then
        log_error "nvidia-smi not found. NVIDIA drivers required."
        exit 1
    fi

    # Check GPU type
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    log_info "Detected GPU: $GPU_NAME"

    # Check CUDA
    if ! check_command nvcc; then
        if [ -x "/usr/local/cuda/bin/nvcc" ]; then
            export PATH="/usr/local/cuda/bin:$PATH"
            log_info "Found CUDA at /usr/local/cuda, added to PATH"
        else
            log_error "CUDA toolkit not found. Please install CUDA 12.8+"
            exit 1
        fi
    fi

    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)
    log_info "CUDA version: $CUDA_VERSION"

    # Check disk space (need ~20GB for Fish Speech + model)
    AVAILABLE_SPACE=$(df -BG "$HOME" | tail -1 | awk '{print $4}' | sed 's/G//')
    if [[ "$AVAILABLE_SPACE" -lt 20 ]]; then
        log_error "Insufficient disk space. Need at least 20GB, have ${AVAILABLE_SPACE}GB"
        exit 1
    fi

    log_success "Pre-flight checks passed!"
}

################################################################################
# Install System Dependencies
################################################################################

install_system_deps() {
    print_header "Step 1/5: Installing System Dependencies"

    log_info "Installing audio processing libraries..."
    
    if ! sudo -n true 2>/dev/null; then
        log_warning "sudo access required to install system dependencies"
        log_info "Please enter your password when prompted"
    fi

    sudo apt-get update
    sudo apt-get install -y \
        portaudio19-dev \
        libsox-dev \
        ffmpeg \
        build-essential \
        git

    log_success "System dependencies installed!"
}

################################################################################
# Install uv Package Manager
################################################################################

install_uv() {
    print_header "Step 2/5: Installing uv Package Manager"

    if check_command uv; then
        UV_VERSION=$(uv --version | awk '{print $2}')
        log_info "uv already installed: v$UV_VERSION"
    else
        log_info "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
        log_success "uv installed successfully"
    fi

    if ! check_command uv; then
        log_error "uv installation failed"
        exit 1
    fi
}

################################################################################
# Clone Fish Speech and Create Virtual Environment
################################################################################

setup_fish_speech() {
    print_header "Step 3/5: Setting Up Fish Speech"

    mkdir -p "$INSTALL_DIR"
    cd "$INSTALL_DIR"

    FISH_SPEECH_DIR="$INSTALL_DIR/fish-speech"
    VENV_DIR="$INSTALL_DIR/.fish-speech"

    # Clone Fish Speech
    if [ -d "$FISH_SPEECH_DIR" ]; then
        log_warning "Fish Speech directory already exists"
        read -p "Remove and re-clone? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$FISH_SPEECH_DIR"
            git clone --depth 1 "$FISH_SPEECH_REPO" "$FISH_SPEECH_DIR"
        else
            log_info "Using existing Fish Speech directory"
        fi
    else
        log_info "Cloning Fish Speech repository..."
        git clone --depth 1 "$FISH_SPEECH_REPO" "$FISH_SPEECH_DIR"
    fi

    # Create virtual environment
    if [ -d "$VENV_DIR" ]; then
        log_warning "Virtual environment already exists at $VENV_DIR"
        read -p "Remove and recreate? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_DIR"
        else
            log_info "Using existing virtual environment"
            return
        fi
    fi

    log_info "Creating Python $PYTHON_VERSION virtual environment..."
    uv venv "$VENV_DIR" --python "$PYTHON_VERSION"

    log_success "Fish Speech cloned and virtual environment created"
}

################################################################################
# Install Fish Speech Dependencies
################################################################################

install_fish_speech() {
    print_header "Step 4/5: Installing Fish Speech"

    cd "$INSTALL_DIR/fish-speech"
    source "$INSTALL_DIR/.fish-speech/bin/activate"

    # Determine CUDA version for PyTorch
    # DGX Spark has CUDA 13.0, but PyTorch wheels are available for cu129
    log_info "Installing PyTorch with CUDA support..."
    
    # Check if CUDA 13.0 (use cu129 wheels which are compatible)
    CUDA_MAJOR=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d'.' -f1 | tr -d 'V')
    
    if [[ "$CUDA_MAJOR" -ge 13 ]]; then
        log_info "CUDA 13.0+ detected, using cu129 PyTorch wheels..."
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
    else
        log_info "Using cu126 PyTorch wheels..."
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    fi

    # Verify PyTorch
    log_info "Verifying PyTorch installation..."
    python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

    # Install Fish Speech
    log_info "Installing Fish Speech package..."
    uv pip install -e .

    # Verify installation
    log_info "Verifying Fish Speech installation..."
    python -c "import fish_speech; print('Fish Speech installed successfully')"

    log_success "Fish Speech installed successfully"
}

################################################################################
# Download Model
################################################################################

download_model() {
    print_header "Step 5/5: Downloading Fish Speech Model"

    if [ "$SKIP_MODEL" = true ]; then
        log_info "Skipping model download (--skip-model flag set)"
        return
    fi

    cd "$INSTALL_DIR/fish-speech"
    source "$INSTALL_DIR/.fish-speech/bin/activate"

    MODEL_DIR="$INSTALL_DIR/fish-speech/checkpoints/openaudio-s1-mini"

    if [ -d "$MODEL_DIR" ] && [ "$(ls -A $MODEL_DIR 2>/dev/null)" ]; then
        log_warning "Model already exists at $MODEL_DIR"
        read -p "Re-download? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Skipping model download"
            return
        fi
        rm -rf "$MODEL_DIR"
    fi

    log_info "Downloading Fish Speech model (fishaudio/openaudio-s1-mini)..."
    log_warning "This may take a while (~12GB download)..."

    # Install huggingface_hub if not present
    uv pip install huggingface_hub

    # Download model
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='fishaudio/openaudio-s1-mini',
    local_dir='checkpoints/openaudio-s1-mini'
)
print('Model downloaded successfully')
"

    log_success "Model downloaded to $MODEL_DIR"
}

################################################################################
# Print Summary
################################################################################

print_summary() {
    print_header "Installation Complete!"

    echo "Fish Speech has been installed to: $INSTALL_DIR"
    echo ""
    echo "To activate the environment:"
    echo "  source $INSTALL_DIR/.fish-speech/bin/activate"
    echo ""
    echo "To start the API server:"
    echo "  cd $INSTALL_DIR/fish-speech"
    echo "  python -m tools.api_server \\"
    echo "    --listen 0.0.0.0:8843 \\"
    echo "    --llama-checkpoint-path checkpoints/openaudio-s1-mini \\"
    echo "    --decoder-checkpoint-path checkpoints/openaudio-s1-mini/codec.pth \\"
    echo "    --decoder-config-name modded_dac_vq \\"
    echo "    --compile"
    echo ""
    echo "Or use the helper script:"
    echo "  $SCRIPT_DIR/fish-speech-serve.sh 8843"
    echo ""
    echo "API endpoints:"
    echo "  POST http://localhost:8843/v1/tts - Generate speech from text"
    echo "  GET  http://localhost:8843/v1/health - Health check"
    echo "  GET  http://localhost:8843/v1/references/list - List voice references"
    echo ""
    log_success "Fish Speech is ready to use!"
}

################################################################################
# Main
################################################################################

main() {
    print_header "Fish Speech Installation for DGX Spark"
    
    echo "Installation directory: $INSTALL_DIR"
    echo "Python version: $PYTHON_VERSION"
    echo ""

    preflight_checks
    install_system_deps
    install_uv
    setup_fish_speech
    install_fish_speech
    download_model
    print_summary
}

main "$@"
