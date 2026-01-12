#!/bin/bash
#
# Model download script for Spark VTuber
# Downloads required models for LLM, TTS, and STT
#

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "================================================"
echo "  Spark VTuber Model Download Script"
echo "================================================"
echo ""

# Create models directory
mkdir -p models
cd models

# Check for HuggingFace CLI (hf command from huggingface_hub)
if ! command -v hf &> /dev/null; then
    log_error "hf command not found"
    log_info "Install with: uv pip install huggingface-hub"
    exit 1
fi

# Check if logged in to HuggingFace
if ! hf auth whoami &> /dev/null; then
    log_warn "Not logged in to HuggingFace"
    log_info "Some models may require authentication"
    log_info "Run: hf auth login"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Function to download model with progress
download_model() {
    local MODEL_NAME=$1
    local LOCAL_DIR=$2
    local SIZE_GB=$3

    echo ""
    log_info "Downloading $MODEL_NAME (~${SIZE_GB}GB)"
    log_warn "This may take a while depending on your connection..."

    if [ -d "$LOCAL_DIR" ] && [ "$(ls -A $LOCAL_DIR)" ]; then
        log_warn "Model already exists in $LOCAL_DIR"
        read -p "Re-download? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Skipping $MODEL_NAME"
            return 0
        fi
        rm -rf "$LOCAL_DIR"
    fi

    hf download \
        "$MODEL_NAME" \
        --local-dir "$LOCAL_DIR"

    log_info "$MODEL_NAME downloaded successfully"
}

# Download LLM (Qwen3 MoE models)
log_info "=== LLM Models ==="

echo ""
echo "Choose LLM model to download:"
echo "  1) Qwen3-30B-A3B AWQ (15-20GB) - Recommended for DGX Spark (MoE: 30B total, 3B active)"
echo "  2) Qwen3-14B AWQ (8GB) - Smaller, faster"
echo "  3) Qwen3-8B AWQ (5GB) - Minimal, for testing"
echo "  4) Skip LLM download"
echo ""
read -p "Enter choice (1-4): " llm_choice

case $llm_choice in
    1)
        download_model \
            "QuixiAI/Qwen3-30B-A3B-AWQ" \
            "qwen3-30b-a3b-awq" \
            "20"
        ;;
    2)
        download_model \
            "Qwen/Qwen3-14B-AWQ" \
            "qwen3-14b-awq" \
            "8"
        ;;
    3)
        download_model \
            "Qwen/Qwen3-8B-AWQ" \
            "qwen3-8b-awq" \
            "5"
        ;;
    4)
        log_info "Skipping LLM download"
        ;;
    *)
        log_error "Invalid choice"
        exit 1
        ;;
esac

# TTS models (CosyVoice 3)
echo ""
log_info "=== TTS Models (CosyVoice 3) ==="
log_info "CosyVoice model will download automatically on first use"
log_info "Default model: FunAudioLLM/Fun-CosyVoice3-0.5B-2512 (~8GB)"
log_info "Features: TRUE streaming (150ms latency), 100+ emotions, zero-shot voice cloning"
echo ""
read -p "Pre-download CosyVoice model now? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "Pre-downloading CosyVoice model..."

    python3 << 'EOF'
try:
    from huggingface_hub import snapshot_download
    print("Downloading CosyVoice model (FunAudioLLM/Fun-CosyVoice3-0.5B-2512)...")
    snapshot_download(
        repo_id="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        local_dir="models/cosyvoice3-0.5b",
        allow_patterns=["*.pt", "*.pth", "*.json", "*.yaml", "config.*", "*.safetensors"]
    )
    print("CosyVoice model downloaded successfully")
except Exception as e:
    print(f"Error downloading CosyVoice: {e}")
    print("CosyVoice will be downloaded on first run")
EOF
fi

# STT models (Parakeet TDT 0.6B V2)
echo ""
log_info "=== STT Models (Parakeet TDT 0.6B V2) ==="
log_info "Model: nvidia/parakeet-tdt-0.6b-v2 (~4GB)"
log_info "16x faster than Whisper Turbo, 6.05% WER"
echo ""
read -p "Pre-download Parakeet TDT model now? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "Pre-downloading Parakeet TDT model..."
    download_model \
        "nvidia/parakeet-tdt-0.6b-v2" \
        "parakeet-tdt-0.6b-v2" \
        "4"
fi

# Sentence transformers (for memory/RAG)
echo ""
log_info "=== Embedding Models ==="
log_info "Model: sentence-transformers/all-MiniLM-L6-v2 (~90MB)"
echo ""
read -p "Pre-download embedding model now? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "Pre-downloading embedding model..."
    download_model \
        "sentence-transformers/all-MiniLM-L6-v2" \
        "all-MiniLM-L6-v2" \
        "0.1"
fi

# Summary
echo ""
echo "================================================"
log_info "Model download complete!"
echo "================================================"
echo ""
log_info "Downloaded models are stored in ./models/"
echo ""
log_info "Update your .env file to use downloaded models:"
echo ""
echo "For Qwen3-30B-A3B model (recommended):"
echo "  LLM__MODEL_NAME=./models/qwen3-30b-a3b-awq"
echo ""
echo "For Qwen3-14B model:"
echo "  LLM__MODEL_NAME=./models/qwen3-14b-awq"
echo ""
echo "For Qwen3-8B model:"
echo "  LLM__MODEL_NAME=./models/qwen3-8b-awq"
echo ""

log_warn "Remember to verify you have enough GPU memory:"
echo "  - Qwen3-30B-A3B model: requires ~15-20GB GPU memory (MoE architecture)"
echo "  - Qwen3-14B model: requires ~8GB GPU memory"
echo "  - Qwen3-8B model: requires ~5GB GPU memory"
echo ""

exit 0
