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

# Check for HuggingFace CLI
if ! command -v huggingface-cli &> /dev/null; then
    log_error "huggingface-cli not found"
    log_info "Install with: uv pip install huggingface-hub"
    exit 1
fi

# Check if logged in to HuggingFace
if ! huggingface-cli whoami &> /dev/null; then
    log_warn "Not logged in to HuggingFace"
    log_info "Some models may require authentication"
    log_info "Run: huggingface-cli login"
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

    huggingface-cli download \
        "$MODEL_NAME" \
        --local-dir "$LOCAL_DIR" \
        --local-dir-use-symlinks False \
        --resume-download

    log_info "$MODEL_NAME downloaded successfully"
}

# Download LLM (Llama 3.1 70B AWQ)
log_info "=== LLM Models ==="

echo ""
echo "Choose LLM model to download:"
echo "  1) Llama 3.1 70B AWQ (40GB) - Recommended for DGX Spark"
echo "  2) Llama 3.1 30B AWQ (16GB) - Smaller, faster"
echo "  3) Llama 3.1 8B AWQ (5GB) - Minimal, for testing"
echo "  4) Skip LLM download"
echo ""
read -p "Enter choice (1-4): " llm_choice

case $llm_choice in
    1)
        download_model \
            "TheBloke/Llama-3.1-70B-Instruct-AWQ" \
            "llama-3.1-70b-awq" \
            "40"
        ;;
    2)
        download_model \
            "TheBloke/Llama-3.1-30B-Instruct-AWQ" \
            "llama-3.1-30b-awq" \
            "16"
        ;;
    3)
        download_model \
            "TheBloke/Llama-3.1-8B-Instruct-AWQ" \
            "llama-3.1-8b-awq" \
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

# TTS models (download on first use)
echo ""
log_info "=== TTS Models ==="
log_info "Coqui TTS models will download automatically on first use"
log_info "Default model: tts_models/en/ljspeech/tacotron2-DDC (~500MB)"
echo ""
read -p "Pre-download TTS models now? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "Pre-downloading TTS models..."

    # This will be downloaded by TTS library on first use
    # We can trigger it with a simple Python script
    python3 << 'EOF'
import os
os.environ['COQUI_TOS_AGREED'] = '1'

try:
    from TTS.api import TTS
    print("Downloading default TTS model...")
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
    print("TTS model downloaded successfully")
except Exception as e:
    print(f"Error downloading TTS: {e}")
    print("TTS will be downloaded on first run")
EOF
fi

# STT models (download on first use)
echo ""
log_info "=== STT Models ==="
log_info "Whisper models will download automatically on first use"
log_info "Default model: large-v3 (~3GB)"
echo ""
read -p "Pre-download Whisper model now? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "Pre-downloading Whisper model..."

    python3 << 'EOF'
try:
    from faster_whisper import WhisperModel
    print("Downloading Whisper large-v3...")
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")
    print("Whisper model downloaded successfully")
except Exception as e:
    print(f"Error downloading Whisper: {e}")
    print("Whisper will be downloaded on first run")
EOF
fi

# Sentence transformers (for memory/RAG)
echo ""
log_info "=== Embedding Models ==="
log_info "Downloading sentence transformer for memory system..."

python3 << 'EOF'
try:
    from sentence_transformers import SentenceTransformer
    print("Downloading all-MiniLM-L6-v2 embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Embedding model downloaded successfully")
except Exception as e:
    print(f"Error downloading embeddings: {e}")
    print("Embeddings will be downloaded on first run")
EOF

# Summary
echo ""
echo "================================================"
log_info "Model download complete! 🎉"
echo "================================================"
echo ""
log_info "Downloaded models are stored in ./models/"
echo ""
log_info "Update your .env file to use downloaded models:"
echo ""
echo "For 70B model:"
echo "  LLM__MODEL_NAME=./models/llama-3.1-70b-awq"
echo ""
echo "For 30B model:"
echo "  LLM__MODEL_NAME=./models/llama-3.1-30b-awq"
echo ""
echo "For 8B model:"
echo "  LLM__MODEL_NAME=./models/llama-3.1-8b-awq"
echo ""

log_warn "Remember to verify you have enough GPU memory:"
echo "  - 70B model: requires ~40GB GPU memory"
echo "  - 30B model: requires ~16GB GPU memory"
echo "  - 8B model: requires ~5GB GPU memory"
echo ""

exit 0
