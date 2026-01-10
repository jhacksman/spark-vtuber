# Spark VTuber Setup Guide

Complete setup instructions for running Spark VTuber on NVIDIA DGX Spark using UV.

---

## Prerequisites

### Hardware Requirements
- **NVIDIA DGX Spark** with GB10 Grace Blackwell superchip
- 128GB unified LPDDR5x memory
- 2TB+ NVMe storage for models
- Network connection for model downloads

### Software Requirements
- **Ubuntu 22.04 LTS** or later
- **CUDA 12.3+** with NVIDIA drivers 545+
- **Python 3.10+**
- **UV** package manager (we'll install this)
- **VTube Studio** (if using avatar features)

---

## Quick Start (5 Minutes)

```bash
# Clone the repository
git clone https://github.com/jhacksman/spark-vtuber.git
cd spark-vtuber

# Run the automated setup script
bash scripts/setup.sh

# Activate the environment
source .venv/bin/activate

# Start the system (with test mode - no external dependencies)
uv run spark-vtuber run --no-chat --no-avatar --no-game
```

---

## Detailed Setup Instructions

### Step 1: Install System Dependencies

```bash
# Update package lists
sudo apt update && sudo apt upgrade -y

# Install required system packages
sudo apt install -y \
    build-essential \
    git \
    curl \
    wget \
    ffmpeg \
    portaudio19-dev \
    libsndfile1 \
    libgomp1 \
    libopenblas-dev
```

### Step 2: Verify CUDA Installation

```bash
# Check CUDA version (should be 12.3+)
nvcc --version

# Check GPU is accessible
nvidia-smi

# If nvidia-smi fails, install NVIDIA drivers:
# sudo apt install nvidia-driver-545 nvidia-utils-545
# sudo reboot
```

### Step 3: Install UV Package Manager

```bash
# Install UV (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH if not already added
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Verify installation
uv --version  # Should show uv 0.5.0 or later
```

### Step 4: Clone Repository and Setup Project

```bash
# Clone the repository
git clone https://github.com/jhacksman/spark-vtuber.git
cd spark-vtuber

# Create virtual environment with UV
uv venv --python 3.10

# Activate environment
source .venv/bin/activate

# Install dependencies (this will take 10-20 minutes)
uv pip install -e ".[dev]"

# Note: UV automatically uses pyproject.toml
```

### Step 5: Download Models

#### Option A: Automated Download (Recommended)

```bash
# Run the model download script
bash scripts/download_models.sh

# This downloads:
# - Qwen3-30B-A3B AWQ (quantized MoE model)
# - Fish Speech 1.5 model (openaudio-s1-mini)
# - Whisper Large-v3
# - Sentence transformer for embeddings
```

#### Option B: Manual Download

```bash
# Create models directory
mkdir -p models

# Download Qwen3-30B-A3B AWQ
# Using HuggingFace CLI (requires login)
huggingface-cli login  # Enter your HF token

huggingface-cli download \
    Qwen/Qwen3-30B-A3B-Instruct-AWQ \
    --local-dir models/qwen3-30b-a3b-awq \
    --local-dir-use-symlinks False

# Fish Speech 1.5 setup (local inference)
git clone https://github.com/fishaudio/fish-speech
cd fish-speech && pip install -e ".[cu129]"
cd ..

# Fish Speech model downloads automatically from HuggingFace on first use
# Or manually download:
# huggingface-cli download fishaudio/openaudio-s1-mini --local-dir models/fish-speech

# Whisper models download automatically on first use
```

### Step 6: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit configuration
nano .env  # or your preferred editor
```

**Minimal `.env` configuration:**

```bash
# LLM Configuration
LLM__MODEL_NAME=./models/qwen3-30b-a3b-awq
LLM__QUANTIZATION=awq
LLM__GPU_MEMORY_UTILIZATION=0.70
LLM__CONTEXT_LENGTH=8192

# TTS Configuration (Fish Speech 1.5 - local inference)
TTS__ENGINE=fish_speech
TTS__USE_API=false
TTS__DEVICE=cuda
TTS__HALF_PRECISION=true

# STT Configuration (Parakeet TDT - ultra-fast, low-latency)
STT__ENGINE=parakeet
STT__MODEL_NAME=nvidia/parakeet-tdt-0.6b-v2
STT__DEVICE=cuda

# Memory Configuration
MEMORY__CHROMA_PERSIST_DIR=./data/chroma

# Avatar Configuration (if using VTube Studio)
AVATAR__VTUBE_STUDIO_HOST=localhost
AVATAR__VTUBE_STUDIO_PORT=8001

# Chat Configuration (if using Twitch)
CHAT__TWITCH_ENABLED=false
CHAT__TWITCH_CHANNEL=your_channel
CHAT__TWITCH_OAUTH_TOKEN=  # Get from https://twitchapps.com/tmi/

# Personality Configuration
PERSONALITY__PRIMARY_NAME=Spark
PERSONALITY__SECONDARY_NAME=Shadow
```

### Step 7: Verify Installation

```bash
# Check version
uv run spark-vtuber version

# Show configuration
uv run spark-vtuber status

# Test LLM (this will load the model - takes 30-60s)
uv run spark-vtuber test-llm "Hello, who are you?" --max-tokens 100

# Test TTS
uv run spark-vtuber test-tts "Hello world" --output test.wav
```

---

## Component-Specific Setup

### VTube Studio (Avatar)

1. **Install VTube Studio**
   ```bash
   # Download from https://denchisoft.com/
   # Or use Steam version
   ```

2. **Enable Plugin API**
   - Open VTube Studio
   - Settings → Plugins
   - Enable "Allow Plugins"
   - Note the port (default: 8001)

3. **Load Live2D Model**
   - Import your Live2D model
   - Configure expressions/hotkeys:
     - `neutral`, `happy`, `sad`, `angry`, `surprised`, `thinking`, `excited`, `confused`

4. **Test Connection**
   ```bash
   # This will request authentication
   uv run python -c "
   import asyncio
   from spark_vtuber.avatar.vtube_studio import VTubeStudioAvatar

   async def test():
       avatar = VTubeStudioAvatar()
       await avatar.connect()
       print('Connected to VTube Studio!')
       await avatar.disconnect()

   asyncio.run(test())
   "
   ```

### Twitch Chat Integration

1. **Get OAuth Token**
   ```bash
   # Visit https://twitchapps.com/tmi/
   # Click "Connect" and authorize
   # Copy the oauth:xxxxx token
   ```

2. **Update `.env`**
   ```bash
   CHAT__TWITCH_ENABLED=true
   CHAT__TWITCH_CHANNEL=your_channel_name
   CHAT__TWITCH_OAUTH_TOKEN=oauth:your_token_here
   ```

3. **Test Connection**
   ```bash
   uv run python -c "
   import asyncio
   from spark_vtuber.chat.twitch import TwitchChat

   async def test():
       chat = TwitchChat(
           channel='your_channel',
           oauth_token='oauth:your_token'
       )
       await chat.connect()
       print('Connected to Twitch!')

       # Read 5 messages
       count = 0
       async for msg in chat.get_messages():
           print(f'{msg.username}: {msg.content}')
           count += 1
           if count >= 5:
               break

       await chat.disconnect()

   asyncio.run(test())
   "
   ```

### VNet Multiplayer Collab Setup (Dual Avatar Mode)

Enable both AI personalities (Spark and Shadow) to be visible on screen simultaneously using VTube Studio's VNet Multiplayer Collab plugin.

#### Requirements

- **VTube Studio Pro** ($14.99)
- **VNet Multiplayer Collab** ($20.00) - Available in VTube Studio
- **Two Live2D models** (one for each personality)

#### Configuration Steps

1. **Install VNet in VTube Studio**
   - Open VTube Studio
   - Settings → Plugins → Install VNet Multiplayer Collab
   - Restart VTube Studio

2. **Launch Two VTube Studio Instances**

   **Instance 1 (Primary - Spark):**
   - Open VTube Studio normally
   - Settings → Plugins → Set port to 8001
   - Load your primary Live2D model (Spark)

   **Instance 2 (Secondary - Shadow):**
   - Launch a second VTube Studio instance
   - Settings → Plugins → Set port to 8002
   - Load your secondary Live2D model (Shadow)

   > **Tip:** Use the portable version or separate installations to run multiple instances.

3. **Enable VNet in Both Instances**
   - In both VTube Studio windows:
     - Settings → VNet → Enable VNet
     - Configure positions (left for primary, right for secondary)
     - Connect both to the same VNet session

4. **Update `.env` Configuration**
   ```bash
   AVATAR__DUAL_AVATAR_ENABLED=true
   AVATAR__PRIMARY_AVATAR_PORT=8001
   AVATAR__SECONDARY_AVATAR_PORT=8002
   AVATAR__PRIMARY_AVATAR_POSITION=left
   AVATAR__SECONDARY_AVATAR_POSITION=right
   ```

5. **Run with Dual Avatar Mode**
   ```bash
   uv run spark-vtuber run --dual-avatar
   ```

#### Troubleshooting VNet

**Issue: Second VTube Studio won't launch**
- Solution: Use portable version or separate installations

**Issue: Ports already in use**
- Solution: Change ports in VTube Studio settings and update `.env`

**Issue: Avatars not synced in VNet**
- Solution: Ensure both instances are connected to the same VNet session

**Issue: Only one avatar receives lip sync**
- Solution: This is expected behavior - only the active speaker receives lip sync. The system automatically switches based on which personality is responding.

---

### Minecraft Integration (Future)

> ⚠️ **Note:** Minecraft integration is currently stubbed. Full implementation coming soon.

For when it's available:

1. **Install Mineflayer**
   ```bash
   npm install -g mineflayer
   ```

2. **Configure Server**
   ```bash
   GAME__MINECRAFT_ENABLED=true
   GAME__MINECRAFT_HOST=localhost
   GAME__MINECRAFT_PORT=25565
   GAME__MINECRAFT_USERNAME=SparkVTuber
   ```

---

## Running the System

### Test Mode (No External Dependencies)

```bash
# Run with just LLM and TTS (good for testing)
uv run spark-vtuber run --no-chat --no-avatar --no-game
```

### With Avatar Only

```bash
# Make sure VTube Studio is running first!
uv run spark-vtuber run --no-chat --no-game
```

### With Twitch Chat

```bash
# Requires Twitch OAuth setup
uv run spark-vtuber run --no-avatar --no-game
```

### Full System

```bash
# All components enabled
# Requires: VTube Studio running, Twitch configured
uv run spark-vtuber run
```

### Debug Mode

```bash
# Enable verbose logging
uv run spark-vtuber run --debug
```

---

## Performance Tuning

### Memory Optimization

If you encounter OOM errors:

```bash
# Reduce GPU memory utilization
LLM__GPU_MEMORY_UTILIZATION=0.60  # from 0.70

# Use smaller context window
LLM__CONTEXT_LENGTH=4096  # from 8192

# Use smaller model variant
LLM__MODEL_NAME=./models/qwen3-14b-awq  # Instead of 30B-A3B
```

### Latency Optimization

For better response times:

```bash
# Reduce temperature for faster sampling
LLM__TEMPERATURE=0.5  # from 0.7

# Use smaller TTS model
TTS__MODEL_NAME=tts_models/en/ljspeech/vits

# Disable STT VAD if not needed
STT__VAD_ENABLED=false
```

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Symptoms:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions:**
1. Check memory usage: `nvidia-smi`
2. Reduce `LLM__GPU_MEMORY_UTILIZATION` to 0.60
3. Use smaller model (30B instead of 70B)
4. Close other GPU applications

### Issue: vLLM Import Error

**Symptoms:**
```
ImportError: cannot import name 'AsyncLLMEngine' from 'vllm'
```

**Solutions:**
1. Verify CUDA installation: `nvcc --version`
2. Reinstall vLLM: `uv pip install --force-reinstall vllm`
3. Check Python version: `python --version` (must be 3.10+)

### Issue: VTube Studio Won't Connect

**Symptoms:**
```
Failed to connect to VTube Studio: Connection refused
```

**Solutions:**
1. Verify VTube Studio is running
2. Check Plugins are enabled in VTube Studio settings
3. Verify port in `.env` matches VTube Studio (default 8001)
4. Allow firewall: `sudo ufw allow 8001/tcp`

### Issue: Twitch Authentication Failed

**Symptoms:**
```
Authentication pending - please accept in VTube Studio
```

**Solutions:**
1. Verify OAuth token format: `oauth:xxxxx`
2. Generate new token at https://twitchapps.com/tmi/
3. Check channel name is lowercase and without `#`
4. Verify internet connection

### Issue: Models Downloading Slowly

**Solutions:**
1. Use mirrors:
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   huggingface-cli download ...
   ```

2. Or download via torrent:
   ```bash
   # Check model card on HuggingFace for torrent links
   ```

3. Resume interrupted downloads:
   ```bash
   # HuggingFace CLI automatically resumes
   huggingface-cli download Qwen/Qwen3-30B-A3B-Instruct-AWQ \
       --local-dir models/qwen3-30b-a3b-awq \
       --resume-download
   ```

---

## Development Setup

### Install Development Tools

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run type checking
mypy src

# Run linting
ruff check src

# Format code
black src
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=spark_vtuber --cov-report=html

# Run specific test file
pytest tests/unit/test_pipeline.py

# Run integration tests (require GPU)
pytest tests/integration/ -m gpu
```

---

## Monitoring and Maintenance

### Check System Health

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor memory usage
htop

# Check disk space (models take ~150GB)
df -h

# View logs
tail -f logs/spark_vtuber.log
```

### Backup Data

```bash
# Backup memory database
tar -czf backup-$(date +%Y%m%d).tar.gz data/chroma

# Backup configuration
cp .env .env.backup
```

### Clear Cache

```bash
# Clear ChromaDB
rm -rf data/chroma/*

# Clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} +

# Clear model cache (CAREFUL - requires re-download)
rm -rf ~/.cache/huggingface
```

---

## Next Steps

1. **Read the audit report**: `docs/AUDIT_REPORT.md`
2. **Review known issues**: Critical fixes needed before production
3. **Customize personalities**: Edit personality configs
4. **Fine-tune LoRA adapters**: Train on your personality data
5. **Join the community**: Contribute improvements!

---

## Additional Resources

- **Project Documentation**: `docs/`
- **Research Reports**: `research/reports/`
- **API Reference**: (coming soon)
- **GitHub Issues**: https://github.com/jhacksman/spark-vtuber/issues

---

## Support

For issues:
1. Check this guide's troubleshooting section
2. Review `docs/AUDIT_REPORT.md` for known issues
3. Open a GitHub issue with logs and error messages

**Log Location**: `logs/spark_vtuber.log`

**Include in bug reports:**
```bash
# System info
uname -a
nvidia-smi
python --version
uv --version

# Configuration (remove tokens!)
cat .env | grep -v TOKEN | grep -v OAUTH

# Recent logs
tail -n 100 logs/spark_vtuber.log
```
