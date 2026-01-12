# Fish Speech Installation for DGX Spark

This directory contains scripts to install and run Fish Speech TTS on NVIDIA DGX Spark (Blackwell GB10).

## Overview

Fish Speech is a high-quality text-to-speech model that supports:
- Zero-shot voice cloning
- Multiple languages
- Emotion control
- Streaming audio generation

**Model:** fishaudio/openaudio-s1-mini (~12GB VRAM)

## Quick Start

### 1. Install Fish Speech

```bash
./install_fish_speech.sh --install-dir ./fish-speech-install
```

This will:
- Install system dependencies (portaudio, sox, ffmpeg)
- Clone Fish Speech repository
- Create Python virtual environment
- Install PyTorch with CUDA support
- Install Fish Speech package
- Download the model (~12GB)

Installation takes approximately 15-20 minutes.

### 2. Start the API Server

```bash
./fish-speech-serve.sh 8843
```

The server will start on `http://0.0.0.0:8843`.

### 3. Test the API

```bash
curl -X POST http://localhost:8843/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, this is a test of Fish Speech TTS."}' \
  --output test.wav
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/tts` | POST | Generate speech from text |
| `/v1/health` | GET | Health check |
| `/v1/references/list` | GET | List available voice references |
| `/v1/references/add` | POST | Add a new voice reference |
| `/v1/references/delete` | DELETE | Delete a voice reference |

### TTS Request Format

```json
{
  "text": "Text to synthesize",
  "reference_id": "optional_voice_id",
  "format": "wav",
  "streaming": false
}
```

### Response

Returns audio file in the requested format (default: WAV).

## Installation Options

```bash
./install_fish_speech.sh [OPTIONS]

Options:
  --install-dir DIR    Installation directory (default: $PWD/fish-speech-install)
  --python-version VER Python version (default: 3.12)
  --skip-model         Skip model download
  --help               Show help message
```

## Directory Structure

After installation:

```
fish-speech-install/
├── .fish-speech/           # Python virtual environment
├── fish-speech/            # Fish Speech source code
│   ├── checkpoints/        # Model weights
│   │   └── openaudio-s1-mini/
│   ├── fish_speech/        # Python package
│   ├── tools/              # API server and utilities
│   └── references/         # Voice reference files (optional)
```

## VRAM Requirements

- Fish Speech model: ~12GB VRAM
- With vLLM (Qwen3-30B-A3B): ~32GB total
- DGX Spark has 64GB unified memory, so both can run simultaneously

## Troubleshooting

### CUDA not found

Ensure CUDA is in your PATH:
```bash
export PATH="/usr/local/cuda/bin:$PATH"
```

### Model download fails

Download manually:
```bash
source fish-speech-install/.fish-speech/bin/activate
cd fish-speech-install/fish-speech
python -c "
from huggingface_hub import snapshot_download
snapshot_download('fishaudio/openaudio-s1-mini', local_dir='checkpoints/openaudio-s1-mini')
"
```

### Server won't start

Check that the model exists:
```bash
ls -la fish-speech-install/fish-speech/checkpoints/openaudio-s1-mini/
```

### Audio quality issues

Try adding `--compile` flag for optimized inference (enabled by default in fish-speech-serve.sh).

## Integration with spark-vtuber

The Fish Speech API server integrates with the spark-vtuber pipeline:

1. vLLM generates text response (port 8000)
2. Fish Speech converts text to audio (port 8843)
3. Audio plays through the avatar system

See `src/spark_vtuber/tts/fish_speech.py` for the Python integration.
