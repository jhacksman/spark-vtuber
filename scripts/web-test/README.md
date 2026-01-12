# TTS Test Interface

A simple web interface for testing the vLLM + Fish Speech TTS pipeline on DGX Spark.

## Overview

This tool provides a browser-based interface to:
1. Submit text questions (simulating Twitch donations)
2. Send questions to vLLM for LLM response generation
3. Strip `<think>` tags from the response
4. Send the response to Fish Speech for TTS audio synthesis
5. Play the audio in the browser

## Prerequisites

Before using this tool, ensure:
1. vLLM server is running (default: port 8000)
2. Fish Speech API server is running (default: port 8843)

## Quick Start

```bash
# Install dependencies (if not already installed)
pip install fastapi uvicorn httpx

# Start the web interface
python server.py --port 8844

# Open in browser
# http://localhost:8844
```

## Usage

### Command Line Options

```bash
python server.py [OPTIONS]

Options:
  --port PORT        Web server port (default: 8844)
  --host HOST        Web server host (default: 0.0.0.0)
  --vllm-url URL     vLLM server URL (default: http://localhost:8000)
  --tts-url URL      Fish Speech TTS server URL (default: http://localhost:8843)
  --model NAME       vLLM model name (default: qwen3-30b-a3b-awq)
```

### Example

```bash
# Start with custom ports
python server.py \
  --port 8844 \
  --vllm-url http://localhost:8000 \
  --tts-url http://localhost:8843 \
  --model "qwen3-30b-a3b-awq"
```

## Features

- **Status indicators**: Shows connection status for vLLM and Fish Speech servers
- **Think tag stripping**: Optionally removes `<think>...</think>` tags from LLM responses
- **Timing statistics**: Displays LLM and TTS latency in milliseconds
- **Debug panel**: Collapsible panel showing raw LLM response (including think tags)
- **Audio playback**: Built-in audio player for generated speech
- **Dark theme**: Easy on the eyes for extended testing sessions

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/health` | GET | Server health check |
| `/api/chat` | POST | Full LLM -> TTS pipeline |
| `/api/tts` | POST | TTS only (bypass LLM) |
| `/api/vllm/health` | GET | Check vLLM server status |
| `/api/tts/health` | GET | Check Fish Speech server status |

### Chat API Request

```json
POST /api/chat
{
  "message": "Hello, who are you?",
  "strip_think_tags": true
}
```

### Chat API Response

```json
{
  "llm_response": "I am an AI assistant...",
  "llm_raw_response": "<think>Let me think...</think>I am an AI assistant...",
  "llm_latency_ms": 1234.56,
  "tts_latency_ms": 567.89,
  "audio_base64": "UklGRi...",
  "audio_format": "wav"
}
```

## Troubleshooting

### vLLM status shows error

1. Check if vLLM server is running:
   ```bash
   curl http://localhost:8000/health
   ```
2. Verify the model is loaded:
   ```bash
   curl http://localhost:8000/v1/models
   ```

### Fish Speech status shows error

1. Check if Fish Speech server is running:
   ```bash
   curl http://localhost:8843/v1/health
   ```
2. Verify the model is loaded (check server logs)

### Audio doesn't play

1. Check browser console for errors
2. Verify Fish Speech is generating audio correctly:
   ```bash
   curl -X POST http://localhost:8843/v1/tts \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello world"}' \
     --output test.wav
   ```

### Slow response times

- LLM latency > 5000ms: Model may be loading or GPU is busy
- TTS latency > 3000ms: Fish Speech model may be loading

## Architecture

```
Browser (port 8844)
    |
    v
FastAPI Server (server.py)
    |
    +---> vLLM (port 8000) - LLM inference
    |
    +---> Fish Speech (port 8843) - TTS synthesis
    |
    v
Audio playback in browser
```

## Notes

- This is a **temporary test tool** for debugging purposes
- Not intended for production use
- Audio is returned as base64-encoded WAV for simplicity
- The interface is accessible on LAN when bound to 0.0.0.0
