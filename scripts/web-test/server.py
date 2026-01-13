#!/usr/bin/env python3
"""
Simple web test interface for vLLM + Fish Speech TTS pipeline.

This is a temporary test tool for debugging the LLM -> TTS -> Audio pipeline.
Not intended for production use.

Usage:
    python server.py [--port 8844] [--vllm-url http://localhost:8000] [--tts-url http://localhost:8843]
"""

import argparse
import asyncio
import io
import re
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="TTS Test Interface", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration (set via command line args)
VLLM_URL = "http://localhost:8000"
TTS_URL = "http://localhost:8843"
MODEL_NAME = "qwen3-30b-a3b-awq"


class ChatRequest(BaseModel):
    message: str
    strip_think_tags: bool = True
    vllm_url: str | None = None
    tts_url: str | None = None
    model_name: str | None = None


class ChatResponse(BaseModel):
    llm_response: str
    llm_raw_response: str
    llm_latency_ms: float
    tts_latency_ms: float
    audio_url: str


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> tags from LLM response."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main HTML page."""
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "vllm_url": VLLM_URL, "tts_url": TTS_URL}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Process a chat message through the LLM -> TTS pipeline.
    
    1. Send message to vLLM for LLM response
    2. Strip think tags if requested
    3. Send response to Fish Speech for TTS
    4. Return audio URL and timing stats
    """
    # Use request URLs if provided, otherwise fall back to defaults
    vllm_url = request.vllm_url or VLLM_URL
    tts_url = request.tts_url or TTS_URL
    model_name = request.model_name or MODEL_NAME
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        # Step 1: Get LLM response from vLLM
        llm_start = time.time()
        try:
            llm_response = await client.post(
                f"{vllm_url}/v1/chat/completions",
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "user", "content": request.message}
                    ],
                    "max_tokens": 512,
                    "temperature": 0.7,
                },
            )
            llm_response.raise_for_status()
            llm_data = llm_response.json()
            llm_raw_text = llm_data["choices"][0]["message"]["content"]
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"vLLM error: {str(e)}")
        except (KeyError, IndexError) as e:
            raise HTTPException(status_code=502, detail=f"Invalid vLLM response: {str(e)}")
        
        llm_latency = (time.time() - llm_start) * 1000

        # Step 2: Strip think tags if requested
        if request.strip_think_tags:
            llm_text = strip_think_tags(llm_raw_text)
        else:
            llm_text = llm_raw_text

        # Step 3: Generate TTS audio
        tts_start = time.time()
        try:
            tts_response = await client.post(
                f"{tts_url}/v1/tts",
                json={
                    "text": llm_text,
                    "format": "wav",
                    "streaming": False,
                },
            )
            tts_response.raise_for_status()
            audio_data = tts_response.content
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"TTS error: {str(e)}")
        
        tts_latency = (time.time() - tts_start) * 1000

        # Step 4: Store audio and return response
        # For simplicity, we'll return the audio as base64 in the response
        import base64
        audio_base64 = base64.b64encode(audio_data).decode("utf-8")

        return {
            "llm_response": llm_text,
            "llm_raw_response": llm_raw_text,
            "llm_latency_ms": round(llm_latency, 2),
            "tts_latency_ms": round(tts_latency, 2),
            "audio_base64": audio_base64,
            "audio_format": "wav",
        }


@app.post("/api/tts")
async def tts_only(text: str):
    """Generate TTS audio for given text (bypass LLM)."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            tts_response = await client.post(
                f"{TTS_URL}/v1/tts",
                json={
                    "text": text,
                    "format": "wav",
                    "streaming": False,
                },
            )
            tts_response.raise_for_status()
            return StreamingResponse(
                io.BytesIO(tts_response.content),
                media_type="audio/wav",
                headers={"Content-Disposition": "attachment; filename=tts.wav"},
            )
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"TTS error: {str(e)}")


@app.get("/api/vllm/health")
async def vllm_health(url: str | None = None):
    """Check vLLM server health."""
    vllm_url = url or VLLM_URL
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            response = await client.get(f"{vllm_url}/health")
            return {"status": "ok", "vllm_status": response.status_code}
        except httpx.HTTPError as e:
            return {"status": "error", "error": str(e)}


@app.get("/api/vllm/models")
async def vllm_models(url: str | None = None):
    """Get available models from vLLM server."""
    vllm_url = url or VLLM_URL
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(f"{vllm_url}/v1/models")
            response.raise_for_status()
            data = response.json()
            models = [m["id"] for m in data.get("data", [])]
            return {"status": "ok", "models": models}
        except httpx.HTTPError as e:
            return {"status": "error", "error": str(e), "models": []}


@app.get("/api/tts/health")
async def tts_health(url: str | None = None):
    """Check Fish Speech TTS server health."""
    tts_url = url or TTS_URL
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            response = await client.get(f"{tts_url}/v1/health")
            return {"status": "ok", "tts_status": response.status_code}
        except httpx.HTTPError as e:
            return {"status": "error", "error": str(e)}


def main():
    global VLLM_URL, TTS_URL, MODEL_NAME
    
    parser = argparse.ArgumentParser(description="TTS Test Interface Server")
    parser.add_argument("--port", type=int, default=8844, help="Server port (default: 8844)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000", help="vLLM server URL")
    parser.add_argument("--tts-url", type=str, default="http://localhost:8843", help="Fish Speech TTS server URL")
    parser.add_argument("--model", type=str, default="qwen3-30b-a3b-awq", help="vLLM model name")
    args = parser.parse_args()

    VLLM_URL = args.vllm_url
    TTS_URL = args.tts_url
    MODEL_NAME = args.model

    print(f"Starting TTS Test Interface on http://{args.host}:{args.port}")
    print(f"vLLM URL: {VLLM_URL}")
    print(f"TTS URL: {TTS_URL}")
    print(f"Model: {MODEL_NAME}")
    print()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
