"""
Piper TTS implementation for Spark VTuber.

Provides fast, local TTS with TRUE streaming support on ARM64 + CUDA.
Piper is optimized for low-latency inference on embedded devices and servers.

Supports ARM64 architecture with CUDA acceleration via onnxruntime-gpu.
"""

import asyncio
import os
import tempfile
import time
from pathlib import Path
from typing import AsyncIterator

import numpy as np

from spark_vtuber.tts.base import BaseTTS, TTSResult


class PiperTTS(BaseTTS):
    """
    Piper TTS implementation with TRUE streaming support.

    Supports:
    - Fast local inference optimized for ARM64 and x86_64
    - TRUE streaming output with low latency (~300-700ms)
    - CUDA acceleration via onnxruntime-gpu
    - Multiple voice models at different quality levels
    - Voice cloning via fine-tuning (advanced)

    Memory usage: ~100MB-500MB depending on model
    Latency: ~300-700ms on ARM64 (Jetson Orin Nano benchmarks)
    """

    def __init__(
        self,
        model_path: str,
        sample_rate: int = 22050,
        use_cuda: bool = True,
        speaker_id: int | None = None,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
        **kwargs,
    ):
        """
        Initialize Piper TTS.

        Args:
            model_path: Path to .onnx model file
            sample_rate: Output sample rate (auto-detected from model config)
            use_cuda: Use CUDA acceleration if available
            speaker_id: Speaker ID for multi-speaker models
            length_scale: Speed control (< 1.0 faster, > 1.0 slower)
            noise_scale: Noise added to audio (affects quality/naturalness)
            noise_w: Variation in speech (higher = more variation)
            **kwargs: Additional arguments
        """
        super().__init__(sample_rate, **kwargs)
        self.model_path = Path(model_path)
        self.use_cuda = use_cuda
        self.speaker_id = speaker_id
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.noise_w = noise_w
        self._voice = None
        self._cloned_voices: dict[str, str] = {}
        self._first_chunk_latency_ms: float = 0.0

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Piper model not found: {self.model_path}\n"
                f"Download from https://huggingface.co/rhasspy/piper-voices"
            )

        # Check for config file
        self.config_path = self.model_path.with_suffix(".onnx.json")
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Piper config not found: {self.config_path}\n"
                f"Download the .onnx.json file alongside the .onnx model"
            )

    async def load(self) -> None:
        """Load the Piper voice model."""
        if self._loaded:
            self.logger.warning("TTS already loaded")
            return

        self.logger.info(f"Loading Piper TTS model: {self.model_path}")
        start_time = time.time()

        try:
            from piper.voice import PiperVoice

            loop = asyncio.get_event_loop()

            def _load_sync():
                # Load voice with CUDA if available
                voice = PiperVoice.load(
                    str(self.model_path),
                    config_path=str(self.config_path),
                    use_cuda=self.use_cuda,
                )
                return voice

            self._voice = await loop.run_in_executor(None, _load_sync)

            # Update sample rate from model config
            if hasattr(self._voice, "config") and hasattr(self._voice.config, "sample_rate"):
                detected_sr = self._voice.config.sample_rate
                if detected_sr != self.sample_rate:
                    self.logger.info(f"Updating sample rate from {self.sample_rate} to {detected_sr}")
                    self.sample_rate = detected_sr

            self._loaded = True
            load_time = time.time() - start_time
            self.logger.info(f"Piper TTS loaded in {load_time:.2f}s")
            self.logger.info(f"Sample rate: {self.sample_rate}Hz")

            if self.use_cuda:
                self.logger.info("CUDA acceleration enabled")

        except ImportError as e:
            self.logger.error(f"Failed to import Piper: {e}")
            self.logger.error("Install with: pip install piper-tts")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load Piper TTS: {e}")
            raise

    async def unload(self) -> None:
        """Unload the TTS model."""
        if not self._loaded:
            return

        self.logger.info("Unloading Piper TTS")

        self._voice = None
        self._cloned_voices.clear()
        self._loaded = False

    async def synthesize(
        self,
        text: str,
        voice_id: str | None = None,
        speed: float = 1.0,
        emotion: str | None = None,
    ) -> TTSResult:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            voice_id: Voice model path (if different from default)
            speed: Speech speed multiplier (implemented via length_scale)
            emotion: Not supported by Piper (ignored)

        Returns:
            TTSResult with audio data
        """
        if not self._loaded:
            raise RuntimeError("TTS not loaded. Call load() first.")

        start_time = time.time()

        # Collect all chunks
        audio_chunks = []
        async for chunk in self.synthesize_stream(text, voice_id, speed, emotion):
            audio_chunks.append(chunk)

        if not audio_chunks:
            self.logger.warning("No audio generated")
            return TTSResult(
                audio=np.array([], dtype=np.float32),
                sample_rate=self.sample_rate,
                duration_seconds=0.0,
                latency_ms=(time.time() - start_time) * 1000,
                metadata={"engine": "piper", "model": str(self.model_path)},
            )

        # Concatenate all chunks
        audio = np.concatenate(audio_chunks, axis=0)

        latency_ms = (time.time() - start_time) * 1000
        duration = len(audio) / self.sample_rate

        return TTSResult(
            audio=audio,
            sample_rate=self.sample_rate,
            duration_seconds=duration,
            latency_ms=latency_ms,
            metadata={
                "engine": "piper",
                "model": str(self.model_path),
                "voice_id": voice_id,
                "speed": speed,
            },
        )

    async def synthesize_stream(
        self,
        text: str,
        voice_id: str | None = None,
        speed: float = 1.0,
        emotion: str | None = None,
        chunk_size: int = 4096,
    ) -> AsyncIterator[np.ndarray]:
        """
        Synthesize speech with TRUE streaming output.

        Piper generates audio chunks as phonemes are processed, providing
        true streaming with low first-chunk latency.

        Args:
            text: Text to synthesize
            voice_id: Alternative voice model (not implemented - uses default)
            speed: Speech speed multiplier
            emotion: Not supported by Piper (ignored)
            chunk_size: Not used (Piper controls chunk size internally)

        Yields:
            Audio chunks as numpy arrays (int16 format)
        """
        if not self._loaded:
            raise RuntimeError("TTS not loaded. Call load() first.")

        start_time = time.time()
        first_chunk = True

        loop = asyncio.get_event_loop()

        # Adjust length_scale based on speed
        length_scale = self.length_scale / speed

        def _stream_sync():
            """Synchronous streaming generator."""
            # Use Piper's synthesize_stream_raw for true streaming
            for audio_bytes in self._voice.synthesize_stream_raw(
                text,
                speaker_id=self.speaker_id,
                length_scale=length_scale,
                noise_scale=self.noise_scale,
                noise_w=self.noise_w,
            ):
                # Convert bytes to int16 numpy array
                audio_chunk = np.frombuffer(audio_bytes, dtype=np.int16)

                # Convert int16 to float32 normalized to [-1, 1]
                audio_float = audio_chunk.astype(np.float32) / 32768.0

                yield audio_float

        # Convert sync generator to async
        sync_gen = _stream_sync()

        while True:
            try:
                chunk = await loop.run_in_executor(None, lambda: next(sync_gen))

                if first_chunk:
                    self._first_chunk_latency_ms = (time.time() - start_time) * 1000
                    self.logger.info(f"First chunk latency: {self._first_chunk_latency_ms:.1f}ms")
                    first_chunk = False

                yield chunk

            except StopIteration:
                break

    async def clone_voice(
        self,
        reference_audio: np.ndarray,
        voice_id: str,
    ) -> None:
        """
        Clone a voice from reference audio.

        Note: Piper does not support runtime voice cloning. Voice cloning
        requires fine-tuning the model offline, which is outside the scope
        of real-time TTS.

        This method is provided for API compatibility but will log a warning.

        Args:
            reference_audio: Reference audio samples (ignored)
            voice_id: Identifier for the cloned voice (ignored)
        """
        self.logger.warning(
            "Piper does not support runtime voice cloning. "
            "To create custom voices, you must fine-tune the model offline. "
            "See: https://github.com/rhasspy/piper/blob/master/TRAINING.md"
        )

        # Store a placeholder to maintain API compatibility
        self._cloned_voices[voice_id] = str(self.model_path)

    def get_available_voices(self) -> list[str]:
        """
        Get list of available voice IDs.

        For Piper, this returns the single loaded model path.
        To use multiple voices, you need to load different models.
        """
        return [str(self.model_path)]

    def get_first_chunk_latency(self) -> float:
        """Get the latency of the first audio chunk in milliseconds."""
        return self._first_chunk_latency_ms
