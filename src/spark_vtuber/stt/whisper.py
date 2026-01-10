"""
Faster-Whisper STT implementation for Spark VTuber.

Provides high-accuracy speech recognition with GPU acceleration.
"""

import asyncio
import time
from typing import AsyncIterator, Literal

import numpy as np

from spark_vtuber.stt.base import BaseSTT, STTResult, STTSegment


class WhisperSTT(BaseSTT):
    """
    Faster-Whisper STT implementation.

    Supports:
    - Multiple model sizes (tiny to large-v3)
    - GPU acceleration with CTranslate2
    - Voice activity detection
    - Streaming transcription
    """

    def __init__(
        self,
        model_size: Literal["tiny", "base", "small", "medium", "large-v3"] = "large-v3",
        device: str = "cuda",
        compute_type: Literal["float16", "int8", "int8_float16"] = "float16",
        language: str = "en",
        vad_enabled: bool = True,
        **kwargs,
    ):
        """
        Initialize Whisper STT.

        Args:
            model_size: Whisper model size
            device: Device for inference (cuda/cpu)
            compute_type: Compute type for CTranslate2
            language: Target language
            vad_enabled: Enable voice activity detection
            **kwargs: Additional arguments
        """
        super().__init__(language, **kwargs)
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.vad_enabled = vad_enabled
        self._model = None
        self._vad = None

    async def load(self) -> None:
        """Load the Whisper model."""
        if self._loaded:
            self.logger.warning("STT already loaded")
            return

        self.logger.info(f"Loading Whisper model: {self.model_size}")
        start_time = time.time()

        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )

            if self.vad_enabled:
                await self._load_vad()

            self._loaded = True

            load_time = time.time() - start_time
            self.logger.info(f"Whisper loaded in {load_time:.2f}s")

        except Exception as e:
            self.logger.error(f"Failed to load Whisper: {e}")
            raise

    async def _load_vad(self) -> None:
        """Load voice activity detection model."""
        try:
            import webrtcvad
            self._vad = webrtcvad.Vad(3)
            self.logger.info("WebRTC VAD loaded")
        except ImportError:
            self.logger.warning("webrtcvad not available, VAD disabled")
            self.vad_enabled = False

    async def unload(self) -> None:
        """Unload the Whisper model."""
        if not self._loaded:
            return

        self.logger.info("Unloading Whisper")

        if self._model:
            del self._model
            self._model = None

        self._vad = None

        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._loaded = False

    async def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> STTResult:
        """Transcribe audio to text."""
        if not self._loaded:
            raise RuntimeError("STT not loaded")

        start_time = time.time()

        if sample_rate != 16000:
            audio = self._resample(audio, sample_rate, 16000)

        loop = asyncio.get_event_loop()
        segments, info = await loop.run_in_executor(
            None,
            lambda: self._transcribe_sync(audio),
        )

        result_segments = []
        full_text = []

        for segment in segments:
            result_segments.append(STTSegment(
                text=segment.text.strip(),
                start_time=segment.start,
                end_time=segment.end,
                confidence=segment.avg_logprob,
            ))
            full_text.append(segment.text.strip())

        latency_ms = (time.time() - start_time) * 1000

        return STTResult(
            text=" ".join(full_text),
            segments=result_segments,
            language=info.language,
            latency_ms=latency_ms,
        )

    def _transcribe_sync(self, audio: np.ndarray):
        """Synchronous transcription."""
        segments, info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=5,
            vad_filter=self.vad_enabled,
        )
        return list(segments), info

    def _resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int,
    ) -> np.ndarray:
        """Resample audio to target sample rate."""
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[np.ndarray],
        sample_rate: int = 16000,
    ) -> AsyncIterator[STTSegment]:
        """Transcribe streaming audio."""
        if not self._loaded:
            raise RuntimeError("STT not loaded")

        buffer = np.array([], dtype=np.float32)
        chunk_duration = 3.0
        chunk_samples = int(sample_rate * chunk_duration)

        async for chunk in audio_stream:
            if sample_rate != 16000:
                chunk = self._resample(chunk, sample_rate, 16000)
                sample_rate = 16000

            buffer = np.concatenate([buffer, chunk])

            while len(buffer) >= chunk_samples:
                audio_chunk = buffer[:chunk_samples]
                buffer = buffer[chunk_samples // 2:]

                if self.vad_enabled and not self._has_speech(audio_chunk, sample_rate):
                    continue

                result = await self.transcribe(audio_chunk, sample_rate)
                for segment in result.segments:
                    yield segment

        if len(buffer) > sample_rate:
            result = await self.transcribe(buffer, sample_rate)
            for segment in result.segments:
                yield segment

    def _has_speech(self, audio: np.ndarray, sample_rate: int) -> bool:
        """Check if audio chunk contains speech using VAD."""
        if not self._vad:
            return True

        audio_int16 = (audio * 32767).astype(np.int16)
        frame_duration = 30
        frame_samples = int(sample_rate * frame_duration / 1000)

        speech_frames = 0
        total_frames = 0

        for i in range(0, len(audio_int16) - frame_samples, frame_samples):
            frame = audio_int16[i:i + frame_samples].tobytes()
            try:
                if self._vad.is_speech(frame, sample_rate):
                    speech_frames += 1
                total_frames += 1
            except Exception:
                pass

        if total_frames == 0:
            return True

        return speech_frames / total_frames > 0.1
