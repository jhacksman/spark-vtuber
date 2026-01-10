"""
Coqui TTS implementation for Spark VTuber.

Provides high-quality TTS with voice cloning support.
"""

import asyncio
import time
from typing import AsyncIterator

import numpy as np

from spark_vtuber.tts.base import BaseTTS, TTSResult


class CoquiTTS(BaseTTS):
    """
    Coqui TTS implementation.

    Supports:
    - Multiple TTS models (Tacotron2, VITS, XTTS)
    - Voice cloning
    - Streaming synthesis
    - Emotion control (model-dependent)
    """

    def __init__(
        self,
        model_name: str = "tts_models/en/ljspeech/tacotron2-DDC",
        sample_rate: int = 22050,
        use_cuda: bool = True,
        **kwargs,
    ):
        """
        Initialize Coqui TTS.

        Args:
            model_name: Coqui TTS model name
            sample_rate: Output sample rate
            use_cuda: Whether to use GPU acceleration
            **kwargs: Additional arguments
        """
        super().__init__(sample_rate, **kwargs)
        self.model_name = model_name
        self.use_cuda = use_cuda
        self._tts = None
        self._cloned_voices: dict[str, str] = {}

    async def load(self) -> None:
        """Load the TTS model."""
        if self._loaded:
            self.logger.warning("TTS already loaded")
            return

        self.logger.info(f"Loading TTS model: {self.model_name}")
        start_time = time.time()

        try:
            from TTS.api import TTS

            self._tts = TTS(model_name=self.model_name, gpu=self.use_cuda)
            self._loaded = True

            load_time = time.time() - start_time
            self.logger.info(f"TTS loaded in {load_time:.2f}s")

        except Exception as e:
            self.logger.error(f"Failed to load TTS: {e}")
            raise

    async def unload(self) -> None:
        """Unload the TTS model."""
        if not self._loaded:
            return

        self.logger.info("Unloading TTS")

        if self._tts:
            del self._tts
            self._tts = None

        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._loaded = False

    async def synthesize(
        self,
        text: str,
        voice_id: str | None = None,
        speed: float = 1.0,
        emotion: str | None = None,
    ) -> TTSResult:
        """Synthesize speech from text."""
        if not self._loaded:
            raise RuntimeError("TTS not loaded")

        start_time = time.time()

        speaker_wav = None
        if voice_id and voice_id in self._cloned_voices:
            speaker_wav = self._cloned_voices[voice_id]

        loop = asyncio.get_event_loop()
        audio = await loop.run_in_executor(
            None,
            lambda: self._synthesize_sync(text, speaker_wav, speed),
        )

        latency_ms = (time.time() - start_time) * 1000
        duration = len(audio) / self.sample_rate

        return TTSResult(
            audio=audio,
            sample_rate=self.sample_rate,
            duration_seconds=duration,
            latency_ms=latency_ms,
        )

    def _synthesize_sync(
        self,
        text: str,
        speaker_wav: str | None,
        speed: float,
    ) -> np.ndarray:
        """Synchronous synthesis."""
        if speaker_wav:
            audio = self._tts.tts(
                text=text,
                speaker_wav=speaker_wav,
                language="en",
            )
        else:
            audio = self._tts.tts(text=text)

        return np.array(audio, dtype=np.float32)

    async def synthesize_stream(
        self,
        text: str,
        voice_id: str | None = None,
        speed: float = 1.0,
        emotion: str | None = None,
        chunk_size: int = 4096,
    ) -> AsyncIterator[np.ndarray]:
        """
        Synthesize speech with streaming output.

        WARNING: This is FAKE streaming - Coqui TTS doesn't natively support
        true streaming, so we synthesize the FULL audio first and then chunk it.
        This means latency = full synthesis time (200-500ms per sentence).

        For true streaming with low latency (<100ms first chunk), use StyleTTS2:
            from spark_vtuber.tts.styletts2 import StyleTTS2

        See: https://github.com/yl4579/StyleTTS2
        """
        import warnings
        warnings.warn(
            "CoquiTTS.synthesize_stream() is fake streaming - full audio is "
            "synthesized before chunking. For true streaming, use StyleTTS2.",
            UserWarning,
            stacklevel=2,
        )

        result = await self.synthesize(text, voice_id, speed, emotion)
        audio = result.audio

        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            yield chunk
            await asyncio.sleep(0)

    async def clone_voice(
        self,
        reference_audio: np.ndarray,
        voice_id: str,
    ) -> None:
        """Clone a voice from reference audio."""
        import tempfile
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, reference_audio, self.sample_rate)
            self._cloned_voices[voice_id] = f.name

        self.logger.info(f"Cloned voice saved as: {voice_id}")

    def get_available_voices(self) -> list[str]:
        """Get list of available voice IDs."""
        voices = list(self._cloned_voices.keys())
        if self._tts and hasattr(self._tts, "speakers"):
            voices.extend(self._tts.speakers or [])
        return voices
