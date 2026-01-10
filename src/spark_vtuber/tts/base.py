"""
Base TTS interface for Spark VTuber.

Provides abstract base class for TTS implementations with streaming support.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator

import numpy as np

from spark_vtuber.utils.logging import LoggerMixin


@dataclass
class TTSResult:
    """Result from TTS synthesis."""

    audio: np.ndarray
    sample_rate: int
    duration_seconds: float
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


class BaseTTS(ABC, LoggerMixin):
    """
    Abstract base class for TTS implementations.

    Provides interface for both streaming and non-streaming synthesis.
    """

    def __init__(self, sample_rate: int = 22050, **kwargs):
        """
        Initialize the TTS engine.

        Args:
            sample_rate: Output audio sample rate
            **kwargs: Additional engine-specific arguments
        """
        self.sample_rate = sample_rate
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    @abstractmethod
    async def load(self) -> None:
        """Load the TTS model into memory."""
        pass

    @abstractmethod
    async def unload(self) -> None:
        """Unload the TTS model from memory."""
        pass

    @abstractmethod
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
            voice_id: Optional voice identifier
            speed: Speech speed multiplier
            emotion: Optional emotion tag

        Returns:
            TTSResult with audio data
        """
        pass

    @abstractmethod
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

        Args:
            text: Text to synthesize
            voice_id: Optional voice identifier
            speed: Speech speed multiplier
            emotion: Optional emotion tag
            chunk_size: Audio chunk size in samples

        Yields:
            Audio chunks as numpy arrays
        """
        pass

    @abstractmethod
    async def clone_voice(
        self,
        reference_audio: np.ndarray,
        voice_id: str,
    ) -> None:
        """
        Clone a voice from reference audio.

        Args:
            reference_audio: Reference audio samples
            voice_id: Identifier for the cloned voice
        """
        pass

    @abstractmethod
    def get_available_voices(self) -> list[str]:
        """Get list of available voice IDs."""
        pass

    def get_memory_usage(self) -> dict[str, float]:
        """Get current memory usage."""
        import torch

        if not torch.cuda.is_available():
            return {"gpu_allocated_gb": 0}

        allocated = torch.cuda.memory_allocated() / (1024**3)
        return {"gpu_allocated_gb": round(allocated, 2)}

    async def __aenter__(self):
        """Async context manager entry."""
        await self.load()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.unload()
