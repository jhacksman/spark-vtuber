"""
Base STT interface for Spark VTuber.

Provides abstract base class for STT implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator

import numpy as np

from spark_vtuber.utils.logging import LoggerMixin


@dataclass
class STTSegment:
    """A transcribed segment with timing information."""

    text: str
    start_time: float
    end_time: float
    confidence: float = 1.0
    speaker: str | None = None


@dataclass
class STTResult:
    """Result from STT transcription."""

    text: str
    segments: list[STTSegment] = field(default_factory=list)
    language: str = "en"
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


class BaseSTT(ABC, LoggerMixin):
    """
    Abstract base class for STT implementations.

    Provides interface for both batch and streaming transcription.
    """

    def __init__(self, language: str = "en", **kwargs):
        """
        Initialize the STT engine.

        Args:
            language: Target language for transcription
            **kwargs: Additional engine-specific arguments
        """
        self.language = language
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    @abstractmethod
    async def load(self) -> None:
        """Load the STT model into memory."""
        pass

    @abstractmethod
    async def unload(self) -> None:
        """Unload the STT model from memory."""
        pass

    @abstractmethod
    async def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> STTResult:
        """
        Transcribe audio to text.

        Args:
            audio: Audio samples as numpy array
            sample_rate: Audio sample rate

        Returns:
            STTResult with transcription
        """
        pass

    @abstractmethod
    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[np.ndarray],
        sample_rate: int = 16000,
    ) -> AsyncIterator[STTSegment]:
        """
        Transcribe streaming audio.

        Args:
            audio_stream: Async iterator of audio chunks
            sample_rate: Audio sample rate

        Yields:
            Transcribed segments as they become available
        """
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
