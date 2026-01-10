"""
Base LLM interface for Spark VTuber.

Provides abstract base class for LLM implementations with streaming support.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator

from spark_vtuber.utils.logging import LoggerMixin


@dataclass
class LLMResponse:
    """Response from LLM generation."""

    text: str
    tokens_generated: int = 0
    finish_reason: str = "stop"
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


class BaseLLM(ABC, LoggerMixin):
    """
    Abstract base class for LLM implementations.

    Provides interface for both streaming and non-streaming generation.
    """

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the LLM.

        Args:
            model_name: Name or path of the model
            **kwargs: Additional model-specific arguments
        """
        self.model_name = model_name
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    @abstractmethod
    async def load(self) -> None:
        """Load the model into memory."""
        pass

    @abstractmethod
    async def unload(self) -> None:
        """Unload the model from memory."""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_sequences: list[str] | None = None,
    ) -> LLMResponse:
        """
        Generate a complete response.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop_sequences: Sequences that stop generation

        Returns:
            LLMResponse with generated text
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_sequences: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop_sequences: Sequences that stop generation

        Yields:
            Generated tokens as they become available
        """
        pass

    @abstractmethod
    async def load_lora_adapter(self, adapter_path: str, adapter_name: str) -> None:
        """
        Load a LoRA adapter for personality switching.

        Args:
            adapter_path: Path to the LoRA adapter
            adapter_name: Name identifier for the adapter
        """
        pass

    @abstractmethod
    async def set_active_adapter(self, adapter_name: str | None) -> None:
        """
        Set the active LoRA adapter.

        Args:
            adapter_name: Name of adapter to activate, or None for base model
        """
        pass

    @abstractmethod
    def get_memory_usage(self) -> dict[str, float]:
        """
        Get current memory usage.

        Returns:
            Dictionary with memory usage in GB
        """
        pass

    async def __aenter__(self):
        """Async context manager entry."""
        await self.load()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.unload()
