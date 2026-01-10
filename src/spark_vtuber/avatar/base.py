"""
Base avatar interface for Spark VTuber.

Provides abstract base class for avatar control implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

from spark_vtuber.utils.logging import LoggerMixin


class Emotion(str, Enum):
    """Avatar emotion states."""

    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    THINKING = "thinking"
    EXCITED = "excited"
    CONFUSED = "confused"


@dataclass
class LipSyncFrame:
    """A single frame of lip sync data."""

    phoneme: str
    intensity: float = 1.0
    duration_ms: float = 50.0


@dataclass
class ExpressionState:
    """Current expression state of the avatar."""

    emotion: Emotion = Emotion.NEUTRAL
    intensity: float = 1.0
    blend_shapes: dict[str, float] | None = None


class BaseAvatar(ABC, LoggerMixin):
    """
    Abstract base class for avatar control implementations.

    Provides interface for:
    - Lip sync control
    - Expression/emotion control
    - Parameter manipulation
    - Animation triggers
    """

    def __init__(self, **kwargs):
        """Initialize the avatar controller."""
        self._connected = False
        self._current_expression = ExpressionState()

    @property
    def is_connected(self) -> bool:
        """Check if connected to avatar software."""
        return self._connected

    @property
    def current_expression(self) -> ExpressionState:
        """Get current expression state."""
        return self._current_expression

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the avatar software."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the avatar software."""
        pass

    @abstractmethod
    async def set_parameter(self, name: str, value: float) -> None:
        """
        Set a single avatar parameter.

        Args:
            name: Parameter name
            value: Parameter value (typically 0-1)
        """
        pass

    @abstractmethod
    async def set_parameters(self, parameters: dict[str, float]) -> None:
        """
        Set multiple avatar parameters at once.

        Args:
            parameters: Dictionary of parameter names to values
        """
        pass

    @abstractmethod
    async def get_parameter(self, name: str) -> float | None:
        """
        Get current value of a parameter.

        Args:
            name: Parameter name

        Returns:
            Current value or None if not found
        """
        pass

    @abstractmethod
    async def get_available_parameters(self) -> list[str]:
        """Get list of available parameter names."""
        pass

    @abstractmethod
    async def set_expression(self, emotion: Emotion, intensity: float = 1.0) -> None:
        """
        Set the avatar's expression.

        Args:
            emotion: Target emotion
            intensity: Expression intensity (0-1)
        """
        pass

    @abstractmethod
    async def update_lip_sync(self, frame: LipSyncFrame) -> None:
        """
        Update lip sync with a single frame.

        Args:
            frame: Lip sync frame data
        """
        pass

    @abstractmethod
    async def trigger_animation(self, animation_name: str) -> None:
        """
        Trigger a named animation.

        Args:
            animation_name: Name of the animation to trigger
        """
        pass

    @abstractmethod
    async def get_available_animations(self) -> list[str]:
        """Get list of available animation names."""
        pass

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
