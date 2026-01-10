"""
Base game interface for Spark VTuber.

Provides abstract base class for game integrations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator

from spark_vtuber.utils.logging import LoggerMixin


class GameStatus(str, Enum):
    """Game connection status."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    PLAYING = "playing"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class GameState:
    """Current state of the game."""

    status: GameStatus = GameStatus.DISCONNECTED
    game_name: str = ""
    current_scene: str = ""
    player_position: tuple[float, float, float] | None = None
    player_health: float = 1.0
    player_inventory: list[str] = field(default_factory=list)
    current_objective: str | None = None
    score: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

    def to_prompt_context(self) -> str:
        """Convert state to context for LLM prompt."""
        parts = [f"Game: {self.game_name}"]

        if self.current_scene:
            parts.append(f"Scene: {self.current_scene}")

        if self.player_position:
            x, y, z = self.player_position
            parts.append(f"Position: ({x:.1f}, {y:.1f}, {z:.1f})")

        if self.player_health < 1.0:
            parts.append(f"Health: {self.player_health * 100:.0f}%")

        if self.player_inventory:
            parts.append(f"Inventory: {', '.join(self.player_inventory[:5])}")

        if self.current_objective:
            parts.append(f"Objective: {self.current_objective}")

        if self.score > 0:
            parts.append(f"Score: {self.score}")

        return " | ".join(parts)


@dataclass
class GameAction:
    """An action to perform in the game."""

    action_type: str
    parameters: dict = field(default_factory=dict)
    description: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


class BaseGame(ABC, LoggerMixin):
    """
    Abstract base class for game integrations.

    Provides interface for:
    - Game connection and state management
    - Action execution
    - Screen capture for vision
    - Event handling
    """

    def __init__(self, game_name: str, **kwargs):
        """
        Initialize the game integration.

        Args:
            game_name: Name of the game
            **kwargs: Additional arguments
        """
        self.game_name = game_name
        self._state = GameState(game_name=game_name)
        self._action_history: list[GameAction] = []

    @property
    def state(self) -> GameState:
        """Get current game state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connected to game."""
        return self._state.status in [GameStatus.CONNECTED, GameStatus.PLAYING]

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the game."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the game."""
        pass

    @abstractmethod
    async def get_state(self) -> GameState:
        """
        Get current game state.

        Returns:
            Current GameState
        """
        pass

    @abstractmethod
    async def execute_action(self, action: GameAction) -> bool:
        """
        Execute an action in the game.

        Args:
            action: Action to execute

        Returns:
            True if action was successful
        """
        pass

    @abstractmethod
    async def get_available_actions(self) -> list[str]:
        """
        Get list of available actions.

        Returns:
            List of action type names
        """
        pass

    @abstractmethod
    async def capture_screen(self) -> bytes | None:
        """
        Capture the current game screen.

        Returns:
            Screenshot as PNG bytes, or None if unavailable
        """
        pass

    async def execute_actions(self, actions: list[GameAction]) -> list[bool]:
        """
        Execute multiple actions in sequence.

        Args:
            actions: List of actions to execute

        Returns:
            List of success/failure for each action
        """
        results = []
        for action in actions:
            result = await self.execute_action(action)
            results.append(result)
            self._action_history.append(action)
        return results

    def get_action_history(self, limit: int = 10) -> list[GameAction]:
        """Get recent action history."""
        return self._action_history[-limit:]

    def clear_action_history(self) -> None:
        """Clear action history."""
        self._action_history.clear()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
