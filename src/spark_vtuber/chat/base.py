"""
Base chat interface for Spark VTuber.

Provides abstract base class for chat platform integrations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import AsyncIterator, Callable

from spark_vtuber.utils.logging import LoggerMixin


class MessageType(str, Enum):
    """Type of chat message."""

    CHAT = "chat"
    SUBSCRIPTION = "subscription"
    DONATION = "donation"
    RAID = "raid"
    COMMAND = "command"
    SYSTEM = "system"


@dataclass
class ChatMessage:
    """A single chat message."""

    id: str
    platform: str
    username: str
    display_name: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    message_type: MessageType = MessageType.CHAT
    is_moderator: bool = False
    is_subscriber: bool = False
    is_vip: bool = False
    badges: list[str] = field(default_factory=list)
    emotes: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    @property
    def is_command(self) -> bool:
        """Check if message is a command."""
        return self.content.startswith("!")

    @property
    def command_name(self) -> str | None:
        """Get command name if message is a command."""
        if not self.is_command:
            return None
        parts = self.content[1:].split()
        return parts[0].lower() if parts else None

    @property
    def command_args(self) -> list[str]:
        """Get command arguments if message is a command."""
        if not self.is_command:
            return []
        parts = self.content[1:].split()
        return parts[1:] if len(parts) > 1 else []


class BaseChat(ABC, LoggerMixin):
    """
    Abstract base class for chat platform integrations.

    Provides interface for:
    - Connecting to chat platforms
    - Receiving messages
    - Sending messages
    - Handling events (subs, donations, etc.)
    """

    def __init__(self, **kwargs):
        """Initialize the chat client."""
        self._connected = False
        self._message_callbacks: list[Callable] = []
        self._event_callbacks: dict[MessageType, list[Callable]] = {}

    @property
    def is_connected(self) -> bool:
        """Check if connected to chat."""
        return self._connected

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the chat platform."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the chat platform."""
        pass

    @abstractmethod
    async def send_message(self, content: str) -> None:
        """
        Send a message to chat.

        Args:
            content: Message content
        """
        pass

    @abstractmethod
    async def get_messages(self) -> AsyncIterator[ChatMessage]:
        """
        Get incoming messages.

        Yields:
            ChatMessage objects as they arrive
        """
        pass

    def on_message(self, callback: Callable[[ChatMessage], None]) -> None:
        """
        Register a callback for all messages.

        Args:
            callback: Function to call with ChatMessage
        """
        self._message_callbacks.append(callback)

    def on_event(
        self,
        event_type: MessageType,
        callback: Callable[[ChatMessage], None],
    ) -> None:
        """
        Register a callback for specific event types.

        Args:
            event_type: Type of event to listen for
            callback: Function to call with ChatMessage
        """
        if event_type not in self._event_callbacks:
            self._event_callbacks[event_type] = []
        self._event_callbacks[event_type].append(callback)

    async def _dispatch_message(self, message: ChatMessage) -> None:
        """Dispatch message to registered callbacks."""
        for callback in self._message_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(message)
                else:
                    callback(message)
            except Exception as e:
                self.logger.error(f"Message callback error: {e}")

        if message.message_type in self._event_callbacks:
            for callback in self._event_callbacks[message.message_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(message)
                    else:
                        callback(message)
                except Exception as e:
                    self.logger.error(f"Event callback error: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()


import asyncio
