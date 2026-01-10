"""
Conversation context management for Spark VTuber.

Handles message history, context window management, and prompt formatting.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal

from spark_vtuber.utils.logging import LoggerMixin


class MessageRole(str, Enum):
    """Role of a message in conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


@dataclass
class Message:
    """A single message in the conversation."""

    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    name: str | None = None
    personality: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for API calls."""
        result = {"role": self.role.value, "content": self.content}
        if self.name:
            result["name"] = self.name
        return result


class ConversationContext(LoggerMixin):
    """
    Manages conversation context and history.

    Handles:
    - Message history storage
    - Context window management
    - Prompt formatting for different models
    - Summarization triggers
    """

    def __init__(
        self,
        system_prompt: str = "",
        max_tokens: int = 8192,
        summarization_threshold: int = 6000,
    ):
        """
        Initialize conversation context.

        Args:
            system_prompt: System prompt for the conversation
            max_tokens: Maximum context window size
            summarization_threshold: Token count to trigger summarization
        """
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.summarization_threshold = summarization_threshold
        self.messages: list[Message] = []
        self._summary: str | None = None
        self._token_count: int = 0

    def add_message(
        self,
        role: MessageRole | Literal["system", "user", "assistant", "function"],
        content: str,
        name: str | None = None,
        personality: str | None = None,
        **metadata,
    ) -> Message:
        """
        Add a message to the conversation.

        Args:
            role: Message role
            content: Message content
            name: Optional speaker name
            personality: Optional personality identifier
            **metadata: Additional metadata

        Returns:
            The created Message
        """
        if isinstance(role, str):
            role = MessageRole(role)

        message = Message(
            role=role,
            content=content,
            name=name,
            personality=personality,
            metadata=metadata,
        )
        self.messages.append(message)
        self._update_token_count()

        self.logger.debug(f"Added {role.value} message: {content[:50]}...")
        return message

    def add_user_message(self, content: str, name: str | None = None) -> Message:
        """Add a user message."""
        return self.add_message(MessageRole.USER, content, name=name)

    def add_assistant_message(
        self, content: str, personality: str | None = None
    ) -> Message:
        """Add an assistant message."""
        return self.add_message(
            MessageRole.ASSISTANT, content, personality=personality
        )

    def get_messages(self, include_system: bool = True) -> list[dict]:
        """
        Get messages formatted for API calls.

        Args:
            include_system: Whether to include system prompt

        Returns:
            List of message dictionaries
        """
        result = []

        if include_system and self.system_prompt:
            result.append({"role": "system", "content": self._get_full_system_prompt()})

        for msg in self.messages:
            result.append(msg.to_dict())

        return result

    def _get_full_system_prompt(self) -> str:
        """Get system prompt with any summary prepended."""
        if self._summary:
            return f"{self.system_prompt}\n\n[Previous conversation summary: {self._summary}]"
        return self.system_prompt

    def _update_token_count(self) -> None:
        """Update estimated token count."""
        total_chars = len(self.system_prompt)
        for msg in self.messages:
            total_chars += len(msg.content)
        self._token_count = total_chars // 4

    @property
    def token_count(self) -> int:
        """Get estimated token count."""
        return self._token_count

    @property
    def needs_summarization(self) -> bool:
        """Check if conversation needs summarization."""
        return self._token_count > self.summarization_threshold

    def set_summary(self, summary: str) -> None:
        """
        Set conversation summary and clear old messages.

        Args:
            summary: Summary of previous conversation
        """
        self._summary = summary
        recent_count = min(10, len(self.messages))
        self.messages = self.messages[-recent_count:]
        self._update_token_count()
        self.logger.info(f"Set summary, kept {recent_count} recent messages")

    def clear(self) -> None:
        """Clear all messages and summary."""
        self.messages.clear()
        self._summary = None
        self._token_count = 0

    def get_recent_messages(self, count: int = 10) -> list[Message]:
        """Get the most recent messages."""
        return self.messages[-count:]

    def format_for_llama(self) -> str:
        """
        Format conversation for Llama-style models.

        Returns:
            Formatted prompt string
        """
        parts = []

        if self.system_prompt:
            parts.append(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{self._get_full_system_prompt()}<|eot_id|>")

        for msg in self.messages:
            role = msg.role.value
            parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{msg.content}<|eot_id|>")

        parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return "".join(parts)

    def format_for_chatml(self) -> str:
        """
        Format conversation for ChatML-style models.

        Returns:
            Formatted prompt string
        """
        parts = []

        if self.system_prompt:
            parts.append(f"<|im_start|>system\n{self._get_full_system_prompt()}<|im_end|>")

        for msg in self.messages:
            role = msg.role.value
            parts.append(f"<|im_start|>{role}\n{msg.content}<|im_end|>")

        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)
