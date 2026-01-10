"""
Message queue and prioritization for Spark VTuber.

Handles message filtering, prioritization, and rate limiting.
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable

from spark_vtuber.chat.base import ChatMessage, MessageType
from spark_vtuber.utils.logging import LoggerMixin


class MessagePriority(IntEnum):
    """Message priority levels."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


@dataclass
class QueuedMessage:
    """A message in the queue with priority."""

    message: ChatMessage
    priority: MessagePriority = MessagePriority.NORMAL
    queued_at: float = field(default_factory=time.time)

    def __lt__(self, other: "QueuedMessage") -> bool:
        """Compare by priority (higher first), then by time (older first)."""
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.queued_at < other.queued_at


class MessageQueue(LoggerMixin):
    """
    Priority queue for chat messages.

    Handles:
    - Message prioritization (subs, donations, mentions)
    - Rate limiting
    - Spam filtering
    - Message deduplication
    """

    def __init__(
        self,
        max_size: int = 100,
        rate_limit_per_minute: int = 20,
        spam_threshold: int = 3,
        mention_keywords: list[str] | None = None,
    ):
        """
        Initialize message queue.

        Args:
            max_size: Maximum queue size
            rate_limit_per_minute: Max messages to process per minute
            spam_threshold: Number of similar messages to consider spam
            mention_keywords: Keywords that increase priority
        """
        self.max_size = max_size
        self.rate_limit_per_minute = rate_limit_per_minute
        self.spam_threshold = spam_threshold
        self.mention_keywords = mention_keywords or ["spark", "shadow", "@spark", "@shadow"]

        self._queue: list[QueuedMessage] = []
        self._processed_times: deque[float] = deque()
        self._recent_messages: deque[str] = deque(maxlen=50)
        self._user_message_counts: dict[str, int] = {}
        self._priority_rules: list[Callable[[ChatMessage], MessagePriority | None]] = []

        self._setup_default_rules()

    def _setup_default_rules(self) -> None:
        """Set up default priority rules."""
        def donation_rule(msg: ChatMessage) -> MessagePriority | None:
            if msg.message_type == MessageType.DONATION:
                return MessagePriority.URGENT
            return None

        def subscription_rule(msg: ChatMessage) -> MessagePriority | None:
            if msg.message_type == MessageType.SUBSCRIPTION:
                return MessagePriority.HIGH
            return None

        def mention_rule(msg: ChatMessage) -> MessagePriority | None:
            content_lower = msg.content.lower()
            for keyword in self.mention_keywords:
                if keyword.lower() in content_lower:
                    return MessagePriority.HIGH
            return None

        def moderator_rule(msg: ChatMessage) -> MessagePriority | None:
            if msg.is_moderator:
                return MessagePriority.HIGH
            return None

        self._priority_rules.extend([
            donation_rule,
            subscription_rule,
            mention_rule,
            moderator_rule,
        ])

    def add_priority_rule(
        self,
        rule: Callable[[ChatMessage], MessagePriority | None],
    ) -> None:
        """
        Add a custom priority rule.

        Args:
            rule: Function that returns priority or None
        """
        self._priority_rules.append(rule)

    def _calculate_priority(self, message: ChatMessage) -> MessagePriority:
        """Calculate priority for a message."""
        highest_priority = MessagePriority.NORMAL

        for rule in self._priority_rules:
            priority = rule(message)
            if priority is not None and priority > highest_priority:
                highest_priority = priority

        return highest_priority

    def _is_spam(self, message: ChatMessage) -> bool:
        """Check if message is spam."""
        content_lower = message.content.lower().strip()

        similar_count = sum(
            1 for msg in self._recent_messages
            if msg.lower().strip() == content_lower
        )

        if similar_count >= self.spam_threshold:
            return True

        user_count = self._user_message_counts.get(message.username, 0)
        if user_count > 10:
            return True

        return False

    def _is_rate_limited(self) -> bool:
        """Check if we're rate limited."""
        now = time.time()

        while self._processed_times and now - self._processed_times[0] > 60:
            self._processed_times.popleft()

        return len(self._processed_times) >= self.rate_limit_per_minute

    async def add(self, message: ChatMessage) -> bool:
        """
        Add a message to the queue.

        Args:
            message: Message to add

        Returns:
            True if message was added, False if filtered
        """
        if self._is_spam(message):
            self.logger.debug(f"Filtered spam from {message.username}")
            return False

        if len(self._queue) >= self.max_size:
            if self._queue:
                self._queue.pop()
            self.logger.warning("Queue full, dropped oldest message")

        priority = self._calculate_priority(message)
        queued = QueuedMessage(message=message, priority=priority)

        self._queue.append(queued)
        self._queue.sort()

        self._recent_messages.append(message.content)
        self._user_message_counts[message.username] = (
            self._user_message_counts.get(message.username, 0) + 1
        )

        return True

    async def get(self, timeout: float = 1.0) -> ChatMessage | None:
        """
        Get the next message from the queue.

        Args:
            timeout: Maximum time to wait

        Returns:
            Next message or None if timeout/rate limited
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self._is_rate_limited():
                await asyncio.sleep(0.1)
                continue

            if self._queue:
                queued = self._queue.pop(0)
                self._processed_times.append(time.time())
                return queued.message

            await asyncio.sleep(0.1)

        return None

    def peek(self) -> ChatMessage | None:
        """Peek at the next message without removing it."""
        if self._queue:
            return self._queue[0].message
        return None

    def size(self) -> int:
        """Get current queue size."""
        return len(self._queue)

    def clear(self) -> None:
        """Clear the queue."""
        self._queue.clear()
        self._recent_messages.clear()
        self._user_message_counts.clear()

    def reset_rate_limit(self) -> None:
        """Reset rate limit counter."""
        self._processed_times.clear()

    def get_stats(self) -> dict:
        """Get queue statistics."""
        return {
            "queue_size": len(self._queue),
            "processed_last_minute": len(self._processed_times),
            "rate_limit": self.rate_limit_per_minute,
            "unique_users": len(self._user_message_counts),
        }
