"""Tests for chat module."""

import pytest
from datetime import datetime

from spark_vtuber.chat.base import ChatMessage, MessageType
from spark_vtuber.chat.queue import MessageQueue, MessagePriority, QueuedMessage


class TestChatMessage:
    """Tests for ChatMessage class."""

    def test_create_message(self):
        """Test message creation."""
        msg = ChatMessage(
            id="123",
            platform="twitch",
            username="testuser",
            display_name="TestUser",
            content="Hello, world!",
        )
        assert msg.id == "123"
        assert msg.platform == "twitch"
        assert msg.username == "testuser"
        assert msg.display_name == "TestUser"
        assert msg.content == "Hello, world!"
        assert msg.message_type == MessageType.CHAT

    def test_is_command(self):
        """Test command detection."""
        msg = ChatMessage(
            id="1",
            platform="twitch",
            username="user",
            display_name="User",
            content="!help",
        )
        assert msg.is_command is True

        msg2 = ChatMessage(
            id="2",
            platform="twitch",
            username="user",
            display_name="User",
            content="hello",
        )
        assert msg2.is_command is False

    def test_command_name(self):
        """Test command name extraction."""
        msg = ChatMessage(
            id="1",
            platform="twitch",
            username="user",
            display_name="User",
            content="!help arg1 arg2",
        )
        assert msg.command_name == "help"

    def test_command_args(self):
        """Test command args extraction."""
        msg = ChatMessage(
            id="1",
            platform="twitch",
            username="user",
            display_name="User",
            content="!play song1 song2",
        )
        assert msg.command_args == ["song1", "song2"]

    def test_badges(self):
        """Test badge handling."""
        msg = ChatMessage(
            id="1",
            platform="twitch",
            username="user",
            display_name="User",
            content="hello",
            is_moderator=True,
            is_subscriber=True,
            badges=["moderator", "subscriber"],
        )
        assert msg.is_moderator is True
        assert msg.is_subscriber is True
        assert "moderator" in msg.badges


class TestQueuedMessage:
    """Tests for QueuedMessage class."""

    def test_priority_comparison(self):
        """Test priority comparison."""
        msg1 = QueuedMessage(
            message=ChatMessage(
                id="1", platform="twitch", username="u", display_name="U", content="hi"
            ),
            priority=MessagePriority.HIGH,
        )
        msg2 = QueuedMessage(
            message=ChatMessage(
                id="2", platform="twitch", username="u", display_name="U", content="hi"
            ),
            priority=MessagePriority.NORMAL,
        )
        assert msg1 < msg2


class TestMessageQueue:
    """Tests for MessageQueue class."""

    @pytest.fixture
    def queue(self):
        """Create a message queue."""
        return MessageQueue(
            max_size=10,
            rate_limit_per_minute=100,
            mention_keywords=["spark", "shadow"],
        )

    @pytest.fixture
    def sample_message(self):
        """Create a sample message."""
        return ChatMessage(
            id="1",
            platform="twitch",
            username="testuser",
            display_name="TestUser",
            content="Hello, world!",
        )

    @pytest.mark.asyncio
    async def test_add_message(self, queue, sample_message):
        """Test adding message to queue."""
        result = await queue.add(sample_message)
        assert result is True
        assert queue.size() == 1

    @pytest.mark.asyncio
    async def test_get_message(self, queue, sample_message):
        """Test getting message from queue."""
        await queue.add(sample_message)
        msg = await queue.get(timeout=0.1)
        assert msg is not None
        assert msg.id == "1"

    @pytest.mark.asyncio
    async def test_priority_ordering(self, queue):
        """Test priority ordering."""
        normal_msg = ChatMessage(
            id="1",
            platform="twitch",
            username="user",
            display_name="User",
            content="normal message",
        )
        mention_msg = ChatMessage(
            id="2",
            platform="twitch",
            username="user",
            display_name="User",
            content="hey @spark!",
        )

        await queue.add(normal_msg)
        await queue.add(mention_msg)

        first = await queue.get(timeout=0.1)
        assert first.id == "2"

    @pytest.mark.asyncio
    async def test_spam_filtering(self, queue):
        """Test spam filtering."""
        for i in range(5):
            msg = ChatMessage(
                id=str(i),
                platform="twitch",
                username="spammer",
                display_name="Spammer",
                content="spam message",
            )
            await queue.add(msg)

        assert queue.size() < 5

    def test_peek(self, queue, sample_message):
        """Test peeking at queue."""
        import asyncio
        asyncio.run(queue.add(sample_message))

        msg = queue.peek()
        assert msg is not None
        assert queue.size() == 1

    def test_clear(self, queue, sample_message):
        """Test clearing queue."""
        import asyncio
        asyncio.run(queue.add(sample_message))

        queue.clear()
        assert queue.size() == 0

    def test_get_stats(self, queue):
        """Test getting stats."""
        stats = queue.get_stats()
        assert "queue_size" in stats
        assert "processed_last_minute" in stats
        assert "rate_limit" in stats
