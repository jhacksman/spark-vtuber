"""Tests for conversation context module."""

import pytest
from datetime import datetime

from spark_vtuber.llm.context import (
    ConversationContext,
    Message,
    MessageRole,
)


class TestMessage:
    """Tests for Message class."""

    def test_create_message(self):
        """Test message creation."""
        msg = Message(
            role=MessageRole.USER,
            content="Hello, world!",
        )
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello, world!"
        assert isinstance(msg.timestamp, datetime)

    def test_message_with_name(self):
        """Test message with name."""
        msg = Message(
            role=MessageRole.USER,
            content="Hello!",
            name="TestUser",
        )
        assert msg.name == "TestUser"

    def test_to_dict(self):
        """Test message to dict conversion."""
        msg = Message(
            role=MessageRole.USER,
            content="Hello!",
            name="TestUser",
        )
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "Hello!"
        assert d["name"] == "TestUser"

    def test_to_dict_without_name(self):
        """Test message to dict without name."""
        msg = Message(
            role=MessageRole.ASSISTANT,
            content="Hi there!",
        )
        d = msg.to_dict()
        assert d["role"] == "assistant"
        assert d["content"] == "Hi there!"
        assert "name" not in d


class TestConversationContext:
    """Tests for ConversationContext class."""

    def test_create_context(self):
        """Test context creation."""
        ctx = ConversationContext(
            system_prompt="You are a helpful assistant.",
            max_tokens=4096,
        )
        assert ctx.system_prompt == "You are a helpful assistant."
        assert ctx.max_tokens == 4096
        assert len(ctx.messages) == 0

    def test_add_message(self):
        """Test adding messages."""
        ctx = ConversationContext()
        msg = ctx.add_message(MessageRole.USER, "Hello!")
        assert len(ctx.messages) == 1
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello!"

    def test_add_message_string_role(self):
        """Test adding message with string role."""
        ctx = ConversationContext()
        msg = ctx.add_message("user", "Hello!")
        assert msg.role == MessageRole.USER

    def test_add_user_message(self):
        """Test add_user_message helper."""
        ctx = ConversationContext()
        msg = ctx.add_user_message("Hello!", name="TestUser")
        assert msg.role == MessageRole.USER
        assert msg.name == "TestUser"

    def test_add_assistant_message(self):
        """Test add_assistant_message helper."""
        ctx = ConversationContext()
        msg = ctx.add_assistant_message("Hi!", personality="spark")
        assert msg.role == MessageRole.ASSISTANT
        assert msg.personality == "spark"

    def test_get_messages(self):
        """Test getting messages."""
        ctx = ConversationContext(system_prompt="System prompt")
        ctx.add_user_message("Hello!")
        ctx.add_assistant_message("Hi!")

        messages = ctx.get_messages()
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"

    def test_get_messages_without_system(self):
        """Test getting messages without system prompt."""
        ctx = ConversationContext(system_prompt="System prompt")
        ctx.add_user_message("Hello!")

        messages = ctx.get_messages(include_system=False)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_token_count(self):
        """Test token count estimation."""
        ctx = ConversationContext(system_prompt="Short prompt")
        assert ctx.token_count > 0

        ctx.add_user_message("A longer message with more content")
        new_count = ctx.token_count
        assert new_count > 0

    def test_needs_summarization(self):
        """Test summarization threshold."""
        ctx = ConversationContext(summarization_threshold=10)
        assert not ctx.needs_summarization

        ctx.add_user_message("A" * 100)
        assert ctx.needs_summarization

    def test_set_summary(self):
        """Test setting summary."""
        ctx = ConversationContext()
        for i in range(20):
            ctx.add_user_message(f"Message {i}")

        ctx.set_summary("Summary of conversation")
        assert len(ctx.messages) <= 10

    def test_clear(self):
        """Test clearing context."""
        ctx = ConversationContext()
        ctx.add_user_message("Hello!")
        ctx.add_assistant_message("Hi!")

        ctx.clear()
        assert len(ctx.messages) == 0

    def test_get_recent_messages(self):
        """Test getting recent messages."""
        ctx = ConversationContext()
        for i in range(20):
            ctx.add_user_message(f"Message {i}")

        recent = ctx.get_recent_messages(5)
        assert len(recent) == 5
        assert recent[-1].content == "Message 19"

    def test_format_for_llama(self):
        """Test Llama format."""
        ctx = ConversationContext(system_prompt="System")
        ctx.add_user_message("Hello!")

        formatted = ctx.format_for_llama()
        assert "<|begin_of_text|>" in formatted
        assert "<|start_header_id|>system<|end_header_id|>" in formatted
        assert "<|start_header_id|>user<|end_header_id|>" in formatted
        assert "<|start_header_id|>assistant<|end_header_id|>" in formatted

    def test_format_for_chatml(self):
        """Test ChatML format."""
        ctx = ConversationContext(system_prompt="System")
        ctx.add_user_message("Hello!")

        formatted = ctx.format_for_chatml()
        assert "<|im_start|>system" in formatted
        assert "<|im_start|>user" in formatted
        assert "<|im_start|>assistant" in formatted
