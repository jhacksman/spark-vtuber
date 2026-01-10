"""Pytest configuration and fixtures."""

import pytest
import asyncio
from pathlib import Path
import tempfile


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_audio():
    """Create sample audio data."""
    import numpy as np
    duration = 1.0
    sample_rate = 22050
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio = np.sin(2 * np.pi * 440 * t) * 0.5
    return audio, sample_rate


@pytest.fixture
def sample_chat_message():
    """Create a sample chat message."""
    from spark_vtuber.chat.base import ChatMessage, MessageType
    return ChatMessage(
        id="test-123",
        platform="twitch",
        username="testuser",
        display_name="TestUser",
        content="Hello, Spark!",
        message_type=MessageType.CHAT,
    )
