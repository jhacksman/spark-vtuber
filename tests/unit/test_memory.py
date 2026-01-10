"""Tests for memory module."""

import pytest
from datetime import datetime

from spark_vtuber.memory.base import MemoryEntry, SearchResult


class TestMemoryEntry:
    """Tests for MemoryEntry class."""

    def test_create_entry(self):
        """Test entry creation."""
        entry = MemoryEntry(
            id="123",
            content="Test memory content",
        )
        assert entry.id == "123"
        assert entry.content == "Test memory content"
        assert entry.category == "general"
        assert entry.importance == 0.5
        assert isinstance(entry.created_at, datetime)

    def test_entry_with_metadata(self):
        """Test entry with metadata."""
        entry = MemoryEntry(
            id="123",
            content="Test content",
            personality="spark",
            category="chat",
            importance=0.8,
            metadata={"source": "twitch"},
        )
        assert entry.personality == "spark"
        assert entry.category == "chat"
        assert entry.importance == 0.8
        assert entry.metadata["source"] == "twitch"

    def test_to_dict(self):
        """Test entry to dict conversion."""
        entry = MemoryEntry(
            id="123",
            content="Test content",
            personality="spark",
        )
        d = entry.to_dict()
        assert d["id"] == "123"
        assert d["content"] == "Test content"
        assert d["personality"] == "spark"
        assert "created_at" in d


class TestSearchResult:
    """Tests for SearchResult class."""

    def test_create_result(self):
        """Test result creation."""
        entry = MemoryEntry(id="123", content="Test")
        result = SearchResult(entry=entry, score=0.95, distance=0.05)
        assert result.entry.id == "123"
        assert result.score == 0.95
        assert result.distance == 0.05
