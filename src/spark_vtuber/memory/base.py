"""
Base memory interface for Spark VTuber.

Provides abstract base class for memory/RAG implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from spark_vtuber.utils.logging import LoggerMixin


@dataclass
class MemoryEntry:
    """A single memory entry."""

    id: str
    content: str
    embedding: list[float] | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    personality: str | None = None
    category: str = "general"
    importance: float = 0.5
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "personality": self.personality,
            "category": self.category,
            "importance": self.importance,
            "metadata": self.metadata,
        }


@dataclass
class SearchResult:
    """Result from memory search."""

    entry: MemoryEntry
    score: float
    distance: float = 0.0


class BaseMemory(ABC, LoggerMixin):
    """
    Abstract base class for memory implementations.

    Provides interface for storing and retrieving memories with RAG.
    """

    def __init__(self, **kwargs):
        """Initialize the memory system."""
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if memory is initialized."""
        return self._initialized

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the memory system."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the memory system."""
        pass

    @abstractmethod
    async def add(
        self,
        content: str,
        personality: str | None = None,
        category: str = "general",
        importance: float = 0.5,
        **metadata,
    ) -> MemoryEntry:
        """
        Add a new memory.

        Args:
            content: Memory content
            personality: Associated personality
            category: Memory category
            importance: Importance score (0-1)
            **metadata: Additional metadata

        Returns:
            Created MemoryEntry
        """
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 5,
        personality: str | None = None,
        category: str | None = None,
        min_importance: float = 0.0,
    ) -> list[SearchResult]:
        """
        Search for relevant memories.

        Args:
            query: Search query
            top_k: Number of results to return
            personality: Filter by personality
            category: Filter by category
            min_importance: Minimum importance threshold

        Returns:
            List of SearchResults
        """
        pass

    @abstractmethod
    async def get(self, memory_id: str) -> MemoryEntry | None:
        """
        Get a specific memory by ID.

        Args:
            memory_id: Memory ID

        Returns:
            MemoryEntry or None if not found
        """
        pass

    @abstractmethod
    async def update(
        self,
        memory_id: str,
        content: str | None = None,
        importance: float | None = None,
        **metadata,
    ) -> MemoryEntry | None:
        """
        Update an existing memory.

        Args:
            memory_id: Memory ID
            content: New content (optional)
            importance: New importance (optional)
            **metadata: Additional metadata to update

        Returns:
            Updated MemoryEntry or None if not found
        """
        pass

    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """
        Delete a memory.

        Args:
            memory_id: Memory ID

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def count(self) -> int:
        """Get total number of memories."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all memories."""
        pass

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
