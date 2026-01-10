"""
ChromaDB memory implementation for Spark VTuber.

Provides vector-based memory storage with semantic search.
"""

import asyncio
import uuid
from datetime import datetime
from pathlib import Path

from spark_vtuber.memory.base import BaseMemory, MemoryEntry, SearchResult


class ChromaMemory(BaseMemory):
    """
    ChromaDB-based memory implementation.

    Supports:
    - Persistent vector storage
    - Semantic similarity search
    - Metadata filtering
    - Personality-tagged memories
    """

    def __init__(
        self,
        persist_dir: Path | str = "./data/chroma",
        collection_name: str = "spark_vtuber_memories",
        embedding_model: str = "all-MiniLM-L6-v2",
        **kwargs,
    ):
        """
        Initialize ChromaDB memory.

        Args:
            persist_dir: Directory for persistent storage
            collection_name: Name of the ChromaDB collection
            embedding_model: Sentence transformer model for embeddings
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self._client = None
        self._collection = None
        self._embedder = None

    async def initialize(self) -> None:
        """Initialize ChromaDB and embedding model."""
        if self._initialized:
            self.logger.warning("Memory already initialized")
            return

        self.logger.info(f"Initializing ChromaDB at {self.persist_dir}")

        self.persist_dir.mkdir(parents=True, exist_ok=True)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._init_sync)

        self._initialized = True
        self.logger.info("ChromaDB initialized")

    def _init_sync(self) -> None:
        """Synchronous initialization."""
        import chromadb
        from chromadb.config import Settings
        from sentence_transformers import SentenceTransformer

        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self._embedder = SentenceTransformer(self.embedding_model)

    async def close(self) -> None:
        """Close the memory system."""
        if not self._initialized:
            return

        self.logger.info("Closing ChromaDB")

        self._client = None
        self._collection = None
        self._embedder = None
        self._initialized = False

    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text."""
        return self._embedder.encode(text).tolist()

    async def add(
        self,
        content: str,
        personality: str | None = None,
        category: str = "general",
        importance: float = 0.5,
        **metadata,
    ) -> MemoryEntry:
        """Add a new memory."""
        if not self._initialized:
            raise RuntimeError("Memory not initialized")

        memory_id = str(uuid.uuid4())
        now = datetime.now()

        entry = MemoryEntry(
            id=memory_id,
            content=content,
            created_at=now,
            updated_at=now,
            personality=personality,
            category=category,
            importance=importance,
            metadata=metadata,
        )

        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: self._get_embedding(content),
        )

        entry.embedding = embedding

        # ChromaDB requires consistent types - store importance as float for numeric filtering
        doc_metadata = {
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "personality": personality or "",
            "category": category,
            "importance": float(importance),  # Ensure float type for $gte filtering
            **{k: str(v) for k, v in metadata.items()},
        }

        await loop.run_in_executor(
            None,
            lambda: self._collection.add(
                ids=[memory_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[doc_metadata],
            ),
        )

        self.logger.debug(f"Added memory: {memory_id}")
        return entry

    async def search(
        self,
        query: str,
        top_k: int = 5,
        personality: str | None = None,
        category: str | None = None,
        min_importance: float = 0.0,
    ) -> list[SearchResult]:
        """Search for relevant memories."""
        if not self._initialized:
            raise RuntimeError("Memory not initialized")

        loop = asyncio.get_event_loop()
        query_embedding = await loop.run_in_executor(
            None,
            lambda: self._get_embedding(query),
        )

        where_filter = {}
        if personality:
            where_filter["personality"] = personality
        if category:
            where_filter["category"] = category
        if min_importance > 0:
            where_filter["importance"] = {"$gte": min_importance}

        results = await loop.run_in_executor(
            None,
            lambda: self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter if where_filter else None,
                include=["documents", "metadatas", "distances"],
            ),
        )

        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, memory_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                content = results["documents"][0][i] if results["documents"] else ""
                distance = results["distances"][0][i] if results["distances"] else 0

                entry = MemoryEntry(
                    id=memory_id,
                    content=content,
                    created_at=datetime.fromisoformat(metadata.get("created_at", datetime.now().isoformat())),
                    updated_at=datetime.fromisoformat(metadata.get("updated_at", datetime.now().isoformat())),
                    personality=metadata.get("personality") or None,
                    category=metadata.get("category", "general"),
                    importance=float(metadata.get("importance", 0.5)),
                )

                score = 1 - distance

                search_results.append(SearchResult(
                    entry=entry,
                    score=score,
                    distance=distance,
                ))

        return search_results

    async def get(self, memory_id: str) -> MemoryEntry | None:
        """Get a specific memory by ID."""
        if not self._initialized:
            raise RuntimeError("Memory not initialized")

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self._collection.get(
                ids=[memory_id],
                include=["documents", "metadatas"],
            ),
        )

        if not results["ids"]:
            return None

        metadata = results["metadatas"][0] if results["metadatas"] else {}
        content = results["documents"][0] if results["documents"] else ""

        return MemoryEntry(
            id=memory_id,
            content=content,
            created_at=datetime.fromisoformat(metadata.get("created_at", datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(metadata.get("updated_at", datetime.now().isoformat())),
            personality=metadata.get("personality") or None,
            category=metadata.get("category", "general"),
            importance=float(metadata.get("importance", 0.5)),
        )

    async def update(
        self,
        memory_id: str,
        content: str | None = None,
        importance: float | None = None,
        **metadata,
    ) -> MemoryEntry | None:
        """Update an existing memory."""
        if not self._initialized:
            raise RuntimeError("Memory not initialized")

        existing = await self.get(memory_id)
        if not existing:
            return None

        now = datetime.now()
        new_content = content if content is not None else existing.content
        new_importance = importance if importance is not None else existing.importance

        loop = asyncio.get_event_loop()

        if content is not None:
            embedding = await loop.run_in_executor(
                None,
                lambda: self._get_embedding(new_content),
            )
        else:
            embedding = None

        doc_metadata = {
            "created_at": existing.created_at.isoformat(),
            "updated_at": now.isoformat(),
            "personality": existing.personality or "",
            "category": existing.category,
            "importance": float(new_importance),  # Ensure float type for $gte filtering
            **{k: str(v) for k, v in metadata.items()},
        }

        update_kwargs = {
            "ids": [memory_id],
            "documents": [new_content],
            "metadatas": [doc_metadata],
        }
        if embedding:
            update_kwargs["embeddings"] = [embedding]

        await loop.run_in_executor(
            None,
            lambda: self._collection.update(**update_kwargs),
        )

        return await self.get(memory_id)

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        if not self._initialized:
            raise RuntimeError("Memory not initialized")

        existing = await self.get(memory_id)
        if not existing:
            return False

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self._collection.delete(ids=[memory_id]),
        )

        return True

    async def count(self) -> int:
        """Get total number of memories."""
        if not self._initialized:
            raise RuntimeError("Memory not initialized")

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._collection.count(),
        )

    async def clear(self) -> None:
        """Clear all memories."""
        if not self._initialized:
            raise RuntimeError("Memory not initialized")

        loop = asyncio.get_event_loop()

        all_ids = await loop.run_in_executor(
            None,
            lambda: self._collection.get()["ids"],
        )

        if all_ids:
            await loop.run_in_executor(
                None,
                lambda: self._collection.delete(ids=all_ids),
            )

        self.logger.info("Cleared all memories")
