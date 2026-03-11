"""
Memory retriever for semantic memory search.

This module provides semantic search over memories (conversations, status,
decisions, etc.) using the same embedding infrastructure as code search.

Design Decisions:
- Separate ChromaDB collection from code (cleaner separation, different metadata)
- Reuses EmbeddingService singleton (no duplicate model loading)
- Supports TTL-based automatic cleanup
- Thread-safe for concurrent access
"""

import json
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb

from codegrok_mcp.indexing.embedding_service import get_embedding_service, EmbeddingService
from codegrok_mcp.core.models import Memory, MemoryType

# TTL duration mappings
TTL_DURATIONS = {
    "session": timedelta(hours=24),  # Cleared on session end or after 24h
    "day": timedelta(days=1),
    "week": timedelta(weeks=1),
    "month": timedelta(days=30),
    "permanent": None,  # Never expires
}


class MemoryRetriever:
    """
    Memory retriever for semantic search over conversations, status, decisions, etc.

    This class handles:
    - Storing memories with embeddings
    - Semantic search with filtering
    - TTL-based cleanup
    - Metadata persistence

    Storage: Uses ChromaDB collection named "memories" (separate from "codebase")
    """

    COLLECTION_NAME = "memories"
    METADATA_FILE = "memory_metadata.json"

    def __init__(
        self,
        project_path: str,
        embedding_model: str = "coderankembed",
        verbose: bool = False,
        persist_path: Optional[str] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        """
        Initialize the memory retriever.

        Args:
            project_path: Path to project root (used as project identifier)
            embedding_model: Embedding model name
            verbose: Print progress messages
            persist_path: Path for ChromaDB persistent storage
            embedding_service: Optional embedding service for dependency injection
        """
        self.project_path = Path(project_path).resolve()
        self.project_id = str(self.project_path)
        self.embedding_model = embedding_model
        self.verbose = verbose
        self.persist_path = persist_path

        # Reuse embedding service singleton (same model as code embeddings)
        self.embedding_service = embedding_service or get_embedding_service(
            embedding_model, show_progress=False
        )

        # Initialize ChromaDB (reuse existing client if possible)
        if persist_path:
            Path(persist_path).mkdir(parents=True, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=persist_path)
        else:
            self.chroma_client = chromadb.Client()

        # Get or create memories collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "Memory storage for conversations, status, decisions"},
        )

        # Statistics
        self.stats = {"total_memories": 0, "by_type": {}, "last_cleanup": None}

        self._load_stats()

    def _log(self, message: str):
        """Print message if verbose mode enabled."""
        if self.verbose:
            print(f"[Memory] {message}")

    def _load_stats(self):
        """Load statistics from metadata file."""
        if self.persist_path:
            metadata_path = Path(self.persist_path).parent / self.METADATA_FILE
            if metadata_path.exists():
                try:
                    with open(metadata_path, "r") as f:
                        data = json.load(f)
                        self.stats = data.get("stats", self.stats)
                except Exception:
                    pass

    def _save_stats(self):
        """Save statistics to metadata file."""
        if self.persist_path:
            metadata_path = Path(self.persist_path).parent / self.METADATA_FILE
            try:
                with open(metadata_path, "w") as f:
                    json.dump({"stats": self.stats}, f, indent=2)
            except Exception:
                pass

    def remember(
        self,
        content: str,
        memory_type: str,
        tags: List[str] = None,
        ttl: str = "permanent",
        source: str = "user",
        metadata: Dict[str, Any] = None,
    ) -> Memory:
        """
        Store a new memory with automatic embedding.

        Args:
            content: The memory content to store
            memory_type: Type of memory (conversation, status, decision, etc.)
            tags: Optional tags for filtering
            ttl: Time-to-live ("session", "day", "week", "month", "permanent")
            source: Origin of memory ("user", "agent", "auto", "import")
            metadata: Additional metadata

        Returns:
            The created Memory object
        """
        # Create memory object
        memory = Memory(
            id=str(uuid.uuid4()),
            content=content,
            memory_type=MemoryType.from_string(memory_type),
            project=self.project_id,
            tags=tags or [],
            ttl=ttl,
            source=source,
            metadata=metadata or {},
        )

        # Generate embedding
        embedding = self.embedding_service.embed(content)

        # Build ChromaDB metadata (flatten for ChromaDB compatibility)
        chroma_metadata = {
            "memory_type": memory.memory_type.value,
            "project": memory.project,
            "tags": ",".join(memory.tags),  # ChromaDB doesn't support list values
            "created_at": memory.created_at,
            "accessed_at": memory.accessed_at,
            "ttl": memory.ttl,
            "source": memory.source,
        }

        # Store in ChromaDB
        self.collection.add(
            ids=[memory.id],
            embeddings=[embedding],
            documents=[memory.content],
            metadatas=[chroma_metadata],
        )

        # Update stats
        self.stats["total_memories"] = self.collection.count()
        type_key = memory.memory_type.value
        self.stats["by_type"][type_key] = self.stats["by_type"].get(type_key, 0) + 1
        self._save_stats()

        self._log(f"Stored memory: {memory.id[:8]}... ({memory_type})")

        return memory

    def recall(
        self,
        query: str,
        memory_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        n_results: int = 5,
        time_range: Optional[str] = None,
        min_relevance: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Semantically search memories.

        Args:
            query: Natural language query
            memory_type: Optional filter by memory type
            tags: Optional filter by tags (matches any)
            n_results: Maximum number of results
            time_range: Optional time filter ("today", "week", "month", "all")
            min_relevance: Minimum relevance score (0.0 to 1.0)

        Returns:
            List of memory results with relevance scores
        """
        if self.collection.count() == 0:
            return []

        # Generate query embedding
        query_embedding = self.embedding_service.embed(query, is_query=True)

        # Parse time_range for post-filtering (ChromaDB date filtering is limited)
        time_cutoff = None
        if time_range and time_range != "all":
            now = datetime.now(timezone.utc)
            if time_range == "today":
                time_cutoff = now - timedelta(days=1)
            elif time_range == "week":
                time_cutoff = now - timedelta(weeks=1)
            elif time_range == "month":
                time_cutoff = now - timedelta(days=30)

        # Query ChromaDB with proper compound filter
        # Build $and filter for multiple conditions
        where_clauses = [{"project": self.project_id}]
        if memory_type:
            where_clauses.append({"memory_type": memory_type})

        chroma_where = {"$and": where_clauses} if len(where_clauses) > 1 else where_clauses[0]

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results * 2, 50),  # Over-fetch for tag/time filtering
                where=chroma_where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            self._log(f"Query error: {e}")
            return []

        if not results["ids"] or not results["ids"][0]:
            return []

        # Process results
        memories = []
        for i, (id_, doc, metadata, distance) in enumerate(
            zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ):
            # Convert distance to relevance (ChromaDB uses L2 distance)
            # Lower distance = higher relevance
            relevance = 1.0 / (1.0 + distance)

            if relevance < min_relevance:
                continue

            # Apply time_range filter (post-filter - ChromaDB date ops are limited)
            if time_cutoff:
                created_str = metadata.get("created_at", "")
                if created_str:
                    try:
                        created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                        if created < time_cutoff:
                            continue
                    except ValueError:
                        pass  # Skip malformed dates

            # Apply tags filter (post-filter - ChromaDB doesn't support list contains)
            if tags:
                stored_tags = set(metadata.get("tags", "").split(","))
                if not stored_tags.intersection(set(tags)):
                    continue

            memories.append(
                {
                    "id": id_,
                    "content": doc,
                    "memory_type": metadata.get("memory_type"),
                    "tags": metadata.get("tags", "").split(",") if metadata.get("tags") else [],
                    "created_at": metadata.get("created_at"),
                    "relevance": round(relevance, 3),
                    "source": metadata.get("source", "unknown"),
                }
            )

            if len(memories) >= n_results:
                break

        # Update accessed_at for returned memories (LRU tracking)
        if memories:
            now = datetime.now(timezone.utc).isoformat()
            for mem in memories:
                try:
                    self.collection.update(ids=[mem["id"]], metadatas=[{"accessed_at": now}])
                except Exception:
                    pass  # Non-critical

        self._log(f"Recalled {len(memories)} memories for query: {query[:50]}...")

        return memories

    def forget(
        self,
        memory_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        older_than: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        Remove memories matching criteria.

        Args:
            memory_id: Specific memory ID to delete
            memory_type: Delete all of this type
            tags: Delete memories with any of these tags
            older_than: Delete memories older than ("1d", "7d", "30d", "1y")

        Returns:
            Dict with count of deleted memories
        """
        deleted = 0

        if memory_id:
            # Delete specific memory - verify it exists first
            try:
                existing = self.collection.get(ids=[memory_id])
                if existing["ids"]:
                    self.collection.delete(ids=[memory_id])
                    deleted = 1
                else:
                    deleted = 0
            except Exception:
                deleted = 0
        else:
            # Build where filter for batch delete
            where_filter = {"project": self.project_id}

            if memory_type:
                where_filter["memory_type"] = memory_type

            # For older_than and tags, we need to fetch and filter
            if older_than or tags:
                # Get all memories for this project
                all_memories = self.collection.get(
                    where={"project": self.project_id}, include=["metadatas"]
                )

                ids_to_delete = []

                for id_, metadata in zip(all_memories["ids"], all_memories["metadatas"]):
                    should_delete = False

                    # Check older_than
                    if older_than:
                        duration_map = {
                            "1d": timedelta(days=1),
                            "7d": timedelta(days=7),
                            "30d": timedelta(days=30),
                            "1y": timedelta(days=365),
                        }
                        if older_than in duration_map:
                            cutoff = datetime.now(timezone.utc) - duration_map[older_than]
                            created = datetime.fromisoformat(metadata.get("created_at", ""))
                            if created < cutoff:
                                should_delete = True

                    # Check tags
                    if tags:
                        stored_tags = set(metadata.get("tags", "").split(","))
                        if stored_tags.intersection(set(tags)):
                            should_delete = True

                    # Check memory_type (if also specified)
                    if memory_type and metadata.get("memory_type") != memory_type:
                        should_delete = False

                    if should_delete:
                        ids_to_delete.append(id_)

                if ids_to_delete:
                    self.collection.delete(ids=ids_to_delete)
                    deleted = len(ids_to_delete)

            elif memory_type:
                # Simple type-based delete
                try:
                    # Get IDs first, then delete (ChromaDB requires $and for multiple conditions)
                    to_delete = self.collection.get(
                        where={"$and": [{"memory_type": memory_type}, {"project": self.project_id}]}
                    )
                    if to_delete["ids"]:
                        self.collection.delete(ids=to_delete["ids"])
                        deleted = len(to_delete["ids"])
                except Exception as e:
                    self._log(f"Delete error: {e}")

        # Update stats
        self.stats["total_memories"] = self.collection.count()
        self._save_stats()

        self._log(f"Forgot {deleted} memories")

        return {"deleted": deleted}

    def cleanup_expired(self) -> Dict[str, int]:
        """
        Remove memories that have exceeded their TTL.

        Called automatically or manually to clean up old memories.

        Returns:
            Dict with count of cleaned up memories by type
        """
        cleaned = {}
        now = datetime.now(timezone.utc)

        # Get all memories for this project
        all_memories = self.collection.get(
            where={"project": self.project_id}, include=["metadatas"]
        )

        ids_to_delete = []

        for id_, metadata in zip(all_memories["ids"], all_memories["metadatas"]):
            ttl = metadata.get("ttl", "permanent")

            if ttl == "permanent":
                continue

            duration = TTL_DURATIONS.get(ttl)
            if not duration:
                continue

            created = datetime.fromisoformat(metadata.get("created_at", now.isoformat()))
            if now - created > duration:
                ids_to_delete.append(id_)
                mem_type = metadata.get("memory_type", "unknown")
                cleaned[mem_type] = cleaned.get(mem_type, 0) + 1

        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)

        # Update stats
        self.stats["total_memories"] = self.collection.count()
        self.stats["last_cleanup"] = now.isoformat()
        self._save_stats()

        total_cleaned = sum(cleaned.values())
        self._log(f"Cleaned up {total_cleaned} expired memories")

        return cleaned

    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics.

        Returns:
            Dict with memory counts by type, total, storage info
        """
        # Refresh count from collection
        total = self.collection.count()

        # Get breakdown by type
        by_type = {}
        for mem_type in MemoryType:
            try:
                # ChromaDB requires $and for multiple conditions
                count = len(
                    self.collection.get(
                        where={
                            "$and": [{"memory_type": mem_type.value}, {"project": self.project_id}]
                        }
                    )["ids"]
                )
                if count > 0:
                    by_type[mem_type.value] = count
            except Exception:
                pass

        return {
            "total_memories": total,
            "by_type": by_type,
            "project": self.project_id,
            "last_cleanup": self.stats.get("last_cleanup"),
        }
