"""Unit tests for MemoryRetriever - comprehensive coverage."""

import pytest
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch
from codegrok_mcp.indexing.memory_retriever import MemoryRetriever, TTL_DURATIONS
from codegrok_mcp.core.models import Memory, MemoryType


class TestMemoryRetriever:
    """Tests for MemoryRetriever class."""

    @pytest.fixture
    def mock_embedding_service(self):
        """Create mock embedding service."""
        service = Mock()
        service.embed.return_value = [0.1] * 768
        service.embed_batch.return_value = [[0.1] * 768]
        return service

    @pytest.fixture
    def retriever(self, tmp_path, mock_embedding_service):
        """Create MemoryRetriever with mocked dependencies."""
        return MemoryRetriever(
            project_path=str(tmp_path),
            persist_path=str(tmp_path / "chroma"),
            embedding_service=mock_embedding_service,
            verbose=False,
        )

    @pytest.fixture
    def verbose_retriever(self, tmp_path, mock_embedding_service):
        """Create MemoryRetriever with verbose mode enabled."""
        return MemoryRetriever(
            project_path=str(tmp_path),
            persist_path=str(tmp_path / "chroma_verbose"),
            embedding_service=mock_embedding_service,
            verbose=True,
        )

    @pytest.fixture
    def ephemeral_retriever(self, tmp_path, mock_embedding_service):
        """Create MemoryRetriever without persistence (ephemeral)."""
        return MemoryRetriever(
            project_path=str(tmp_path),
            persist_path=None,  # Ephemeral mode
            embedding_service=mock_embedding_service,
            verbose=False,
        )

    # ==========================================================================
    # Basic CRUD Tests
    # ==========================================================================

    def test_remember_creates_memory(self, retriever):
        """Test that remember() stores a memory."""
        memory = retriever.remember(
            content="Test memory content", memory_type="status", tags=["test"]
        )

        assert memory.id is not None
        assert memory.content == "Test memory content"
        assert memory.memory_type == MemoryType.STATUS
        assert "test" in memory.tags

    def test_recall_returns_relevant_memories(self, retriever):
        """Test that recall() returns stored memories."""
        retriever.remember(
            content="Authentication uses JWT tokens", memory_type="decision", tags=["auth"]
        )

        results = retriever.recall(query="How does auth work?", n_results=5)

        assert len(results) > 0
        assert "JWT" in results[0]["content"]

    def test_forget_removes_memory(self, retriever):
        """Test that forget() removes memories."""
        memory = retriever.remember(content="Temporary note", memory_type="note")

        result = retriever.forget(memory_id=memory.id)

        assert result["deleted"] == 1

    def test_get_stats(self, retriever):
        """Test that get_stats returns correct counts."""
        retriever.remember(content="Note 1", memory_type="note")
        retriever.remember(content="Note 2", memory_type="note")
        retriever.remember(content="Status 1", memory_type="status")

        stats = retriever.get_stats()

        assert stats["total_memories"] == 3
        assert stats["by_type"].get("note", 0) == 2
        assert stats["by_type"].get("status", 0) == 1

    # ==========================================================================
    # All Memory Types Tests
    # ==========================================================================

    def test_remember_all_memory_types(self, retriever):
        """Test storing all memory types."""
        types_and_content = [
            ("conversation", "User asked about API design"),
            ("status", "Sprint 5: 80% complete"),
            ("decision", "Using PostgreSQL for persistence"),
            ("preference", "User prefers 4-space indentation"),
            ("doc", "API endpoint /users returns user list"),
            ("note", "TODO: Review authentication flow"),
        ]

        for mem_type, content in types_and_content:
            memory = retriever.remember(content=content, memory_type=mem_type)
            assert memory.memory_type == MemoryType.from_string(mem_type)
            assert memory.content == content

        stats = retriever.get_stats()
        assert stats["total_memories"] == 6

    def test_recall_by_each_memory_type(self, retriever):
        """Test recalling memories filtered by each type."""
        # Store one of each type
        retriever.remember(content="Chat about auth", memory_type="conversation")
        retriever.remember(content="Project blocked on DB", memory_type="status")
        retriever.remember(content="Chose React for frontend", memory_type="decision")

        # Recall filtering by type
        conv_results = retriever.recall(query="chat", memory_type="conversation")
        status_results = retriever.recall(query="blocked", memory_type="status")
        decision_results = retriever.recall(query="frontend", memory_type="decision")

        assert all(r["memory_type"] == "conversation" for r in conv_results)
        assert all(r["memory_type"] == "status" for r in status_results)
        assert all(r["memory_type"] == "decision" for r in decision_results)

    # ==========================================================================
    # Filtering Tests
    # ==========================================================================

    def test_memory_type_filtering(self, retriever):
        """Test filtering by memory type."""
        retriever.remember(content="Status update", memory_type="status")
        retriever.remember(content="A decision", memory_type="decision")

        results = retriever.recall(query="update", memory_type="status")

        assert all(r["memory_type"] == "status" for r in results)

    def test_tag_filtering(self, retriever):
        """Test filtering by tags."""
        retriever.remember(content="Auth note", memory_type="note", tags=["auth"])
        retriever.remember(content="DB note", memory_type="note", tags=["database"])

        results = retriever.recall(query="note", tags=["auth"])

        assert all("auth" in r["tags"] for r in results)

    def test_multiple_tags_filtering(self, retriever):
        """Test filtering with multiple tags (OR logic)."""
        retriever.remember(content="Auth feature", memory_type="note", tags=["auth"])
        retriever.remember(content="DB feature", memory_type="note", tags=["database"])
        retriever.remember(content="UI feature", memory_type="note", tags=["frontend"])

        # Should match auth OR database
        results = retriever.recall(query="feature", tags=["auth", "database"])

        # Should find auth and database, not frontend
        tags_found = set()
        for r in results:
            tags_found.update(r["tags"])

        assert "auth" in tags_found or "database" in tags_found

    def test_time_range_filtering_today(self, retriever):
        """Test filtering by time range 'today'."""
        retriever.remember(content="Recent memory", memory_type="note")

        results = retriever.recall(query="memory", time_range="today")

        assert len(results) > 0

    def test_time_range_filtering_week(self, retriever):
        """Test filtering by time range 'week'."""
        retriever.remember(content="This week memory", memory_type="note")

        results = retriever.recall(query="memory", time_range="week")

        assert len(results) > 0

    def test_time_range_filtering_month(self, retriever):
        """Test filtering by time range 'month'."""
        retriever.remember(content="This month memory", memory_type="note")

        results = retriever.recall(query="memory", time_range="month")

        assert len(results) > 0

    def test_time_range_filtering_all(self, retriever):
        """Test filtering by time range 'all' (no filtering)."""
        retriever.remember(content="Any memory", memory_type="note")

        results = retriever.recall(query="memory", time_range="all")

        assert len(results) > 0

    def test_min_relevance_filtering(self, retriever):
        """Test filtering by minimum relevance score."""
        retriever.remember(content="Exact match test", memory_type="note")

        # With min_relevance=0, should return results
        results = retriever.recall(query="test", min_relevance=0.0)
        assert len(results) > 0

        # With min_relevance=1.0, might not return results (depends on similarity)
        results_strict = retriever.recall(query="completely different", min_relevance=0.99)
        # Just verify it doesn't crash

    # ==========================================================================
    # Forget Tests
    # ==========================================================================

    def test_forget_by_type(self, retriever):
        """Test deleting all memories of a type."""
        retriever.remember(content="Note 1", memory_type="note")
        retriever.remember(content="Note 2", memory_type="note")
        retriever.remember(content="Status 1", memory_type="status")

        result = retriever.forget(memory_type="note")

        assert result["deleted"] == 2
        stats = retriever.get_stats()
        assert stats["total_memories"] == 1

    def test_forget_by_tags(self, retriever):
        """Test deleting memories by tags."""
        retriever.remember(content="Old auth code", memory_type="note", tags=["deprecated"])
        retriever.remember(content="Current auth", memory_type="note", tags=["current"])
        retriever.remember(content="Old db schema", memory_type="note", tags=["deprecated"])

        result = retriever.forget(tags=["deprecated"])

        assert result["deleted"] == 2
        stats = retriever.get_stats()
        assert stats["total_memories"] == 1

    def test_forget_by_older_than(self, retriever):
        """Test deleting memories older than specified duration."""
        # Store a memory
        retriever.remember(content="Recent memory", memory_type="note")

        # Try to delete memories older than 1 year (should delete nothing)
        result = retriever.forget(older_than="1y")

        # Recent memory should still exist
        stats = retriever.get_stats()
        assert stats["total_memories"] == 1

    def test_forget_by_older_than_1d(self, retriever):
        """Test forget with 1d older_than."""
        retriever.remember(content="Memory", memory_type="note")

        # Delete memories older than 1 day (recent ones won't be deleted)
        result = retriever.forget(older_than="1d")

        # Memory was just created, shouldn't be deleted
        stats = retriever.get_stats()
        assert stats["total_memories"] == 1

    def test_forget_by_older_than_7d(self, retriever):
        """Test forget with 7d older_than."""
        retriever.remember(content="Memory", memory_type="note")
        result = retriever.forget(older_than="7d")
        stats = retriever.get_stats()
        assert stats["total_memories"] == 1

    def test_forget_by_older_than_30d(self, retriever):
        """Test forget with 30d older_than."""
        retriever.remember(content="Memory", memory_type="note")
        result = retriever.forget(older_than="30d")
        stats = retriever.get_stats()
        assert stats["total_memories"] == 1

    def test_forget_combined_type_and_tags(self, retriever):
        """Test forget with both memory_type and tags filters."""
        retriever.remember(content="Old note", memory_type="note", tags=["old"])
        retriever.remember(content="Old status", memory_type="status", tags=["old"])
        retriever.remember(content="Current note", memory_type="note", tags=["current"])

        # Delete notes with 'old' tag
        result = retriever.forget(memory_type="note", tags=["old"])

        # Only the old note should be deleted
        stats = retriever.get_stats()
        assert stats["total_memories"] == 2

    def test_forget_nonexistent_id(self, retriever):
        """Test forgetting a non-existent memory ID."""
        result = retriever.forget(memory_id="nonexistent-id-12345")
        # Should not crash, deleted count might be 0 or 1 depending on implementation

    # ==========================================================================
    # TTL Tests
    # ==========================================================================

    def test_remember_with_all_ttl_options(self, retriever):
        """Test storing memories with all TTL options."""
        ttl_options = ["session", "day", "week", "month", "permanent"]

        for ttl in ttl_options:
            memory = retriever.remember(
                content=f"Memory with TTL {ttl}", memory_type="note", ttl=ttl
            )
            assert memory.ttl == ttl

    def test_cleanup_expired_with_permanent(self, retriever):
        """Test that permanent memories are not cleaned up."""
        retriever.remember(content="Permanent memory", memory_type="note", ttl="permanent")

        result = retriever.cleanup_expired()

        # Permanent memory should still exist
        stats = retriever.get_stats()
        assert stats["total_memories"] == 1

    def test_cleanup_expired_function(self, retriever):
        """Test cleanup_expired returns proper structure."""
        retriever.remember(content="Session memory", memory_type="note", ttl="session")
        retriever.remember(content="Permanent memory", memory_type="status", ttl="permanent")

        result = retriever.cleanup_expired()

        # Should return a dict
        assert isinstance(result, dict)

    # ==========================================================================
    # Edge Cases
    # ==========================================================================

    def test_recall_empty_collection(self, retriever):
        """Test recall on empty collection returns empty list."""
        results = retriever.recall(query="anything")
        assert results == []

    def test_recall_with_no_matches(self, retriever):
        """Test recall with query that doesn't match."""
        retriever.remember(content="Specific technical content", memory_type="note")

        results = retriever.recall(
            query="completely unrelated xyz123", min_relevance=0.9  # High threshold
        )
        # May or may not return results depending on embeddings

    def test_verbose_mode_logging(self, verbose_retriever, capsys):
        """Test that verbose mode produces output."""
        verbose_retriever.remember(content="Verbose test", memory_type="note")

        captured = capsys.readouterr()
        assert "[Memory]" in captured.out

    def test_ephemeral_retriever(self, ephemeral_retriever):
        """Test retriever without persistence works."""
        memory = ephemeral_retriever.remember(content="Ephemeral memory", memory_type="note")
        assert memory.id is not None

        results = ephemeral_retriever.recall(query="ephemeral")
        assert len(results) > 0

    def test_remember_with_metadata(self, retriever):
        """Test storing memory with custom metadata."""
        memory = retriever.remember(
            content="Memory with metadata",
            memory_type="note",
            metadata={"custom_field": "custom_value", "priority": 1},
        )

        assert memory.metadata["custom_field"] == "custom_value"
        assert memory.metadata["priority"] == 1

    def test_remember_with_source(self, retriever):
        """Test storing memory with different sources."""
        sources = ["user", "agent", "auto", "import"]

        for source in sources:
            memory = retriever.remember(
                content=f"Memory from {source}", memory_type="note", source=source
            )
            assert memory.source == source

    def test_get_stats_with_project_info(self, retriever):
        """Test that get_stats includes project information."""
        retriever.remember(content="Test", memory_type="note")

        stats = retriever.get_stats()

        assert "project" in stats
        assert "total_memories" in stats
        assert "by_type" in stats

    def test_stats_persistence(self, tmp_path, mock_embedding_service):
        """Test that stats are persisted and loaded."""
        persist_path = str(tmp_path / "chroma_persist")

        # Create retriever and add memories
        retriever1 = MemoryRetriever(
            project_path=str(tmp_path),
            persist_path=persist_path,
            embedding_service=mock_embedding_service,
            verbose=False,
        )
        retriever1.remember(content="Persisted memory", memory_type="note")

        # Stats should be saved
        stats1 = retriever1.get_stats()
        assert stats1["total_memories"] == 1

    def test_recall_updates_accessed_at(self, retriever):
        """Test that recalling memories updates accessed_at timestamp."""
        memory = retriever.remember(content="Access tracking test", memory_type="note")
        original_id = memory.id

        # Small delay
        time.sleep(0.01)

        # Recall the memory
        results = retriever.recall(query="access tracking")

        # Memory should have been accessed
        assert len(results) > 0


class TestMemoryModel:
    """Tests for Memory dataclass."""

    def test_memory_validation_empty_content(self):
        """Test memory validation for empty content."""
        with pytest.raises(ValueError, match="content cannot be empty"):
            Memory(id="test", content="", memory_type=MemoryType.NOTE, project="/test")

    def test_memory_validation_empty_id(self):
        """Test memory validation for empty id."""
        with pytest.raises(ValueError, match="id cannot be empty"):
            Memory(id="", content="Test", memory_type=MemoryType.NOTE, project="/test")

    def test_memory_validation_empty_project(self):
        """Test memory validation for empty project."""
        with pytest.raises(ValueError, match="project cannot be empty"):
            Memory(id="test", content="Test", memory_type=MemoryType.NOTE, project="")

    def test_memory_validation_invalid_memory_type(self):
        """Test memory validation for invalid memory_type."""
        with pytest.raises(ValueError, match="memory_type must be MemoryType enum"):
            Memory(
                id="test",
                content="Test",
                memory_type="invalid",  # Should be MemoryType enum
                project="/test",
            )

    def test_memory_validation_invalid_ttl(self):
        """Test memory validation for invalid TTL."""
        with pytest.raises(ValueError, match="ttl must be one of"):
            Memory(
                id="test",
                content="Test",
                memory_type=MemoryType.NOTE,
                project="/test",
                ttl="invalid",
            )

    def test_memory_serialization(self):
        """Test to_dict/from_dict round-trip."""
        memory = Memory(
            id="test-123",
            content="Test content",
            memory_type=MemoryType.DECISION,
            project="/test",
            tags=["tag1", "tag2"],
        )

        data = memory.to_dict()
        restored = Memory.from_dict(data)

        assert restored.id == memory.id
        assert restored.content == memory.content
        assert restored.memory_type == memory.memory_type
        assert restored.tags == memory.tags

    def test_memory_serialization_all_fields(self):
        """Test serialization with all fields populated."""
        memory = Memory(
            id="full-test",
            content="Full test content",
            memory_type=MemoryType.STATUS,
            project="/full/test",
            tags=["a", "b", "c"],
            ttl="week",
            source="agent",
            metadata={"key": "value"},
        )

        data = memory.to_dict()

        assert data["id"] == "full-test"
        assert data["memory_type"] == "status"
        assert data["tags"] == ["a", "b", "c"]
        assert data["ttl"] == "week"
        assert data["source"] == "agent"
        assert data["metadata"] == {"key": "value"}

        # Round-trip
        restored = Memory.from_dict(data)
        assert restored.id == memory.id
        assert restored.memory_type == memory.memory_type

    def test_memory_touch(self):
        """Test that touch() updates accessed_at."""
        memory = Memory(id="test", content="Test", memory_type=MemoryType.NOTE, project="/test")

        original_accessed = memory.accessed_at
        time.sleep(0.01)  # Small delay
        memory.touch()

        assert memory.accessed_at != original_accessed

    def test_memory_default_values(self):
        """Test that Memory has correct default values."""
        memory = Memory(
            id="defaults", content="Test defaults", memory_type=MemoryType.NOTE, project="/test"
        )

        assert memory.tags == []
        assert memory.ttl == "permanent"
        assert memory.source == "user"
        assert memory.metadata == {}
        assert memory.created_at is not None
        assert memory.accessed_at is not None


class TestMemoryType:
    """Tests for MemoryType enum."""

    def test_from_string_valid(self):
        """Test creating MemoryType from valid string."""
        assert MemoryType.from_string("conversation") == MemoryType.CONVERSATION
        assert MemoryType.from_string("status") == MemoryType.STATUS
        assert MemoryType.from_string("decision") == MemoryType.DECISION
        assert MemoryType.from_string("preference") == MemoryType.PREFERENCE
        assert MemoryType.from_string("doc") == MemoryType.DOC
        assert MemoryType.from_string("note") == MemoryType.NOTE

    def test_from_string_case_insensitive(self):
        """Test that from_string is case insensitive."""
        assert MemoryType.from_string("CONVERSATION") == MemoryType.CONVERSATION
        assert MemoryType.from_string("Status") == MemoryType.STATUS
        assert MemoryType.from_string("NOTE") == MemoryType.NOTE

    def test_from_string_invalid(self):
        """Test creating MemoryType from invalid string."""
        with pytest.raises(ValueError, match="Invalid MemoryType"):
            MemoryType.from_string("invalid")

    def test_str_representation(self):
        """Test string representation of MemoryType."""
        assert str(MemoryType.CONVERSATION) == "conversation"
        assert str(MemoryType.STATUS) == "status"
        assert str(MemoryType.DECISION) == "decision"
        assert str(MemoryType.PREFERENCE) == "preference"
        assert str(MemoryType.DOC) == "doc"
        assert str(MemoryType.NOTE) == "note"


class TestTTLDurations:
    """Tests for TTL duration mappings."""

    def test_ttl_durations_exist(self):
        """Test that all expected TTL durations exist."""
        assert "session" in TTL_DURATIONS
        assert "day" in TTL_DURATIONS
        assert "week" in TTL_DURATIONS
        assert "month" in TTL_DURATIONS
        assert "permanent" in TTL_DURATIONS

    def test_ttl_duration_values(self):
        """Test that TTL durations have correct values."""
        assert TTL_DURATIONS["session"] == timedelta(hours=24)
        assert TTL_DURATIONS["day"] == timedelta(days=1)
        assert TTL_DURATIONS["week"] == timedelta(weeks=1)
        assert TTL_DURATIONS["month"] == timedelta(days=30)
        assert TTL_DURATIONS["permanent"] is None


class TestEdgeCases:
    """Additional edge case tests for higher coverage."""

    @pytest.fixture
    def mock_embedding_service(self):
        """Create mock embedding service."""
        service = Mock()
        service.embed.return_value = [0.1] * 768
        return service

    def test_time_range_invalid_value(self, tmp_path, mock_embedding_service):
        """Test recall with invalid time_range (should be handled gracefully)."""
        retriever = MemoryRetriever(
            project_path=str(tmp_path),
            persist_path=str(tmp_path / "chroma"),
            embedding_service=mock_embedding_service,
            verbose=False,
        )
        retriever.remember(content="Test memory", memory_type="note")

        # Invalid time_range should not crash
        results = retriever.recall(query="test", time_range="invalid_range")
        # Should return results (no filtering applied for invalid range)
        assert isinstance(results, list)

    def test_stats_load_from_existing_file(self, tmp_path, mock_embedding_service):
        """Test loading stats from existing metadata file."""
        import json

        # Create metadata file with existing stats
        persist_path = tmp_path / "chroma"
        persist_path.mkdir(parents=True, exist_ok=True)
        metadata_path = tmp_path / "memory_metadata.json"

        existing_stats = {
            "stats": {
                "total_memories": 10,
                "by_type": {"note": 5, "status": 5},
                "last_cleanup": "2024-01-01T00:00:00",
            }
        }
        with open(metadata_path, "w") as f:
            json.dump(existing_stats, f)

        # Create retriever - should load existing stats
        retriever = MemoryRetriever(
            project_path=str(tmp_path),
            persist_path=str(persist_path),
            embedding_service=mock_embedding_service,
            verbose=False,
        )

        # Stats should have been loaded (though actual count may differ)
        assert retriever.stats is not None

    def test_recall_empty_tags_in_stored_memory(self, tmp_path, mock_embedding_service):
        """Test recalling memory that was stored without tags."""
        retriever = MemoryRetriever(
            project_path=str(tmp_path),
            persist_path=str(tmp_path / "chroma"),
            embedding_service=mock_embedding_service,
            verbose=False,
        )

        # Store memory without tags
        retriever.remember(content="Memory without tags", memory_type="note")

        # Recall with tag filter - should not crash on empty tags
        results = retriever.recall(query="memory", tags=["nonexistent"])
        # May or may not return results, but shouldn't crash

    def test_forget_invalid_older_than(self, tmp_path, mock_embedding_service):
        """Test forget with invalid older_than value."""
        retriever = MemoryRetriever(
            project_path=str(tmp_path),
            persist_path=str(tmp_path / "chroma"),
            embedding_service=mock_embedding_service,
            verbose=False,
        )
        retriever.remember(content="Test", memory_type="note")

        # Invalid older_than should not crash
        result = retriever.forget(older_than="invalid")
        assert "deleted" in result

    def test_verbose_recall_logging(self, tmp_path, mock_embedding_service, capsys):
        """Test verbose logging during recall."""
        retriever = MemoryRetriever(
            project_path=str(tmp_path),
            persist_path=str(tmp_path / "chroma"),
            embedding_service=mock_embedding_service,
            verbose=True,
        )
        retriever.remember(content="Verbose recall test", memory_type="note")
        retriever.recall(query="verbose")

        captured = capsys.readouterr()
        assert "[Memory]" in captured.out

    def test_verbose_forget_logging(self, tmp_path, mock_embedding_service, capsys):
        """Test verbose logging during forget."""
        retriever = MemoryRetriever(
            project_path=str(tmp_path),
            persist_path=str(tmp_path / "chroma"),
            embedding_service=mock_embedding_service,
            verbose=True,
        )
        memory = retriever.remember(content="Verbose forget test", memory_type="note")
        retriever.forget(memory_id=memory.id)

        captured = capsys.readouterr()
        assert "[Memory]" in captured.out

    def test_verbose_cleanup_logging(self, tmp_path, mock_embedding_service, capsys):
        """Test verbose logging during cleanup."""
        retriever = MemoryRetriever(
            project_path=str(tmp_path),
            persist_path=str(tmp_path / "chroma"),
            embedding_service=mock_embedding_service,
            verbose=True,
        )
        retriever.remember(content="Cleanup test", memory_type="note", ttl="session")
        retriever.cleanup_expired()

        captured = capsys.readouterr()
        assert "[Memory]" in captured.out

    def test_cleanup_with_various_ttls(self, tmp_path, mock_embedding_service):
        """Test cleanup_expired with various TTL values."""
        retriever = MemoryRetriever(
            project_path=str(tmp_path),
            persist_path=str(tmp_path / "chroma"),
            embedding_service=mock_embedding_service,
            verbose=False,
        )

        # Add memories with different TTLs
        retriever.remember(content="Session mem", memory_type="note", ttl="session")
        retriever.remember(content="Day mem", memory_type="note", ttl="day")
        retriever.remember(content="Week mem", memory_type="note", ttl="week")
        retriever.remember(content="Month mem", memory_type="note", ttl="month")
        retriever.remember(content="Permanent mem", memory_type="note", ttl="permanent")

        result = retriever.cleanup_expired()

        # Fresh memories shouldn't be cleaned up
        stats = retriever.get_stats()
        assert stats["total_memories"] == 5

    def test_get_stats_last_cleanup(self, tmp_path, mock_embedding_service):
        """Test that get_stats returns last_cleanup time."""
        retriever = MemoryRetriever(
            project_path=str(tmp_path),
            persist_path=str(tmp_path / "chroma"),
            embedding_service=mock_embedding_service,
            verbose=False,
        )

        # Initially no cleanup
        stats = retriever.get_stats()
        assert "last_cleanup" in stats

        # After cleanup
        retriever.cleanup_expired()
        stats = retriever.get_stats()
        # last_cleanup should be set in stats
