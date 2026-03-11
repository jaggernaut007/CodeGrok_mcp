"""
Integration tests for SourceRetriever - the core indexing/search engine.
These test the actual functionality without MCP overhead.
"""

import json
import pytest
import tempfile
from pathlib import Path
from codegrok_mcp.indexing.source_retriever import (
    SourceRetriever,
    _save_checkpoint,
    _load_checkpoint,
)


from unittest.mock import patch


class TestSourceRetrieverIndexing:
    """Test indexing functionality."""

    def test_index_python_project(self, temp_project):
        with tempfile.TemporaryDirectory() as persist_dir:
            retriever = SourceRetriever(codebase_path=str(temp_project), persist_path=persist_dir)

            retriever.index_codebase()
            stats = retriever.get_stats()

            assert stats["total_files"] >= 2
            assert stats["total_symbols"] > 0

    def test_index_respects_file_extensions(self, multi_lang_project):
        with tempfile.TemporaryDirectory() as persist_dir:
            retriever = SourceRetriever(
                codebase_path=str(multi_lang_project), persist_path=persist_dir
            )

            # Index only Python files
            retriever.index_codebase(file_extensions=[".py"])
            stats = retriever.get_stats()

            assert stats["total_files"] == 1  # Only app.py

    def test_index_creates_persist_directory(self, temp_project):
        with tempfile.TemporaryDirectory() as persist_dir:
            persist_path = Path(persist_dir) / "chroma"
            retriever = SourceRetriever(
                codebase_path=str(temp_project), persist_path=str(persist_path)
            )

            retriever.index_codebase()

            assert persist_path.exists()


class TestSourceRetrieverSearch:
    """Test semantic search functionality."""

    @pytest.fixture
    def indexed_retriever(self, temp_project):
        with tempfile.TemporaryDirectory() as persist_dir:
            retriever = SourceRetriever(codebase_path=str(temp_project), persist_path=persist_dir)
            retriever.index_codebase()
            yield retriever

    def test_get_sources_before_indexing(self, temp_project):
        with tempfile.TemporaryDirectory() as persist_dir:
            retriever = SourceRetriever(codebase_path=str(temp_project), persist_path=persist_dir)
            results, _ = retriever.get_sources_for_question("test")
            assert results == []

    def test_search_returns_results(self, indexed_retriever):
        results, _ = indexed_retriever.get_sources_for_question("calculator", n_results=5)

        assert len(results) > 0

    def test_search_respects_n_results(self, indexed_retriever):
        results, _ = indexed_retriever.get_sources_for_question("function", n_results=2)

        assert len(results) <= 2

    def test_search_returns_relevant_results(self, indexed_retriever):
        results, _ = indexed_retriever.get_sources_for_question(
            "add numbers calculator", n_results=5
        )

        # Should find calculator-related code
        result_text = " ".join([str(r) for r in results])
        assert "add" in result_text.lower() or "calculator" in result_text.lower()


class TestIncrementalReindex:
    """Test incremental reindexing."""

    def test_detects_modified_files(self, temp_project):
        with tempfile.TemporaryDirectory() as persist_dir:
            retriever = SourceRetriever(codebase_path=str(temp_project), persist_path=persist_dir)

            # Initial index
            retriever.index_codebase()

            # Modify a file
            main_py = temp_project / "main.py"
            main_py.write_text("def brand_new_function(): pass")

            # Incremental reindex
            retriever.incremental_reindex()

            # Search should find the new function
            results, _ = retriever.get_sources_for_question("brand_new_function")
            result_text = " ".join([str(r) for r in results])
            assert "brand_new" in result_text.lower()

    def test_handles_new_files(self, temp_project):
        with tempfile.TemporaryDirectory() as persist_dir:
            retriever = SourceRetriever(codebase_path=str(temp_project), persist_path=persist_dir)

            # Initial index
            retriever.index_codebase()

            # Add a new file with unique content
            new_file = temp_project / "new_module.py"
            new_file.write_text("def unique_xyz_function_12345(): pass")

            # Incremental reindex
            retriever.incremental_reindex()

            # Verify the new function is searchable
            results, _ = retriever.get_sources_for_question("unique_xyz_function_12345")
            result_text = " ".join([str(r) for r in results])
            assert "unique_xyz_function_12345" in result_text.lower()

    def test_incremental_reindex_with_parse_error(self, temp_project):
        with tempfile.TemporaryDirectory() as persist_dir:
            retriever = SourceRetriever(codebase_path=str(temp_project), persist_path=persist_dir)
            retriever.index_codebase()

            # Create broken file
            (temp_project / "broken.py").write_text("def (")

            # Should not raise exception
            result = retriever.incremental_reindex()

            # Check result properties if possible, or just sufficient that it didn't crash
            assert result is not None
            assert result["files_added"] == 1 or result["files_modified"] == 1

    def test_index_codebase_with_parse_error(self, temp_project):
        with tempfile.TemporaryDirectory() as persist_dir:
            retriever = SourceRetriever(
                codebase_path=str(temp_project), persist_path=persist_dir, parallel=False
            )

            with patch.object(retriever.parser, "parse_file", side_effect=Exception("Boom")):
                retriever.index_codebase()
                # Should handle error and continue/finish
                assert retriever.stats["parse_errors"] > 0


class TestUpsertBehavior:
    """Test that upsert-based indexing is idempotent and handles stale chunks."""

    def test_index_codebase_upsert_idempotent(self, temp_project):
        """Re-indexing same codebase produces same chunk count (no duplicates)."""
        with tempfile.TemporaryDirectory() as persist_dir:
            retriever = SourceRetriever(codebase_path=str(temp_project), persist_path=persist_dir)

            retriever.index_codebase()
            count_first = retriever.collection.count()

            # Re-index the same codebase
            retriever.index_codebase()
            count_second = retriever.collection.count()

            assert count_first == count_second
            assert count_first > 0

    def test_stale_chunk_removal(self, temp_project):
        """After deleting a file and re-indexing, old chunks are removed."""
        with tempfile.TemporaryDirectory() as persist_dir:
            retriever = SourceRetriever(codebase_path=str(temp_project), persist_path=persist_dir)

            retriever.index_codebase()
            count_before = retriever.collection.count()

            # Delete a file
            (temp_project / "utils.py").unlink()

            # Re-index
            retriever.index_codebase()
            count_after = retriever.collection.count()

            assert count_after < count_before


class TestCheckpointing:
    """Test checkpoint save/load/resume functionality."""

    def test_checkpoint_save_and_load(self, tmp_path):
        """Checkpoint round-trip: save → load → verify data."""
        cp_path = tmp_path / "checkpoint.json"

        _save_checkpoint(cp_path, chunks_completed=500, total_chunks=1000)

        data = _load_checkpoint(cp_path)
        assert data is not None
        assert data["chunks_completed"] == 500
        assert data["total_chunks"] == 1000
        assert "timestamp" in data

    def test_checkpoint_load_missing_file(self, tmp_path):
        """Returns None when checkpoint file doesn't exist."""
        data = _load_checkpoint(tmp_path / "nonexistent.json")
        assert data is None

    def test_checkpoint_load_corrupted(self, tmp_path):
        """Returns None for corrupted checkpoint file."""
        cp_path = tmp_path / "checkpoint.json"
        cp_path.write_text("not valid json{{{")

        data = _load_checkpoint(cp_path)
        assert data is None

    def test_checkpoint_cleanup_on_success(self, temp_project):
        """Checkpoint file is deleted after successful indexing."""
        with tempfile.TemporaryDirectory() as persist_dir:
            retriever = SourceRetriever(codebase_path=str(temp_project), persist_path=persist_dir)

            retriever.index_codebase()

            checkpoint_path = Path(persist_dir).parent / "checkpoint.json"
            # Also check the actual persist dir parent
            actual_cp = Path(persist_dir) / ".." / "checkpoint.json"
            assert not checkpoint_path.exists() or not actual_cp.resolve().exists()

    def test_checkpoint_load_none_path(self):
        """Returns None when checkpoint_path is None."""
        data = _load_checkpoint(None)
        assert data is None
