"""Tests for background indexing and IndexingStatus."""

import threading
import time
from unittest.mock import patch, MagicMock

import pytest

from codegrok_mcp.mcp.state import IndexingStatus, MCPSessionState, get_state, reset_state


class TestIndexingStatus:
    """Tests for the IndexingStatus thread-safe dataclass."""

    def test_initial_state(self):
        status = IndexingStatus()
        assert status.active is False
        assert status.progress == 0
        assert status.message == ""
        assert status.error is None
        assert status.result is None

    def test_start(self):
        status = IndexingStatus()
        status.start("Starting full index...")
        assert status.active is True
        assert status.progress == 0
        assert status.message == "Starting full index..."
        assert status.error is None
        assert status.result is None

    def test_start_clears_previous_error(self):
        status = IndexingStatus()
        status.fail("previous error")
        status.start("Retrying...")
        assert status.active is True
        assert status.error is None
        assert status.result is None

    def test_start_clears_previous_result(self):
        status = IndexingStatus()
        status.complete({"success": True})
        status.start("Re-indexing...")
        assert status.active is True
        assert status.result is None

    def test_update(self):
        status = IndexingStatus()
        status.start()
        status.update(50, "Halfway there...")
        assert status.progress == 50
        assert status.message == "Halfway there..."

    def test_update_caps_at_99(self):
        status = IndexingStatus()
        status.start()
        status.update(100, "Should cap")
        assert status.progress == 99
        status.update(500, "Way over")
        assert status.progress == 99

    def test_complete(self):
        status = IndexingStatus()
        status.start()
        result = {"success": True, "stats": {"files": 10}}
        status.complete(result)
        assert status.active is False
        assert status.progress == 100
        assert status.message == "Indexing complete"
        assert status.result == result

    def test_fail(self):
        status = IndexingStatus()
        status.start()
        status.fail("Out of memory")
        assert status.active is False
        assert status.message == "Indexing failed: Out of memory"
        assert status.error == "Out of memory"

    def test_to_dict(self):
        status = IndexingStatus()
        status.start("Testing...")
        status.update(42, "Processing...")
        d = status.to_dict()
        assert d == {
            "active": True,
            "progress": 42,
            "message": "Processing...",
            "error": None,
        }

    def test_to_dict_excludes_result(self):
        """to_dict should not include the result field (it's for internal use)."""
        status = IndexingStatus()
        status.complete({"big": "data"})
        d = status.to_dict()
        assert "result" not in d

    def test_thread_safety(self):
        """Multiple threads can update IndexingStatus without errors."""
        status = IndexingStatus()
        status.start()
        errors = []

        def updater(thread_id):
            try:
                for i in range(100):
                    status.update(i, f"Thread {thread_id} at {i}")
                    status.to_dict()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=updater, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert status.active is True  # No one called complete/fail


class TestMCPSessionStateIndexing:
    """Tests for IndexingStatus integration in MCPSessionState."""

    def test_state_has_indexing_status(self):
        state = MCPSessionState()
        assert isinstance(state.indexing, IndexingStatus)
        assert state.indexing.active is False

    def test_each_state_gets_own_indexing(self):
        state1 = MCPSessionState()
        state2 = MCPSessionState()
        state1.indexing.start("s1")
        assert state2.indexing.active is False

    def test_singleton_state_has_indexing(self):
        reset_state()
        state = get_state()
        assert isinstance(state.indexing, IndexingStatus)
        reset_state()


class TestBackgroundProgressCallback:
    """Tests for _create_bg_progress_callback."""

    def test_callback_updates_indexing_status(self):
        from codegrok_mcp.mcp.server import _create_bg_progress_callback

        status = IndexingStatus()
        status.start()
        cb = _create_bg_progress_callback(status)

        cb("files_found", {"files": list(range(100))})
        assert status.progress == 5
        assert "100 files" in status.message

    def test_callback_discovery_progress(self):
        from codegrok_mcp.mcp.server import _create_bg_progress_callback

        status = IndexingStatus()
        status.start()
        cb = _create_bg_progress_callback(status)

        cb("discovery_progress", {"files_found": 5000})
        assert status.progress <= 4
        assert "5000" in status.message

    def test_callback_parsing_start(self):
        from codegrok_mcp.mcp.server import _create_bg_progress_callback

        status = IndexingStatus()
        status.start()
        cb = _create_bg_progress_callback(status)

        cb("parsing_start", {"total": 50})
        assert status.progress == 10
        assert "50 files" in status.message

    def test_callback_embedding_progress_with_eta(self):
        from codegrok_mcp.mcp.server import _create_bg_progress_callback

        status = IndexingStatus()
        status.start()
        cb = _create_bg_progress_callback(status)

        cb("embedding_progress", {"current": 500, "total": 1000, "remaining_seconds": 120})
        assert status.progress > 35
        assert "500/1000" in status.message
        assert "remaining" in status.message

    def test_callback_embedding_progress_without_eta(self):
        from codegrok_mcp.mcp.server import _create_bg_progress_callback

        status = IndexingStatus()
        status.start()
        cb = _create_bg_progress_callback(status)

        cb("embedding_progress", {"current": 500, "total": 1000, "remaining_seconds": None})
        assert "500/1000" in status.message
        assert "remaining" not in status.message

    def test_callback_changes_detected(self):
        from codegrok_mcp.mcp.server import _create_bg_progress_callback

        status = IndexingStatus()
        status.start()
        cb = _create_bg_progress_callback(status)

        cb("changes_detected", {"new": 5, "modified": 3})
        assert status.progress == 10
        assert "5 new" in status.message
        assert "3 modified" in status.message

    def test_callback_complete(self):
        from codegrok_mcp.mcp.server import _create_bg_progress_callback

        status = IndexingStatus()
        status.start()
        cb = _create_bg_progress_callback(status)

        cb("complete", {})
        assert status.progress == 99

    def test_unknown_event_ignored(self):
        from codegrok_mcp.mcp.server import _create_bg_progress_callback

        status = IndexingStatus()
        status.start()
        cb = _create_bg_progress_callback(status)

        cb("unknown_event", {"foo": "bar"})
        # Progress stays at 0 from start
        assert status.progress == 0


class TestLearnToolBehavior:
    """Tests for learn tool's background indexing flow (without actual indexing)."""

    def setup_method(self):
        reset_state()

    def teardown_method(self):
        reset_state()

    @pytest.mark.asyncio
    async def test_learn_returns_in_progress_when_active(self):
        """If indexing is active, learn should return status without starting another."""
        from codegrok_mcp.mcp.server import learn

        state = get_state()
        state.indexing.start("Already running...")

        result = await learn(path="/tmp", mode="auto")
        assert result["status"] == "indexing_in_progress"
        assert "already running" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_learn_returns_completed_result(self):
        """If last indexing completed, learn should return the result."""
        from codegrok_mcp.mcp.server import learn

        state = get_state()
        state.indexing.complete({"success": True, "mode_used": "full", "stats": {"files": 10}})

        result = await learn(path="/tmp", mode="auto")
        assert result["status"] == "complete"
        assert result["stats"] == {"files": 10}
        # Result should be cleared after retrieval
        assert state.indexing.result is None

    @pytest.mark.asyncio
    async def test_learn_raises_on_previous_error(self):
        """If last indexing failed, learn should raise and clear the error."""
        from codegrok_mcp.mcp.server import learn
        from fastmcp.exceptions import ToolError

        state = get_state()
        state.indexing.error = "Disk full"

        with pytest.raises(ToolError, match="Disk full"):
            await learn(path="/tmp", mode="auto")
        # Error should be cleared
        assert state.indexing.error is None

    @pytest.mark.asyncio
    async def test_learn_invalid_mode(self):
        from codegrok_mcp.mcp.server import learn
        from fastmcp.exceptions import ToolError

        with pytest.raises(ToolError, match="Invalid mode"):
            await learn(path="/tmp", mode="bad_mode")

    @pytest.mark.asyncio
    async def test_learn_invalid_path(self):
        from codegrok_mcp.mcp.server import learn
        from fastmcp.exceptions import ToolError

        with pytest.raises(ToolError, match="does not exist"):
            await learn(path="/nonexistent/path/12345", mode="auto")

    @pytest.mark.asyncio
    async def test_learn_starts_background_thread(self, tmp_path):
        """learn should start a daemon thread and return immediately."""
        from codegrok_mcp.mcp.server import learn

        # Create a minimal directory
        (tmp_path / "test.py").write_text("x = 1")

        with patch("codegrok_mcp.mcp.server.threading.Thread") as mock_thread_cls:
            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread

            result = await learn(path=str(tmp_path), mode="full")

            assert result["status"] == "indexing_started"
            mock_thread_cls.assert_called_once()
            assert mock_thread_cls.call_args[1]["daemon"] is True
            mock_thread.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_learn_auto_with_existing_uses_incremental(self, tmp_path):
        """auto mode with existing index should start incremental reindex thread."""
        from codegrok_mcp.mcp.server import learn

        # Create fake existing index
        codegrok_dir = tmp_path / ".codegrok"
        codegrok_dir.mkdir()
        (codegrok_dir / "chroma").mkdir()
        (codegrok_dir / "metadata.json").write_text("{}")

        with patch("codegrok_mcp.mcp.server.threading.Thread") as mock_thread_cls:
            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread

            result = await learn(path=str(tmp_path), mode="auto")

            assert result["status"] == "indexing_started"
            # Should use incremental reindex
            call_args = mock_thread_cls.call_args
            assert call_args[1]["target"].__name__ == "_run_incremental_reindex_bg"

    @pytest.mark.asyncio
    async def test_learn_full_mode_uses_full_index(self, tmp_path):
        """full mode should always start full index thread."""
        from codegrok_mcp.mcp.server import learn

        (tmp_path / "test.py").write_text("x = 1")

        with patch("codegrok_mcp.mcp.server.threading.Thread") as mock_thread_cls:
            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread

            result = await learn(path=str(tmp_path), mode="full")

            call_args = mock_thread_cls.call_args
            assert call_args[1]["target"].__name__ == "_run_full_index_bg"


class TestGetStatsIndexingInfo:
    """Tests for get_stats returning indexing progress."""

    def setup_method(self):
        reset_state()

    def teardown_method(self):
        reset_state()

    def test_get_stats_includes_indexing_when_active(self):
        from codegrok_mcp.mcp.server import get_stats

        state = get_state()
        state.indexing.start("Indexing...")
        state.indexing.update(42, "Processing...")

        result = get_stats()
        assert "indexing" in result
        assert result["indexing"]["active"] is True
        assert result["indexing"]["progress"] == 42

    def test_get_stats_includes_indexing_on_error(self):
        from codegrok_mcp.mcp.server import get_stats

        state = get_state()
        state.indexing.fail("Something broke")

        result = get_stats()
        assert "indexing" in result
        assert result["indexing"]["error"] == "Something broke"

    def test_get_stats_no_indexing_when_idle(self):
        from codegrok_mcp.mcp.server import get_stats

        result = get_stats()
        assert "indexing" not in result
        assert result["loaded"] is False
