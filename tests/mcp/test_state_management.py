"""Test state management for isolation between tests and calls."""

import asyncio
import pytest
from pathlib import Path
from codegrok_mcp.mcp.state import get_state, reset_state, MCPSessionState
from codegrok_mcp.mcp.server import learn as learn_tool

# Access underlying function from FastMCP FunctionTool wrapper
learn_fn = learn_tool.fn


def learn(**kwargs):
    """Helper to run async learn function synchronously."""
    return asyncio.get_event_loop().run_until_complete(learn_fn(**kwargs))


class TestStateIsolation:
    """Verify state isolation works correctly for testing."""

    def test_initial_state_is_empty(self):
        reset_state()
        state = get_state()

        assert state.retriever is None
        assert state.codebase_path is None
        assert state.is_loaded is False

    def test_state_persists_between_calls(self, temp_project):
        reset_state()

        # First call sets state
        learn(path=str(temp_project))
        state1 = get_state()
        assert state1.is_loaded is True

        # Second call sees same state
        state2 = get_state()
        assert state2.is_loaded is True
        assert state2.codebase_path == state1.codebase_path

    def test_reset_clears_state(self, temp_project):
        learn(path=str(temp_project))
        assert get_state().is_loaded is True

        reset_state()

        assert get_state().is_loaded is False
        assert get_state().retriever is None

    def test_multiple_resets_are_safe(self):
        reset_state()
        reset_state()
        reset_state()

        state = get_state()
        assert state is not None
        assert not state.is_loaded


class TestStateSingleton:
    """Test singleton behavior of state."""

    def test_same_instance_returned(self):
        reset_state()

        state1 = get_state()
        state2 = get_state()

        assert state1 is state2

    def test_new_instance_after_reset(self):
        state1 = get_state()
        reset_state()
        state2 = get_state()

        # After reset, we get a fresh instance
        assert state1 is not state2
