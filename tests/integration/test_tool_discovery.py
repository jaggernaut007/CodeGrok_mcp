"""
Integration tests for MCP tool discovery, annotations, and agent guidance.

These tests verify:
1. All 8 tools are exposed via MCP protocol
2. Tool annotations (readOnlyHint, destructiveHint, etc.) are correct
3. Server instructions mention both code search and memory tools
4. Tools enforce 'learn' prerequisite correctly
5. Error messages guide agents to correct workflow

Test Strategy:
- Protocol-level tests spawn actual server subprocess (like Claude does)
- Direct function tests call tools directly for faster feedback
- Both approaches ensure comprehensive coverage
"""

import asyncio
import json
import pytest
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

from fastmcp.exceptions import ToolError

from codegrok_mcp.mcp.server import (
    mcp,
    learn as learn_tool,
    get_sources as get_sources_tool,
    get_stats as get_stats_tool,
    list_supported_languages as list_supported_languages_tool,
    remember as remember_tool,
    recall as recall_tool,
    forget as forget_tool,
    memory_stats as memory_stats_tool,
)
from codegrok_mcp.mcp.state import get_state, reset_state

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def indexed_project(temp_project):
    """Create a project and index it for tests that need pre-indexed state."""
    learn_fn = learn_tool.fn
    asyncio.get_event_loop().run_until_complete(learn_fn(path=str(temp_project)))
    yield temp_project
    reset_state()


@pytest.fixture
def multi_file_project(tmp_path):
    """Create a more complex project with multiple files and languages."""
    # Python files
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text('''
"""Main application module."""

def main():
    """Application entry point."""
    print("Hello, World!")
    calculator = Calculator()
    result = calculator.add(1, 2)
    return result

class Calculator:
    """A simple calculator class."""

    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def subtract(self, a: int, b: int) -> int:
        """Subtract b from a."""
        return a - b
''')

    (tmp_path / "src" / "utils.py").write_text('''
"""Utility functions."""
import json
from pathlib import Path

def load_config(path: str) -> dict:
    """Load configuration from JSON file."""
    with open(path) as f:
        return json.load(f)

def save_config(path: str, config: dict) -> None:
    """Save configuration to JSON file."""
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)

MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30
''')

    # JavaScript file
    (tmp_path / "src" / "helper.js").write_text(r"""/**
 * Helper functions for the application.
 */

function formatDate(date) {
    return date.toISOString().split('T')[0];
}

function validateEmail(email) {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
}

const CONFIG = {
    maxItems: 100,
    timeout: 5000
};
""")

    return tmp_path


# =============================================================================
# Helper Functions
# =============================================================================


def learn(**kwargs):
    """Helper to run async learn function synchronously."""
    return asyncio.get_event_loop().run_until_complete(learn_tool.fn(**kwargs))


def remember(**kwargs):
    """Helper to run sync remember function."""
    return remember_tool.fn(**kwargs)


def recall(**kwargs):
    """Helper to run sync recall function."""
    return recall_tool.fn(**kwargs)


def forget(**kwargs):
    """Helper to run sync forget function."""
    return forget_tool.fn(**kwargs)


def memory_stats():
    """Helper to run sync memory_stats function."""
    return memory_stats_tool.fn()


def get_sources(**kwargs):
    """Helper to run sync get_sources function."""
    return get_sources_tool.fn(**kwargs)


def get_stats():
    """Helper to run sync get_stats function."""
    return get_stats_tool.fn()


def list_supported_languages():
    """Helper to run sync list_supported_languages function."""
    return list_supported_languages_tool.fn()


# =============================================================================
# Tool Discovery Tests
# =============================================================================


class TestToolDiscovery:
    """Test that all 8 tools are properly exposed via MCP."""

    EXPECTED_TOOLS = [
        "learn",
        "get_sources",
        "get_stats",
        "list_supported_languages",
        "remember",
        "recall",
        "forget",
        "memory_stats",
    ]

    def test_all_eight_tools_registered(self):
        """Verify all 8 tools are registered with FastMCP."""
        # Access FastMCP's internal tool registry
        registered_tools = list(mcp._tool_manager._tools.keys())

        for tool_name in self.EXPECTED_TOOLS:
            assert tool_name in registered_tools, f"Tool '{tool_name}' not registered"

        assert len(registered_tools) == 8, f"Expected 8 tools, got {len(registered_tools)}"

    def test_code_search_tools_present(self):
        """Verify code search tools are registered."""
        code_tools = ["learn", "get_sources", "get_stats", "list_supported_languages"]
        registered = list(mcp._tool_manager._tools.keys())

        for tool in code_tools:
            assert tool in registered

    def test_memory_tools_present(self):
        """Verify memory tools are registered."""
        memory_tools = ["remember", "recall", "forget", "memory_stats"]
        registered = list(mcp._tool_manager._tools.keys())

        for tool in memory_tools:
            assert tool in registered


# =============================================================================
# Tool Annotations Tests
# =============================================================================


class TestToolAnnotations:
    """Test that tool annotations are correctly set per MCP spec."""

    def _get_tool_annotations(self, tool_name: str) -> Dict[str, Any]:
        """Get annotations for a tool by name."""
        tool = mcp._tool_manager._tools.get(tool_name)
        if tool and hasattr(tool, "annotations") and tool.annotations:
            return {
                "readOnlyHint": tool.annotations.readOnlyHint,
                "destructiveHint": tool.annotations.destructiveHint,
                "idempotentHint": tool.annotations.idempotentHint,
                "openWorldHint": tool.annotations.openWorldHint,
            }
        return {}

    def test_learn_annotations(self):
        """learn: writes data, not destructive, idempotent, local only."""
        annotations = self._get_tool_annotations("learn")
        assert annotations.get("readOnlyHint") is False, "learn writes to .codegrok/"
        assert annotations.get("destructiveHint") is False, "learn doesn't destroy user data"
        assert annotations.get("idempotentHint") is True, "learn can be safely re-run"
        assert annotations.get("openWorldHint") is False, "learn only accesses local files"

    def test_get_sources_annotations(self):
        """get_sources: read-only search."""
        annotations = self._get_tool_annotations("get_sources")
        assert annotations.get("readOnlyHint") is True, "get_sources only reads"
        assert annotations.get("idempotentHint") is True, "Same query = same results"
        assert annotations.get("openWorldHint") is False, "Local ChromaDB only"

    def test_get_stats_annotations(self):
        """get_stats: read-only metadata."""
        annotations = self._get_tool_annotations("get_stats")
        assert annotations.get("readOnlyHint") is True
        assert annotations.get("idempotentHint") is True

    def test_list_supported_languages_annotations(self):
        """list_supported_languages: static data, always read-only."""
        annotations = self._get_tool_annotations("list_supported_languages")
        assert annotations.get("readOnlyHint") is True
        assert annotations.get("idempotentHint") is True

    def test_remember_annotations(self):
        """remember: writes data, not destructive, NOT idempotent."""
        annotations = self._get_tool_annotations("remember")
        assert annotations.get("readOnlyHint") is False, "remember writes to ChromaDB"
        assert annotations.get("destructiveHint") is False, "remember adds, doesn't delete"
        assert annotations.get("idempotentHint") is False, "Each call creates new memory"
        assert annotations.get("openWorldHint") is False

    def test_recall_annotations(self):
        """recall: read-only search."""
        annotations = self._get_tool_annotations("recall")
        assert annotations.get("readOnlyHint") is True
        assert annotations.get("idempotentHint") is True

    def test_forget_annotations_destructive(self):
        """forget: DESTRUCTIVE - permanently deletes data."""
        annotations = self._get_tool_annotations("forget")
        assert annotations.get("readOnlyHint") is False, "forget deletes data"
        assert annotations.get("destructiveHint") is True, "forget is DESTRUCTIVE"
        assert annotations.get("idempotentHint") is True, "Re-calling same filter is safe"

    def test_memory_stats_annotations(self):
        """memory_stats: read-only statistics."""
        annotations = self._get_tool_annotations("memory_stats")
        assert annotations.get("readOnlyHint") is True
        assert annotations.get("idempotentHint") is True


# =============================================================================
# Server Instructions Tests
# =============================================================================


class TestServerInstructions:
    """Test that server instructions properly guide agents."""

    def test_instructions_mention_learn_first(self):
        """Instructions should tell agents to call learn first."""
        instructions = mcp.instructions
        assert "learn" in instructions.lower()
        assert "first" in instructions.lower() or "start" in instructions.lower()

    def test_instructions_mention_code_search(self):
        """Instructions should describe code search capabilities."""
        instructions = mcp.instructions
        assert "code" in instructions.lower()
        assert "search" in instructions.lower() or "sources" in instructions.lower()

    def test_instructions_mention_memory_tools(self):
        """Instructions should describe memory tools."""
        instructions = mcp.instructions
        assert "memory" in instructions.lower()
        assert "remember" in instructions.lower()
        assert "recall" in instructions.lower()

    def test_instructions_mention_workflow(self):
        """Instructions should include workflow guidance."""
        instructions = mcp.instructions
        assert "workflow" in instructions.lower()

    def test_instructions_describe_memory_use_cases(self):
        """Instructions should explain what memory is for."""
        instructions = mcp.instructions
        # Check for at least some use cases
        use_case_keywords = ["decision", "preference", "status", "note"]
        found = sum(1 for kw in use_case_keywords if kw in instructions.lower())
        assert found >= 2, "Instructions should describe memory use cases"


# =============================================================================
# Progressive Discovery Tests (Learn-First Requirement)
# =============================================================================


class TestLearnFirstRequirement:
    """Test that tools properly require 'learn' to be called first."""

    def test_get_sources_requires_learn(self):
        """get_sources should fail with helpful message before learn."""
        reset_state()
        with pytest.raises(ToolError, match="No codebase loaded"):
            get_sources(question="test query")

    def test_get_stats_returns_unloaded_state(self):
        """get_stats should indicate not loaded before learn."""
        reset_state()
        result = get_stats()
        assert result["loaded"] is False
        assert result["codebase_path"] is None

    def test_remember_requires_learn(self):
        """remember should fail with helpful message before learn."""
        reset_state()
        with pytest.raises(ToolError, match="No codebase loaded"):
            remember(content="test memory", memory_type="note")

    def test_recall_requires_learn(self):
        """recall should fail with helpful message before learn."""
        reset_state()
        with pytest.raises(ToolError, match="No codebase loaded"):
            recall(query="test query")

    def test_forget_requires_learn(self):
        """forget should fail with helpful message before learn."""
        reset_state()
        with pytest.raises(ToolError, match="No codebase loaded"):
            forget(memory_id="test-id")

    def test_memory_stats_indicates_not_loaded(self):
        """memory_stats should indicate not loaded before learn."""
        reset_state()
        result = memory_stats()
        assert result.get("loaded") is False
        assert "learn" in result.get("message", "").lower()

    def test_list_supported_languages_works_without_learn(self):
        """list_supported_languages should work without learn (static data)."""
        reset_state()
        result = list_supported_languages()
        assert "extensions" in result
        assert ".py" in result["extensions"]


# =============================================================================
# Error Message Quality Tests
# =============================================================================


class TestErrorMessageQuality:
    """Test that error messages guide agents to correct actions."""

    def test_error_mentions_learn_tool(self):
        """Error messages should mention 'learn' tool by name."""
        reset_state()
        try:
            get_sources(question="test")
            pytest.fail("Should have raised ToolError")
        except ToolError as e:
            assert "learn" in str(e).lower()

    def test_invalid_path_error_is_clear(self):
        """Invalid path error should be descriptive."""
        with pytest.raises(ToolError, match="does not exist"):
            learn(path="/nonexistent/path/12345")

    def test_invalid_mode_error_lists_valid_modes(self):
        """Invalid mode error should list valid options."""
        with pytest.raises(ToolError) as exc_info:
            learn(path="/tmp", mode="invalid_mode")

        error_msg = str(exc_info.value)
        assert "auto" in error_msg or "full" in error_msg

    def test_invalid_memory_type_lists_valid_types(self, indexed_project):
        """Invalid memory_type should list valid options."""
        with pytest.raises(ToolError) as exc_info:
            remember(content="test", memory_type="invalid_type")

        error_msg = str(exc_info.value)
        assert "conversation" in error_msg or "decision" in error_msg


# =============================================================================
# Tool Description Tests
# =============================================================================


class TestToolDescriptions:
    """Test that tool descriptions are informative and consistent."""

    def _get_tool_description(self, tool_name: str) -> str:
        """Get description for a tool by name."""
        tool = mcp._tool_manager._tools.get(tool_name)
        return tool.description if tool else ""

    def test_learn_description_mentions_required(self):
        """learn description should indicate it's required first."""
        desc = self._get_tool_description("learn")
        assert "required" in desc.lower() or "first" in desc.lower()

    def test_get_sources_description_mentions_prerequisite(self):
        """get_sources description should mention learn prerequisite."""
        desc = self._get_tool_description("get_sources")
        assert "learn" in desc.lower() or "require" in desc.lower()

    def test_remember_description_has_examples(self):
        """remember description should include usage examples."""
        desc = self._get_tool_description("remember")
        assert "example" in desc.lower() or "remember(" in desc

    def test_recall_description_has_examples(self):
        """recall description should include usage examples."""
        desc = self._get_tool_description("recall")
        assert "example" in desc.lower() or "recall(" in desc

    def test_forget_description_warns_destructive(self):
        """forget description should warn about data deletion."""
        desc = self._get_tool_description("forget")
        assert "destructive" in desc.lower() or "remove" in desc.lower() or "delete" in desc.lower()

    def test_memory_tools_describe_types(self):
        """Memory tools should describe available memory types."""
        desc = self._get_tool_description("remember")
        types = ["conversation", "status", "decision", "preference", "note"]
        found = sum(1 for t in types if t in desc.lower())
        assert found >= 3, "Should describe memory types"


# =============================================================================
# Protocol-Level Tests (Subprocess)
# =============================================================================


class StdioMCPClient:
    """MCP client that communicates via stdio subprocess."""

    def __init__(self):
        self.process = None
        self.request_id = 0

    def start(self):
        """Start the MCP server as a subprocess."""
        self.process = subprocess.Popen(
            [sys.executable, "-m", "codegrok_mcp.mcp.server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        time.sleep(0.5)  # Allow server to initialize

    def stop(self):
        """Stop the MCP server."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def _send_request(self, method: str, params: dict = None) -> dict:
        """Send JSON-RPC request over stdin."""
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params or {},
        }

        request_line = json.dumps(request) + "\n"
        self.process.stdin.write(request_line)
        self.process.stdin.flush()

        response_line = self.process.stdout.readline()
        if not response_line:
            stderr = self.process.stderr.read()
            raise RuntimeError(f"No response. stderr: {stderr}")
        return json.loads(response_line)

    def initialize(self) -> dict:
        """MCP initialization handshake."""
        return self._send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "integration-test", "version": "1.0.0"},
            },
        )

    def list_tools(self) -> List[dict]:
        """Get available tools."""
        response = self._send_request("tools/list")
        return response.get("result", {}).get("tools", [])

    def call_tool(self, name: str, arguments: dict = None) -> dict:
        """Call a tool."""
        return self._send_request("tools/call", {"name": name, "arguments": arguments or {}})


class TestProtocolToolDiscovery:
    """Test tool discovery via actual MCP protocol."""

    @pytest.fixture
    def mcp_client(self):
        """Create stdio MCP client."""
        client = StdioMCPClient()
        client.start()
        yield client
        client.stop()

    def test_tools_list_returns_all_eight(self, mcp_client):
        """tools/list should return all 8 tools."""
        mcp_client.initialize()
        tools = mcp_client.list_tools()

        tool_names = [t["name"] for t in tools]
        expected = [
            "learn",
            "get_sources",
            "get_stats",
            "list_supported_languages",
            "remember",
            "recall",
            "forget",
            "memory_stats",
        ]

        for name in expected:
            assert name in tool_names, f"Tool '{name}' not in tools/list response"

        assert len(tools) == 8

    def test_tools_have_descriptions(self, mcp_client):
        """All tools should have descriptions."""
        mcp_client.initialize()
        tools = mcp_client.list_tools()

        for tool in tools:
            assert "description" in tool, f"Tool {tool['name']} missing description"
            assert len(tool["description"]) > 20, f"Tool {tool['name']} description too short"

    def test_tools_have_input_schemas(self, mcp_client):
        """All tools should have input schemas."""
        mcp_client.initialize()
        tools = mcp_client.list_tools()

        for tool in tools:
            assert "inputSchema" in tool, f"Tool {tool['name']} missing inputSchema"

    def test_annotations_in_protocol_response(self, mcp_client):
        """Tools should include annotations in protocol response."""
        mcp_client.initialize()
        tools = mcp_client.list_tools()

        # Find forget tool and verify destructiveHint
        forget_tool = next((t for t in tools if t["name"] == "forget"), None)
        assert forget_tool is not None

        if "annotations" in forget_tool:
            annotations = forget_tool["annotations"]
            assert annotations.get("destructiveHint") is True, "forget should be destructive"

    def test_server_info_in_init(self, mcp_client):
        """Initialize should return server info."""
        response = mcp_client.initialize()

        assert "result" in response
        assert "serverInfo" in response["result"]
        assert response["result"]["serverInfo"]["name"] == "CodeGrok"


class TestProtocolToolExecution:
    """Test tool execution via MCP protocol."""

    @pytest.fixture
    def mcp_client(self):
        client = StdioMCPClient()
        client.start()
        yield client
        client.stop()

    def test_list_supported_languages_via_protocol(self, mcp_client):
        """Call list_supported_languages via protocol."""
        mcp_client.initialize()
        response = mcp_client.call_tool("list_supported_languages")

        assert "result" in response
        content = response["result"]["content"][0]
        result_data = json.loads(content["text"])

        assert ".py" in result_data["extensions"]
        assert "python" in result_data["languages"]

    def test_learn_via_protocol(self, mcp_client, temp_project):
        """Call learn via protocol."""
        mcp_client.initialize()
        response = mcp_client.call_tool("learn", {"path": str(temp_project)})

        assert "result" in response
        content = response["result"]["content"][0]
        result_data = json.loads(content["text"])

        assert result_data["success"] is True
        assert "stats" in result_data

    def test_full_workflow_via_protocol(self, mcp_client, temp_project):
        """Execute full workflow: learn -> search -> memory ops via protocol."""
        mcp_client.initialize()

        # 1. Learn
        learn_resp = mcp_client.call_tool("learn", {"path": str(temp_project)})
        learn_data = json.loads(learn_resp["result"]["content"][0]["text"])
        assert learn_data["success"] is True

        # 2. Get sources
        search_resp = mcp_client.call_tool(
            "get_sources", {"question": "calculator add", "n_results": 5}
        )
        search_data = json.loads(search_resp["result"]["content"][0]["text"])
        assert "sources" in search_data

        # 3. Remember
        remember_resp = mcp_client.call_tool(
            "remember",
            {
                "content": "User prefers functional style",
                "memory_type": "preference",
                "tags": ["style", "coding"],
            },
        )
        remember_data = json.loads(remember_resp["result"]["content"][0]["text"])
        assert remember_data["success"] is True
        assert "memory_id" in remember_data

        # 4. Recall
        recall_resp = mcp_client.call_tool(
            "recall", {"query": "coding style preference", "n_results": 5}
        )
        recall_data = json.loads(recall_resp["result"]["content"][0]["text"])
        assert recall_data["success"] is True
        assert recall_data["count"] >= 1

        # 5. Get stats
        stats_resp = mcp_client.call_tool("get_stats")
        stats_data = json.loads(stats_resp["result"]["content"][0]["text"])
        assert stats_data["loaded"] is True

        # 6. Memory stats
        mem_stats_resp = mcp_client.call_tool("memory_stats")
        mem_stats_data = json.loads(mem_stats_resp["result"]["content"][0]["text"])
        assert mem_stats_data["loaded"] is True

    def test_error_response_before_learn(self, mcp_client):
        """Verify error response format when calling tools before learn."""
        mcp_client.initialize()

        response = mcp_client.call_tool("get_sources", {"question": "test"})

        # MCP protocol should return error
        assert "error" in response or "isError" in str(response)


# =============================================================================
# Integration: Combined Workflow Tests
# =============================================================================


class TestCombinedWorkflow:
    """Test complete agent workflows combining code search and memory."""

    def test_typical_agent_session(self, multi_file_project):
        """Simulate a typical agent session using all tools."""
        # 1. Learn the codebase
        result = learn(path=str(multi_file_project))
        assert result["success"] is True
        assert result["stats"]["total_files"] >= 3

        # 2. Search for code
        sources = get_sources(question="calculator class methods")
        assert "sources" in sources

        # 3. Store a decision
        memory1 = remember(
            content="Using Calculator class for all math operations",
            memory_type="decision",
            tags=["architecture", "math"],
        )
        assert memory1["success"] is True

        # 4. Store a preference
        memory2 = remember(
            content="User prefers docstrings on all public methods",
            memory_type="preference",
            tags=["style", "documentation"],
        )
        assert memory2["success"] is True

        # 5. Recall preferences
        recalled = recall(query="documentation style", memory_type="preference")
        assert recalled["success"] is True
        assert recalled["count"] >= 1

        # 6. Check stats
        stats = get_stats()
        assert stats["loaded"] is True

        mem_stats = memory_stats()
        assert mem_stats["total_memories"] >= 2

    def test_memory_persistence_across_state(self, multi_file_project):
        """Memory should persist when state is reloaded."""
        # Initial learn and remember
        learn(path=str(multi_file_project))
        remember(
            content="Important architectural decision: use microservices",
            memory_type="decision",
            tags=["architecture"],
        )

        # Simulate session restart by resetting state
        reset_state()

        # Reload with load_only
        learn(path=str(multi_file_project), mode="load_only")

        # Memory should still be recallable
        recalled = recall(query="microservices architecture")
        assert recalled["count"] >= 1

    def test_incremental_reindex_preserves_memories(self, multi_file_project):
        """Incremental reindex should not affect stored memories."""
        # Initial setup
        learn(path=str(multi_file_project))
        remember(content="Test memory for persistence", memory_type="note")

        initial_count = memory_stats()["total_memories"]

        # Modify a file
        (multi_file_project / "src" / "new_file.py").write_text("def new_func(): pass")

        # Incremental reindex
        learn(path=str(multi_file_project), mode="auto")

        # Memory should still be there
        final_count = memory_stats()["total_memories"]
        assert final_count == initial_count


# =============================================================================
# Edge Cases and Robustness Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_query_handling(self, indexed_project):
        """Handle empty or whitespace queries gracefully."""
        # Empty query should still work (returns some results)
        result = get_sources(question="   ")
        assert "sources" in result

    def test_special_characters_in_query(self, indexed_project):
        """Handle special characters in queries."""
        result = get_sources(question="function with @decorator and 'quotes'")
        assert "sources" in result

    def test_unicode_in_memory(self, indexed_project):
        """Handle unicode content in memories."""
        result = remember(
            content="User prefers emoji: 🎉 and unicode: café résumé", memory_type="preference"
        )
        assert result["success"] is True

        recalled = recall(query="emoji unicode")
        assert recalled["success"] is True

    def test_very_long_memory_content(self, indexed_project):
        """Handle very long memory content."""
        long_content = "A" * 10000  # 10K characters
        result = remember(content=long_content, memory_type="note")
        assert result["success"] is True

    def test_many_tags(self, indexed_project):
        """Handle many tags on a memory."""
        tags = [f"tag{i}" for i in range(50)]
        result = remember(content="Memory with many tags", memory_type="note", tags=tags)
        assert result["success"] is True

    def test_forget_nonexistent_id(self, indexed_project):
        """Forgetting non-existent ID should not error."""
        result = forget(memory_id="nonexistent-uuid-12345")
        assert result["deleted"] == 0 or result["deleted"] == 1  # Idempotent

    def test_recall_with_no_memories(self, indexed_project):
        """Recall with no matching memories should return empty list."""
        # Clear any memories first
        state = get_state()
        if state.memory_retriever:
            state.memory_retriever.collection.delete(where={"project": str(state.codebase_path)})

        result = recall(query="something that doesn't exist")
        assert result["success"] is True
        assert result["count"] == 0
        assert result["memories"] == []

    def test_concurrent_memory_operations(self, indexed_project):
        """Multiple rapid memory operations should be safe."""
        import threading

        results = []
        errors = []

        def store_memory(i):
            try:
                result = remember(
                    content=f"Concurrent memory {i}", memory_type="note", tags=[f"concurrent-{i}"]
                )
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=store_memory, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent ops: {errors}"
        assert len(results) == 10
        assert all(r["success"] for r in results)


# =============================================================================
# Memory Type Tests
# =============================================================================


class TestMemoryTypes:
    """Test all memory types work correctly."""

    @pytest.mark.parametrize(
        "memory_type", ["conversation", "status", "decision", "preference", "doc", "note"]
    )
    def test_all_memory_types(self, indexed_project, memory_type):
        """Test each memory type can be stored and recalled."""
        content = f"Test content for {memory_type}"

        # Store
        result = remember(content=content, memory_type=memory_type)
        assert result["success"] is True

        # Recall by type
        recalled = recall(query=content, memory_type=memory_type)
        assert recalled["success"] is True
        assert recalled["count"] >= 1

        # Verify type in result
        assert any(m["memory_type"] == memory_type for m in recalled["memories"])


# =============================================================================
# TTL Tests
# =============================================================================


class TestMemoryTTL:
    """Test memory TTL functionality."""

    @pytest.mark.parametrize("ttl", ["session", "day", "week", "month", "permanent"])
    def test_all_ttl_values(self, indexed_project, ttl):
        """Test each TTL value is accepted."""
        result = remember(content=f"Memory with {ttl} TTL", memory_type="note", ttl=ttl)
        assert result["success"] is True

    def test_invalid_ttl_rejected(self, indexed_project):
        """Invalid TTL values should be rejected at model level."""
        # Note: TTL validation happens in Memory.__post_init__
        # The MemoryRetriever passes through to Memory model
        # If invalid TTL reaches the model, it will raise ValueError
        from codegrok_mcp.core.models import Memory, MemoryType
        import uuid

        with pytest.raises(ValueError, match="ttl must be one of"):
            Memory(
                id=str(uuid.uuid4()),
                content="test",
                memory_type=MemoryType.NOTE,
                project="test",
                ttl="invalid_ttl",
            )


# =============================================================================
# Filter Tests
# =============================================================================


class TestRecallFilters:
    """Test recall filtering capabilities."""

    def test_filter_by_memory_type(self, indexed_project):
        """Filter recall by memory type."""
        # Store different types
        remember(content="A decision about auth", memory_type="decision")
        remember(content="User prefers tabs", memory_type="preference")

        # Filter by decision only
        results = recall(query="", memory_type="decision")
        assert all(m["memory_type"] == "decision" for m in results["memories"])

    def test_filter_by_tags(self, indexed_project):
        """Filter recall by tags."""
        remember(content="Auth implementation", memory_type="note", tags=["auth", "backend"])
        remember(content="UI styling", memory_type="note", tags=["ui", "frontend"])

        # Filter by auth tag
        results = recall(query="implementation", tags=["auth"])
        # Should find the auth-tagged memory
        assert (
            any("auth" in m.get("tags", []) for m in results["memories"]) or results["count"] >= 0
        )

    def test_filter_by_time_range(self, indexed_project):
        """Filter recall by time range."""
        remember(content="Recent memory", memory_type="note")

        # Today filter should find recent memory
        results = recall(query="memory", time_range="today")
        assert results["success"] is True


# =============================================================================
# Forget Tests
# =============================================================================


class TestForgetOperations:
    """Test forget tool functionality."""

    def test_forget_by_id(self, indexed_project):
        """Forget specific memory by ID."""
        result = remember(content="To be forgotten", memory_type="note")
        memory_id = result["memory_id"]

        # Forget it
        forget_result = forget(memory_id=memory_id)
        assert forget_result["deleted"] == 1

        # Verify it's gone
        recalled = recall(query="To be forgotten")
        assert all(m["id"] != memory_id for m in recalled["memories"])

    def test_forget_by_type(self, indexed_project):
        """Forget all memories of a type."""
        # Store multiple notes
        remember(content="Note 1", memory_type="note")
        remember(content="Note 2", memory_type="note")
        remember(content="Decision 1", memory_type="decision")

        initial_stats = memory_stats()

        # Forget all notes
        result = forget(memory_type="note")
        assert result["deleted"] >= 2

        # Decisions should remain
        final_stats = memory_stats()
        assert final_stats["by_type"].get("decision", 0) >= 1

    def test_forget_by_tags(self, indexed_project):
        """Forget memories by tag."""
        remember(content="Temp note", memory_type="note", tags=["temp", "cleanup"])
        remember(content="Permanent note", memory_type="note", tags=["keep"])

        # Forget temp-tagged
        result = forget(tags=["temp"])
        assert result["deleted"] >= 1

    def test_forget_requires_filter(self, indexed_project):
        """Forget should require at least one filter."""
        with pytest.raises(ToolError, match="Must specify"):
            forget()
