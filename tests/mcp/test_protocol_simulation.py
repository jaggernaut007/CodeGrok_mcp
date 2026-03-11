"""
Simulate MCP protocol messages over STDIO as if from a real agent (Claude).

MCP Protocol Overview:
- Transport: stdio (stdin/stdout)
- Format: JSON-RPC 2.0
- Message Types:
  - initialize: Client handshake
  - tools/list: Discover available tools
  - tools/call: Execute a tool

This test file spawns the actual MCP server as a subprocess and
communicates with it via stdin/stdout - the EXACT same way Claude does.
"""

import pytest
import json
import subprocess
import sys
import time
from pathlib import Path


class StdioMCPClient:
    """
    A real stdio-based MCP client that spawns the server as a subprocess.
    This simulates EXACTLY how Claude Desktop or any MCP client works.
    """

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
            bufsize=1,  # Line buffered
        )
        # Give server time to initialize
        time.sleep(0.5)

    def stop(self):
        """Stop the MCP server."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def _send_request(self, method: str, params: dict = None) -> dict:
        """Send a JSON-RPC request over stdin and read response from stdout."""
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params or {},
        }

        # Write to stdin (this is what Claude does!)
        request_line = json.dumps(request) + "\n"
        self.process.stdin.write(request_line)
        self.process.stdin.flush()

        # Read from stdout (server's response)
        response_line = self.process.stdout.readline()
        if not response_line:
            stderr = self.process.stderr.read()
            raise RuntimeError(f"No response from server. stderr: {stderr}")
        return json.loads(response_line)

    def initialize(self) -> dict:
        """MCP initialization handshake."""
        return self._send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "pytest-simulator", "version": "1.0.0"},
            },
        )

    def list_tools(self) -> list:
        """Discover available tools (tools/list)."""
        response = self._send_request("tools/list")
        return response.get("result", {}).get("tools", [])

    def call_tool(self, name: str, arguments: dict = None) -> dict:
        """Call an MCP tool (tools/call)."""
        return self._send_request("tools/call", {"name": name, "arguments": arguments or {}})


class TestStdioProtocol:
    """
    Test the full MCP protocol over stdio.
    These tests spawn the real server and communicate via pipes.
    """

    @pytest.fixture
    def mcp_client(self):
        """Create and start an MCP client connected to real server."""
        client = StdioMCPClient()
        client.start()
        yield client
        client.stop()

    def test_initialization_handshake(self, mcp_client):
        """Test MCP initialization protocol."""
        response = mcp_client.initialize()

        assert "result" in response
        assert "protocolVersion" in response["result"]
        assert "serverInfo" in response["result"]

    def test_tool_discovery(self, mcp_client):
        """Test tools/list - agent discovers available tools."""
        mcp_client.initialize()
        tools = mcp_client.list_tools()

        tool_names = [t["name"] for t in tools]
        assert "learn" in tool_names
        assert "get_sources" in tool_names
        assert "get_stats" in tool_names
        assert "list_supported_languages" in tool_names

    def test_call_tool_list_languages(self, mcp_client):
        """Test calling a simple tool via stdio."""
        mcp_client.initialize()
        response = mcp_client.call_tool("list_supported_languages")

        assert "result" in response
        content = response["result"]["content"][0]
        result_data = json.loads(content["text"])
        assert ".py" in result_data["extensions"]

    def test_full_agent_workflow(self, mcp_client, temp_project):
        """
        Simulate complete agent workflow over stdio:
        1. Initialize connection
        2. Discover tools
        3. Learn codebase
        4. Search for code
        5. Get stats
        """
        # Step 1: Initialize
        init_response = mcp_client.initialize()
        assert "error" not in init_response

        # Step 2: Discover tools
        tools = mcp_client.list_tools()
        assert len(tools) >= 4

        # Step 3: Learn codebase
        learn_response = mcp_client.call_tool("learn", {"path": str(temp_project)})
        learn_result = json.loads(learn_response["result"]["content"][0]["text"])
        assert learn_result["success"] is True

        # Step 4: Search for code
        search_response = mcp_client.call_tool(
            "get_sources", {"question": "calculator add", "n_results": 5}
        )
        search_result = json.loads(search_response["result"]["content"][0]["text"])
        assert "sources" in search_result

        # Step 5: Get stats
        stats_response = mcp_client.call_tool("get_stats")
        stats_result = json.loads(stats_response["result"]["content"][0]["text"])
        assert stats_result["loaded"] is True


class TestStdioEdgeCases:
    """Test edge cases in stdio communication."""

    @pytest.fixture
    def mcp_client(self):
        client = StdioMCPClient()
        client.start()
        yield client
        client.stop()

    def test_multiple_rapid_requests(self, mcp_client, temp_project):
        """Test rapid sequential requests (agent asking multiple questions)."""
        mcp_client.initialize()
        mcp_client.call_tool("learn", {"path": str(temp_project)})

        # Rapid fire search requests
        for i in range(5):
            response = mcp_client.call_tool(
                "get_sources", {"question": f"query {i}", "n_results": 2}
            )
            assert "result" in response
