"""
Unit tests for MCP timeout configuration.
"""

import os
import pytest
from unittest.mock import patch

from codegrok_mcp.mcp.server import _get_timeout, DEFAULT_TIMEOUT_SECONDS


class TestGetTimeout:
    """Test timeout resolution logic."""

    def test_default_timeout(self):
        """Returns DEFAULT_TIMEOUT_SECONDS when no override or env var."""
        with patch.dict(os.environ, {}, clear=True):
            assert _get_timeout() == DEFAULT_TIMEOUT_SECONDS

    def test_per_call_override(self):
        """Per-call override takes highest priority."""
        with patch.dict(os.environ, {"CODEGROK_TIMEOUT": "999"}):
            assert _get_timeout(override=120) == 120

    def test_env_var_override(self):
        """CODEGROK_TIMEOUT env var overrides default."""
        with patch.dict(os.environ, {"CODEGROK_TIMEOUT": "1200"}):
            assert _get_timeout() == 1200

    def test_env_var_invalid_ignored(self):
        """Invalid CODEGROK_TIMEOUT falls back to default."""
        with patch.dict(os.environ, {"CODEGROK_TIMEOUT": "not_a_number"}):
            assert _get_timeout() == DEFAULT_TIMEOUT_SECONDS

    def test_env_var_zero_ignored(self):
        """Zero CODEGROK_TIMEOUT falls back to default."""
        with patch.dict(os.environ, {"CODEGROK_TIMEOUT": "0"}):
            assert _get_timeout() == DEFAULT_TIMEOUT_SECONDS

    def test_env_var_negative_ignored(self):
        """Negative CODEGROK_TIMEOUT falls back to default."""
        with patch.dict(os.environ, {"CODEGROK_TIMEOUT": "-5"}):
            assert _get_timeout() == DEFAULT_TIMEOUT_SECONDS

    def test_override_zero_uses_env(self):
        """Override of 0 or None falls through to env var."""
        with patch.dict(os.environ, {"CODEGROK_TIMEOUT": "300"}):
            assert _get_timeout(override=0) == 300
            assert _get_timeout(override=None) == 300

    def test_override_negative_uses_env(self):
        """Negative override falls through to env var."""
        with patch.dict(os.environ, {"CODEGROK_TIMEOUT": "300"}):
            assert _get_timeout(override=-1) == 300

    def test_default_is_600(self):
        """Default timeout is 600 seconds (10 minutes)."""
        assert DEFAULT_TIMEOUT_SECONDS == 600
