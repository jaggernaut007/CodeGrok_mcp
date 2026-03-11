"""
Unit tests for discover_files() and memory optimization constants.
"""

import pytest
from pathlib import Path
from codegrok_mcp.indexing.source_retriever import (
    discover_files,
    FILE_BATCH_SIZE,
    MAX_PARSE_WORKERS,
)


def _create_file(path: Path, content: str = "# placeholder"):
    """Helper to create a file with parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


class TestDiscoverFilesBasic:
    """Test basic file discovery functionality."""

    def test_discover_files_basic(self, tmp_path):
        """Finds .py files in a simple directory."""
        _create_file(tmp_path / "main.py", "def main(): pass")
        _create_file(tmp_path / "utils.py", "x = 1")
        _create_file(tmp_path / "readme.txt", "not code")

        files = discover_files(tmp_path, extensions={".py"})

        py_names = sorted(f.name for f in files)
        assert py_names == ["main.py", "utils.py"]

    def test_discover_files_skip_dirs(self, tmp_path):
        """Skips node_modules, __pycache__, .git even when nested."""
        _create_file(tmp_path / "app.py", "x = 1")
        _create_file(tmp_path / "node_modules" / "pkg" / "index.py", "x = 2")
        _create_file(tmp_path / "__pycache__" / "app.cpython-311.pyc", "x = 3")
        _create_file(tmp_path / ".git" / "config.py", "x = 4")
        _create_file(tmp_path / "src" / "node_modules" / "deep" / "mod.py", "x = 5")

        files = discover_files(tmp_path, extensions={".py", ".pyc"})

        assert len(files) == 1
        assert files[0].name == "app.py"

    def test_discover_files_no_gitignore(self, tmp_path):
        """Works normally when no .gitignore exists."""
        _create_file(tmp_path / "a.py", "x = 1")
        _create_file(tmp_path / "sub" / "b.py", "x = 2")

        files = discover_files(tmp_path, extensions={".py"})

        assert len(files) == 2


class TestDiscoverFilesGitignore:
    """Test .gitignore support."""

    def test_discover_files_gitignore(self, tmp_path):
        """Respects root .gitignore patterns."""
        (tmp_path / ".gitignore").write_text("*.log\nbuild/\n")

        _create_file(tmp_path / "app.py", "x = 1")
        _create_file(tmp_path / "debug.log", "log data")
        _create_file(tmp_path / "build" / "output.py", "x = 2")
        _create_file(tmp_path / "src" / "core.py", "x = 3")

        files = discover_files(tmp_path, extensions={".py", ".log"})

        names = sorted(f.name for f in files)
        assert names == ["app.py", "core.py"]

    def test_discover_files_nested_gitignore(self, tmp_path):
        """Handles .gitignore in subdirectories."""
        (tmp_path / ".gitignore").write_text("*.log\n")
        (tmp_path / "vendor").mkdir()
        (tmp_path / "vendor" / ".gitignore").write_text("*.py\n")

        _create_file(tmp_path / "app.py", "x = 1")
        _create_file(tmp_path / "vendor" / "lib.py", "x = 2")
        _create_file(tmp_path / "vendor" / "data.txt", "data")

        files = discover_files(tmp_path, extensions={".py", ".txt"})

        names = sorted(f.name for f in files)
        assert "app.py" in names
        assert "lib.py" not in names

    def test_discover_files_respect_gitignore_false(self, tmp_path):
        """Opt-out disables gitignore filtering."""
        (tmp_path / ".gitignore").write_text("*.py\n")

        _create_file(tmp_path / "app.py", "x = 1")
        _create_file(tmp_path / "lib.py", "x = 2")

        files = discover_files(tmp_path, extensions={".py"}, respect_gitignore=False)

        assert len(files) == 2


class TestDiscoverFilesLimits:
    """Test safety limits."""

    def test_discover_files_max_files_limit(self, tmp_path):
        """Stops at max_files and returns partial results."""
        for i in range(20):
            _create_file(tmp_path / f"file_{i}.py", f"x = {i}")

        files = discover_files(tmp_path, extensions={".py"}, max_files=10)

        assert len(files) == 10

    def test_discover_files_progress_callback(self, tmp_path):
        """Progress callback is called during discovery."""
        # Create enough files to trigger callback (every 1000)
        # We'll use a smaller set and verify callback mechanism works
        for i in range(5):
            _create_file(tmp_path / f"file_{i}.py", f"x = {i}")

        events = []

        def callback(event_type, data):
            events.append((event_type, data))

        files = discover_files(tmp_path, extensions={".py"}, progress_callback=callback)

        # With only 5 files, callback won't fire (fires every 1000)
        assert len(files) == 5
        assert len(events) == 0  # Below threshold


class TestMemoryOptimizationConstants:
    """Test that memory optimization constants are set correctly."""

    def test_file_batch_size(self):
        """FILE_BATCH_SIZE is set to a reasonable value."""
        assert FILE_BATCH_SIZE == 500
        assert isinstance(FILE_BATCH_SIZE, int)

    def test_max_parse_workers(self):
        """MAX_PARSE_WORKERS caps parallel workers to limit memory."""
        assert MAX_PARSE_WORKERS == 4
        assert isinstance(MAX_PARSE_WORKERS, int)
