import pytest
import tempfile
from pathlib import Path
from codegrok_mcp.mcp.state import reset_state, get_state


@pytest.fixture(autouse=True)
def clean_state():
    """Reset MCP state before and after each test."""
    reset_state()
    yield
    reset_state()


@pytest.fixture
def temp_project(tmp_path):
    """Create a minimal Python project for testing."""
    main_py = tmp_path / "main.py"
    main_py.write_text('''
def hello(name: str) -> str:
    """Say hello to someone."""
    return f"Hello, {name}!"

class Calculator:
    """A simple calculator."""

    def add(self, a: int, b: int) -> int:
        return a + b

    def multiply(self, a: int, b: int) -> int:
        return a * b
''')

    utils_py = tmp_path / "utils.py"
    utils_py.write_text('''
import json

def load_config(path: str) -> dict:
    """Load configuration from JSON file."""
    with open(path) as f:
        return json.load(f)

CONSTANT_VALUE = 42
''')

    return tmp_path


@pytest.fixture
def multi_lang_project(tmp_path):
    """Create a multi-language project."""
    (tmp_path / "app.py").write_text("def main(): pass")
    (tmp_path / "helper.js").write_text("function helper() { return 1; }")
    (tmp_path / "server.go").write_text("package main\n\nfunc main() {}")
    return tmp_path


@pytest.fixture
def python_project_fixture():
    """Return path to the static Python project fixture."""
    return Path(__file__).parent / "fixtures" / "sample_projects" / "python_project"


@pytest.fixture
def multi_lang_fixture():
    """Return path to the static multi-language fixture."""
    return Path(__file__).parent / "fixtures" / "sample_projects" / "multi_lang"
