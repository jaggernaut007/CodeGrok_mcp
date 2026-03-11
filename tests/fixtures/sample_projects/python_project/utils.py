"""Utility functions for common operations."""

import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional

CONSTANT_VALUE = 42
DEFAULT_ENCODING = "utf-8"


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    with open(path, encoding=DEFAULT_ENCODING) as f:
        return json.load(f)


def save_config(path: str, config: Dict[str, Any]) -> None:
    """Save configuration to a JSON file."""
    with open(path, "w", encoding=DEFAULT_ENCODING) as f:
        json.dump(config, f, indent=2)


def compute_hash(content: str) -> str:
    """Compute SHA256 hash of content."""
    return hashlib.sha256(content.encode()).hexdigest()


def ensure_directory(path: Path) -> Path:
    """Ensure directory exists, create if necessary."""
    path.mkdir(parents=True, exist_ok=True)
    return path


class FileManager:
    """Manage file operations with caching."""

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self._cache: Dict[str, str] = {}

    def read(self, filename: str) -> Optional[str]:
        """Read file contents with caching."""
        if filename in self._cache:
            return self._cache[filename]

        filepath = self.base_path / filename
        if not filepath.exists():
            return None

        content = filepath.read_text(encoding=DEFAULT_ENCODING)
        self._cache[filename] = content
        return content

    def write(self, filename: str, content: str) -> None:
        """Write content to file and update cache."""
        filepath = self.base_path / filename
        filepath.write_text(content, encoding=DEFAULT_ENCODING)
        self._cache[filename] = content
