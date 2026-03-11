"""Session state management for MCP server."""

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from codegrok_mcp.indexing.source_retriever import SourceRetriever
    from codegrok_mcp.indexing.memory_retriever import MemoryRetriever


@dataclass
class IndexingStatus:
    """Thread-safe status of a background indexing operation."""

    active: bool = False
    progress: int = 0  # 0-100
    message: str = ""
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def start(self, message: str = "Starting indexing..."):
        with self._lock:
            self.active = True
            self.progress = 0
            self.message = message
            self.error = None
            self.result = None

    def update(self, progress: int, message: str):
        with self._lock:
            self.progress = min(progress, 99)
            self.message = message

    def complete(self, result: Dict[str, Any]):
        with self._lock:
            self.active = False
            self.progress = 100
            self.message = "Indexing complete"
            self.result = result

    def fail(self, error: str):
        with self._lock:
            self.active = False
            self.message = f"Indexing failed: {error}"
            self.error = error

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "active": self.active,
                "progress": self.progress,
                "message": self.message,
                "error": self.error,
            }


@dataclass
class MCPSessionState:
    """Singleton state for MCP server session."""

    retriever: Optional["SourceRetriever"] = None
    memory_retriever: Optional["MemoryRetriever"] = None
    codebase_path: Optional[Path] = None
    indexing: IndexingStatus = field(default_factory=IndexingStatus)

    @property
    def is_loaded(self) -> bool:
        """Check if a codebase is currently loaded."""
        return self.retriever is not None

    @property
    def has_memories(self) -> bool:
        """Check if memory retriever is initialized."""
        return self.memory_retriever is not None


_state: Optional[MCPSessionState] = None


def get_state() -> MCPSessionState:
    """Get or create the singleton state instance."""
    global _state
    if _state is None:
        _state = MCPSessionState()
    return _state


def reset_state() -> None:
    """Reset the singleton state (useful for testing)."""
    global _state
    _state = None
