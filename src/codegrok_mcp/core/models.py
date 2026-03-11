"""
Core data models for CodeGrok MCP.

This module defines the fundamental data structures used throughout the system
for representing code symbols, parsed files, and codebase indices.

All models are designed for:
- Immutability (frozen dataclasses where appropriate)
- Serialization (JSON-compatible via to_dict/from_dict)
- Performance (efficient for 100K+ symbols)
- Type safety (comprehensive type hints)
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone


class SymbolType(Enum):
    """
    Enumeration of code symbol types that can be parsed and indexed.

    Attributes:
        FUNCTION: Standalone function definition
        CLASS: Class definition
        METHOD: Method within a class
        VARIABLE: Module-level or class-level variable
    """

    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    VARIABLE = "variable"

    def __str__(self) -> str:
        """String representation for serialization."""
        return self.value

    @classmethod
    def from_string(cls, value: str) -> "SymbolType":
        """
        Create SymbolType from string value.

        Args:
            value: String representation of symbol type

        Returns:
            SymbolType enum member

        Raises:
            ValueError: If value doesn't match any SymbolType
        """
        for member in cls:
            if member.value == value.lower():
                return member
        raise ValueError(f"Invalid SymbolType: {value}")


@dataclass(frozen=True)
class Symbol:
    """
    Represents a single code symbol (function, class, method, or variable).

    This is the most critical model in the system - every parsed code element
    is represented as a Symbol. Designed for immutability and efficient storage.

    Attributes:
        name: The identifier name of the symbol (e.g., "login_user", "UserClass")
        type: The kind of symbol (FUNCTION, CLASS, METHOD, VARIABLE)
        filepath: Absolute path to the file containing this symbol
        line_start: Starting line number (1-indexed, inclusive)
        line_end: Ending line number (1-indexed, inclusive)
        language: Programming language (python, javascript, cpp, bash, go, etc.)
        signature: Full signature/declaration (e.g., "def login_user(username: str) -> bool:")
        docstring: Extracted documentation string (empty if none)
        parent: Name of parent class (for methods) or None for top-level symbols
        code_snippet: First 500 characters of the symbol's code for preview
        imports: List of import statements used within this symbol's scope
        calls: List of function/method names called by this symbol
        metadata: Extensible dictionary for additional language-specific or custom data
    """

    name: str
    type: SymbolType
    filepath: str
    line_start: int
    line_end: int
    language: str
    signature: str
    docstring: str = ""
    parent: Optional[str] = None
    code_snippet: str = ""
    imports: List[str] = field(default_factory=list)
    calls: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """
        Validate symbol data after initialization.

        Raises:
            ValueError: If any validation constraint is violated
        """
        if not self.name:
            raise ValueError("Symbol name cannot be empty")
        if not self.filepath:
            raise ValueError("Symbol filepath cannot be empty")
        if self.line_start < 1:
            raise ValueError(f"line_start must be >= 1, got {self.line_start}")
        if self.line_end < self.line_start:
            raise ValueError(
                f"line_end ({self.line_end}) must be >= line_start ({self.line_start})"
            )
        if not self.language:
            raise ValueError("Symbol language cannot be empty")
        if not isinstance(self.type, SymbolType):
            raise ValueError(f"type must be SymbolType enum, got {type(self.type)}")

    @property
    def qualified_name(self) -> str:
        """
        Returns fully qualified name including parent class if applicable.

        Returns:
            Qualified name (e.g., "MyClass.my_method" or "standalone_function")
        """
        if self.parent:
            return f"{self.parent}.{self.name}"
        return self.name

    @property
    def line_count(self) -> int:
        """
        Returns number of lines in this symbol.

        Returns:
            Number of lines (inclusive)
        """
        return self.line_end - self.line_start + 1

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Symbol to dictionary for serialization.

        Returns:
            Dictionary representation with all fields
        """
        data = asdict(self)
        data["type"] = self.type.value  # Convert enum to string
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Symbol":
        """
        Create Symbol from dictionary.

        Args:
            data: Dictionary containing symbol data

        Returns:
            Symbol instance

        Raises:
            ValueError: If required fields are missing or invalid
        """
        # Convert type string back to enum
        if "type" in data and isinstance(data["type"], str):
            data = data.copy()  # Don't modify original
            data["type"] = SymbolType.from_string(data["type"])
        return cls(**data)


@dataclass(frozen=True)
class ParsedFile:
    """
    Represents a single parsed source code file.

    Contains all symbols extracted from the file plus metadata about the
    parsing operation. Immutable to ensure consistency across the system.

    Attributes:
        filepath: Absolute path to the source file
        language: Programming language detected
        symbols: List of all symbols found in this file
        imports: File-level import statements
        parse_time: Time taken to parse this file (seconds)
        error: None if parsing succeeded, error message if failed
    """

    filepath: str
    language: str
    symbols: List[Symbol] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    parse_time: float = 0.0
    error: Optional[str] = None

    def __post_init__(self):
        """
        Validate parsed file data.

        Raises:
            ValueError: If validation fails
        """
        if not self.filepath:
            raise ValueError("ParsedFile filepath cannot be empty")
        if not self.language:
            raise ValueError("ParsedFile language cannot be empty")
        if self.parse_time < 0:
            raise ValueError(f"parse_time must be >= 0, got {self.parse_time}")

    @property
    def is_successful(self) -> bool:
        """
        Check if file was parsed successfully.

        Returns:
            True if no error occurred, False otherwise
        """
        return self.error is None

    @property
    def symbol_count(self) -> int:
        """
        Get number of symbols in this file.

        Returns:
            Count of symbols
        """
        return len(self.symbols)

    def get_symbols_by_type(self, symbol_type: SymbolType) -> List[Symbol]:
        """
        Filter symbols by type.

        Args:
            symbol_type: Type of symbols to retrieve

        Returns:
            List of symbols matching the type
        """
        return [s for s in self.symbols if s.type == symbol_type]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert ParsedFile to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "filepath": self.filepath,
            "language": self.language,
            "symbols": [s.to_dict() for s in self.symbols],
            "imports": self.imports,
            "parse_time": self.parse_time,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParsedFile":
        """
        Create ParsedFile from dictionary.

        Args:
            data: Dictionary containing parsed file data

        Returns:
            ParsedFile instance
        """
        data = data.copy()
        if "symbols" in data:
            data["symbols"] = [Symbol.from_dict(s) for s in data["symbols"]]
        return cls(**data)


@dataclass(frozen=True)
class CodebaseIndex:
    """
    Represents a complete index of a codebase.

    This is the top-level data structure containing all parsed files and
    aggregated statistics. Designed for efficient serialization and lookup.

    Attributes:
        root_path: Absolute path to the codebase root directory
        files: Mapping from filepath to ParsedFile
        total_files: Total number of files indexed
        total_symbols: Total number of symbols across all files
        indexed_at: ISO 8601 timestamp of when indexing completed
    """

    root_path: str
    files: Dict[str, ParsedFile] = field(default_factory=dict)
    total_files: int = 0
    total_symbols: int = 0
    indexed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        """
        Validate codebase index data.

        Raises:
            ValueError: If validation fails
        """
        if not self.root_path:
            raise ValueError("CodebaseIndex root_path cannot be empty")
        if self.total_files < 0:
            raise ValueError(f"total_files must be >= 0, got {self.total_files}")
        if self.total_symbols < 0:
            raise ValueError(f"total_symbols must be >= 0, got {self.total_symbols}")

    @property
    def successful_parses(self) -> int:
        """
        Count files that were parsed successfully.

        Returns:
            Number of files with no errors
        """
        return sum(1 for f in self.files.values() if f.is_successful)

    @property
    def failed_parses(self) -> int:
        """
        Count files that failed to parse.

        Returns:
            Number of files with errors
        """
        return sum(1 for f in self.files.values() if not f.is_successful)

    def get_symbols_by_name(self, name: str) -> List[Symbol]:
        """
        Find all symbols with a given name across the codebase.

        Args:
            name: Symbol name to search for

        Returns:
            List of matching symbols
        """
        results = []
        for parsed_file in self.files.values():
            results.extend(s for s in parsed_file.symbols if s.name == name)
        return results

    def get_symbols_by_type(self, symbol_type: SymbolType) -> List[Symbol]:
        """
        Find all symbols of a given type across the codebase.

        Args:
            symbol_type: Type of symbols to retrieve

        Returns:
            List of matching symbols
        """
        results = []
        for parsed_file in self.files.values():
            results.extend(parsed_file.get_symbols_by_type(symbol_type))
        return results

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert CodebaseIndex to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "root_path": self.root_path,
            "files": {path: f.to_dict() for path, f in self.files.items()},
            "total_files": self.total_files,
            "total_symbols": self.total_symbols,
            "indexed_at": self.indexed_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodebaseIndex":
        """
        Create CodebaseIndex from dictionary.

        Args:
            data: Dictionary containing index data

        Returns:
            CodebaseIndex instance
        """
        data = data.copy()
        if "files" in data:
            data["files"] = {path: ParsedFile.from_dict(f) for path, f in data["files"].items()}
        return cls(**data)


# =============================================================================
# Memory Models for Universal Memory Layer
# =============================================================================


class MemoryType(Enum):
    """
    Enumeration of memory types that can be stored and retrieved.

    Attributes:
        CONVERSATION: Chat history, Q&A exchanges, discussion summaries
        STATUS: Project status updates, blockers, progress notes
        DECISION: Architecture decisions, technical choices, rationale
        PREFERENCE: User preferences, coding style, tool settings
        DOC: Documentation snippets, README content, API docs
        NOTE: General notes, reminders, TODOs
    """

    CONVERSATION = "conversation"
    STATUS = "status"
    DECISION = "decision"
    PREFERENCE = "preference"
    DOC = "doc"
    NOTE = "note"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_string(cls, value: str) -> "MemoryType":
        for member in cls:
            if member.value == value.lower():
                return member
        raise ValueError(f"Invalid MemoryType: {value}")


@dataclass
class Memory:
    """
    Represents a single memory entry (conversation, status, decision, etc.).

    Unlike Symbol (which is frozen/immutable), Memory is mutable because
    we may update accessed_at timestamps for LRU tracking.

    Attributes:
        id: Unique identifier (UUID)
        content: The actual memory content (text)
        memory_type: Type of memory (CONVERSATION, STATUS, etc.)
        project: Project identifier (typically codebase path)
        tags: List of tags for filtering and organization
        created_at: ISO 8601 timestamp when memory was created
        accessed_at: ISO 8601 timestamp when memory was last accessed
        ttl: Time-to-live policy ("session", "day", "week", "permanent")
        source: Origin of memory ("user", "agent", "auto", "import")
        metadata: Extensible dictionary for additional data
    """

    id: str
    content: str
    memory_type: MemoryType
    project: str
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    accessed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ttl: str = "permanent"
    source: str = "user"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate memory data after initialization."""
        if not self.id:
            raise ValueError("Memory id cannot be empty")
        if not self.content:
            raise ValueError("Memory content cannot be empty")
        if not self.project:
            raise ValueError("Memory project cannot be empty")
        if not isinstance(self.memory_type, MemoryType):
            raise ValueError(f"memory_type must be MemoryType enum, got {type(self.memory_type)}")
        valid_ttls = ("session", "day", "week", "month", "permanent")
        if self.ttl not in valid_ttls:
            raise ValueError(f"ttl must be one of {valid_ttls}, got {self.ttl}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert Memory to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "project": self.project,
            "tags": self.tags,
            "created_at": self.created_at,
            "accessed_at": self.accessed_at,
            "ttl": self.ttl,
            "source": self.source,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        """Create Memory from dictionary."""
        data = data.copy()
        if "memory_type" in data and isinstance(data["memory_type"], str):
            data["memory_type"] = MemoryType.from_string(data["memory_type"])
        return cls(**data)

    def touch(self) -> None:
        """Update accessed_at timestamp (for LRU tracking)."""
        self.accessed_at = datetime.now(timezone.utc).isoformat()
