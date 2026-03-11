"""Custom exceptions for CodeGrok MCP.

This module defines a hierarchy of exceptions for better error handling
and debugging throughout the CodeGrok MCP system.

Usage:
    from codegrok_mcp.core.exceptions import ParseError, IndexingError

    try:
        parser.parse_file("bad_file.py")
    except ParseError as e:
        print(f"Failed to parse {e.filepath}: {e.details}")
"""


class CodeGrokException(Exception):
    """Base exception for all CodeGrok operations.

    All custom exceptions in CodeGrok inherit from this class,
    allowing for broad exception catching when needed.
    """

    pass


class ParseError(CodeGrokException):
    """Raised when file parsing fails.

    Attributes:
        filepath: Path to the file that failed to parse
        language: Detected language of the file
        details: Specific error details
    """

    def __init__(self, filepath: str, language: str, details: str):
        self.filepath = filepath
        self.language = language
        self.details = details
        super().__init__(f"Failed to parse {filepath} ({language}): {details}")


class IndexingError(CodeGrokException):
    """Raised when indexing operation fails.

    This covers failures in the indexing pipeline including:
    - File discovery issues
    - Chunking failures
    - ChromaDB storage errors
    """

    pass


class EmbeddingError(CodeGrokException):
    """Raised when embedding generation fails.

    This covers failures in the embedding service including:
    - Model loading failures
    - Encoding errors
    - Memory issues
    """

    pass


class SearchError(CodeGrokException):
    """Raised when semantic search fails.

    This covers failures in the search operation including:
    - ChromaDB query errors
    - Missing index
    - Invalid query parameters
    """

    pass


class ConfigurationError(CodeGrokException):
    """Raised when configuration is invalid.

    This covers failures related to:
    - Invalid model names
    - Invalid file paths
    - Missing required parameters
    """

    pass
