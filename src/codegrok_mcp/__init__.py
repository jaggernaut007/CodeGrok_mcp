"""
CodeGrok MCP - Model Context Protocol server for codebase understanding.

A lightweight MCP server that provides code parsing and indexing capabilities
through the Model Context Protocol, enabling AI assistants to understand
and navigate codebases.

Usage:
    # As an MCP server
    python -m codegrok_mcp

    # Programmatic usage
    from codegrok_mcp import TreeSitterParser
    parser = TreeSitterParser()
    result = parser.parse_file("example.py")
"""

__version__ = "0.1.0"
__author__ = "CodeGrok Contributors"


# Lazy imports to avoid loading heavy dependencies at import time
def __getattr__(name: str):
    """Lazy import heavy modules only when accessed."""
    if name == "SourceRetriever":
        from codegrok_mcp.indexing.source_retriever import SourceRetriever

        return SourceRetriever
    elif name == "TreeSitterParser":
        from codegrok_mcp.parsers.treesitter_parser import TreeSitterParser

        return TreeSitterParser
    elif name == "ThreadLocalParserFactory":
        from codegrok_mcp.parsers.treesitter_parser import ThreadLocalParserFactory

        return ThreadLocalParserFactory
    elif name == "Symbol":
        from codegrok_mcp.core.models import Symbol

        return Symbol
    elif name == "SymbolType":
        from codegrok_mcp.core.models import SymbolType

        return SymbolType
    elif name == "ParsedFile":
        from codegrok_mcp.core.models import ParsedFile

        return ParsedFile
    elif name == "CodebaseIndex":
        from codegrok_mcp.core.models import CodebaseIndex

        return CodebaseIndex
    elif name == "IParser":
        from codegrok_mcp.core.interfaces import IParser

        return IParser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "__author__",
    "SourceRetriever",
    "TreeSitterParser",
    "ThreadLocalParserFactory",
    "Symbol",
    "SymbolType",
    "ParsedFile",
    "CodebaseIndex",
    "IParser",
]
