"""
CodeGrok MCP Indexing Module.

This module provides semantic search capabilities for code:
- EmbeddingService: Native embedding generation using SentenceTransformers
- SourceRetriever: Semantic code search with ChromaDB storage
- parallel_parse_files: Parallel file parsing for faster indexing

Usage:
    from codegrok_mcp.indexing import SourceRetriever, get_embedding_service

    # Create a retriever for a codebase
    retriever = SourceRetriever(
        codebase_path="/path/to/code",
        persist_path="/path/to/code/.codegrok/chroma"
    )

    # Index the codebase (one-time operation)
    retriever.index_codebase()

    # Query for relevant code
    documents, sources = retriever.get_sources_for_question("How does auth work?")
"""

from codegrok_mcp.indexing.embedding_service import (
    EmbeddingService,
    get_embedding_service,
    reset_embedding_service,
    ChromaDBEmbeddingFunction,
    embed,
    EMBEDDING_MODELS,
    DEFAULT_MODEL,
)

from codegrok_mcp.indexing.source_retriever import (
    SourceRetriever,
    CodeChunk,
    SUPPORTED_EXTENSIONS,
    count_codebase_files,
)

from codegrok_mcp.indexing.parallel_indexer import (
    parallel_parse_files,
    ParseResult,
    ParallelProgress,
)

__all__ = [
    # Embedding service
    "EmbeddingService",
    "get_embedding_service",
    "reset_embedding_service",
    "ChromaDBEmbeddingFunction",
    "embed",
    "EMBEDDING_MODELS",
    "DEFAULT_MODEL",
    # Source retriever
    "SourceRetriever",
    "CodeChunk",
    "SUPPORTED_EXTENSIONS",
    "count_codebase_files",
    # Parallel indexing
    "parallel_parse_files",
    "ParseResult",
    "ParallelProgress",
]
