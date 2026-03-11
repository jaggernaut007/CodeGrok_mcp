"""
Source retriever for semantic code search.

This module provides semantic search over code using embeddings with persistent
storage. It handles indexing, storage, and retrieval of code chunks for RAG
(Retrieval-Augmented Generation) applications.

Note: This module is LLM-agnostic. It provides source retrieval only.
LLM integration is handled by the MCP layer.

Usage:
    from codegrok_mcp.indexing.source_retriever import SourceRetriever

    # Learn mode (with persistence)
    retriever = SourceRetriever(
        codebase_path="/path/to/code",
        persist_path="/path/to/code/.codegrok/chroma"
    )
    retriever.index_codebase()
    retriever.save_metadata("/path/to/code/.codegrok/metadata.json")

    # Retrieval mode (load existing)
    retriever = SourceRetriever(
        codebase_path="/path/to/code",
        persist_path="/path/to/code/.codegrok/chroma"
    )
    retriever.load_existing_index()
    sources = retriever.get_sources_for_question("How does authentication work?")
"""

import gc
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass

import pathspec

try:
    import chromadb
except ImportError:
    raise ImportError("chromadb is required. Install with: pip install chromadb")

# Import native embedding service
from codegrok_mcp.indexing.embedding_service import get_embedding_service, EmbeddingService

from codegrok_mcp.parsers.treesitter_parser import TreeSitterParser
from codegrok_mcp.parsers.language_configs import get_supported_extensions, get_language_for_file
from codegrok_mcp.core.models import Symbol, SymbolType

# Derived from authoritative EXTENSION_MAP in language_configs.py (30+ extensions, 9 languages)
# This eliminates duplication and ensures extensions stay in sync
SUPPORTED_EXTENSIONS = list(get_supported_extensions())
SUPPORTED_EXTENSIONS_SET = set(SUPPORTED_EXTENSIONS)  # For O(1) lookup

# Common directories to skip during file discovery
SKIP_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".codegrok",
    "venv",
    ".venv",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    ".eggs",
}

# Memory optimization: process files in batches to limit peak memory usage.
# Each batch's symbols are converted to chunks then freed before the next batch.
FILE_BATCH_SIZE = 500

# Cap parallel parse workers to limit memory (each worker holds a tree-sitter parser).
# Lower than the 32 max in parallel_indexer.py to prevent OOM on large codebases.
MAX_PARSE_WORKERS = 4


def _load_gitignore(directory: Path) -> Optional[pathspec.PathSpec]:
    """Load a .gitignore file from a directory, returning a PathSpec or None."""
    gitignore_path = directory / ".gitignore"
    if gitignore_path.is_file():
        try:
            with open(gitignore_path, "r", encoding="utf-8", errors="ignore") as f:
                return pathspec.PathSpec.from_lines("gitignore", f)
        except (OSError, IOError):
            pass
    return None


def discover_files(
    codebase_path: Path,
    extensions: set = None,
    respect_gitignore: bool = True,
    max_files: int = 200_000,
    progress_callback: Callable = None,
) -> List[Path]:
    """Single-pass file discovery with extension filtering and .gitignore support.

    Uses os.walk() to traverse the directory tree once, pruning ignored
    directories in-place so they are never descended into. Respects .gitignore
    patterns (including nested .gitignore files) and a hardcoded SKIP_DIRS set.

    Args:
        codebase_path: Path to the codebase root directory.
        extensions: Set of extensions to include (default: SUPPORTED_EXTENSIONS_SET).
        respect_gitignore: Whether to respect .gitignore patterns (default: True).
        max_files: Safety limit on number of files to discover (default: 200,000).
        progress_callback: Optional callback(event_type, data) for progress events.

    Returns:
        List of file paths matching the extensions.
    """
    if extensions is None:
        extensions = SUPPORTED_EXTENSIONS_SET

    # Load root .gitignore
    gitignore_specs: List[pathspec.PathSpec] = []
    if respect_gitignore:
        root_spec = _load_gitignore(codebase_path)
        if root_spec is not None:
            gitignore_specs.append(root_spec)

    files: List[Path] = []
    codebase_str = str(codebase_path)

    for dirpath_str, dirnames, filenames in os.walk(codebase_path, followlinks=False):
        dirpath = Path(dirpath_str)

        # Load nested .gitignore for this directory (not root)
        if respect_gitignore and dirpath != codebase_path:
            nested_spec = _load_gitignore(dirpath)
            if nested_spec is not None:
                gitignore_specs.append(nested_spec)

        # Prune directories in-place: remove SKIP_DIRS and gitignored dirs
        filtered_dirs = []
        for d in dirnames:
            if d in SKIP_DIRS:
                continue
            if respect_gitignore and gitignore_specs:
                # Compute relative path with trailing slash for directory matching
                rel_dir = str((dirpath / d).relative_to(codebase_path)) + "/"
                if any(spec.match_file(rel_dir) for spec in gitignore_specs):
                    continue
            filtered_dirs.append(d)
        dirnames[:] = filtered_dirs

        # Check files
        for filename in filenames:
            filepath = dirpath / filename
            if filepath.suffix not in extensions:
                continue
            if respect_gitignore and gitignore_specs:
                rel_path = str(filepath.relative_to(codebase_path))
                if any(spec.match_file(rel_path) for spec in gitignore_specs):
                    continue
            files.append(filepath)

            # Report progress every 1000 files
            if progress_callback and len(files) % 1000 == 0:
                progress_callback("discovery_progress", {"files_found": len(files)})

            # Safety limit
            if len(files) >= max_files:
                logging.warning(
                    f"discover_files: reached max_files limit ({max_files}). "
                    f"Stopping discovery. Consider using .gitignore to exclude files."
                )
                return files

    return files


def _save_checkpoint(checkpoint_path: Path, chunks_completed: int, total_chunks: int) -> None:
    """Atomically save a checkpoint file for resumable indexing."""
    data = {
        "chunks_completed": chunks_completed,
        "total_chunks": total_chunks,
        "timestamp": datetime.now().isoformat(),
    }
    tmp_path = checkpoint_path.with_suffix(".tmp")
    try:
        with open(tmp_path, "w") as f:
            json.dump(data, f)
        os.replace(str(tmp_path), str(checkpoint_path))
    except OSError:
        # Best-effort: if we can't save checkpoint, indexing still continues
        pass


def _load_checkpoint(checkpoint_path: Path) -> Optional[Dict[str, Any]]:
    """Load a checkpoint file, returning the data dict or None."""
    if checkpoint_path is None or not checkpoint_path.exists():
        return None
    try:
        with open(checkpoint_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def count_codebase_files(codebase_path: Path) -> int:
    """Quick file count for ETA estimation.

    Args:
        codebase_path: Path to the codebase root directory.

    Returns:
        Total count of files with supported extensions.
    """
    return len(discover_files(codebase_path))


@dataclass
class CodeChunk:
    """A chunk of code suitable for embedding."""

    id: str
    text: str
    filepath: str
    symbol_name: str
    symbol_type: str
    line_start: int
    metadata: Dict[str, Any]


class SourceRetriever:
    """
    Source retriever for semantic code search.

    This class handles:
    - Phase 1: Parsing (using TreeSitterParser)
    - Phase 2: Storage (ChromaDB persistent/in-memory)
    - Phase 4: Semantic search (embeddings)

    Note: LLM integration is NOT handled here. This is a retrieval-only component.
    """

    def __init__(
        self,
        codebase_path: str,
        embedding_model: str = "coderankembed",  # SOTA for code retrieval
        collection_name: str = "codebase",
        verbose: bool = True,
        persist_path: Optional[str] = None,
        # Parallel indexing options (3-5x faster for large codebases)
        parallel: bool = True,  # Enabled by default for better performance
        max_workers: Optional[int] = None,
        # Dependency injection for testability
        parser: Optional["TreeSitterParser"] = None,
        embedding_service: Optional["EmbeddingService"] = None,
    ):
        """
        Initialize the source retriever.

        Args:
            codebase_path: Path to codebase root directory
            embedding_model: Embedding model name (native or Ollama)
            collection_name: ChromaDB collection name
            verbose: Print progress messages
            persist_path: Path for ChromaDB persistent storage (None = in-memory)
            parallel: Enable parallel file parsing for faster indexing (3-5x speedup)
            max_workers: Number of parallel workers (default: CPU count - 1)
            parser: Optional parser instance for dependency injection (for testing)
            embedding_service: Optional embedding service for dependency injection (for testing)
        """
        self.codebase_path = Path(codebase_path)
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.verbose = verbose
        self.persist_path = persist_path

        self.parallel = parallel
        self.max_workers = max_workers

        # Initialize parser (use injected or create default)
        self.parser = parser or TreeSitterParser()

        # Initialize embedding service (use injected or create default)
        self._log(f"Using native embedding: {embedding_model}")
        self.embedding_service = embedding_service or get_embedding_service(
            embedding_model, show_progress=verbose  # Only show tqdm progress bar if verbose
        )

        # Initialize ChromaDB (persistent or in-memory)
        if persist_path:
            Path(persist_path).mkdir(parents=True, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=persist_path)
            self._log(f"Using persistent storage: {persist_path}")
        else:
            self.chroma_client = chromadb.Client()
            self._log("Using in-memory storage (data will be lost on exit)")

        self.collection = None

        # Statistics
        self.stats = {
            "total_files": 0,
            "total_symbols": 0,
            "total_chunks": 0,
            "parse_errors": 0,
            "indexing_time": 0.0,
        }

        # Metadata storage for incremental reindexing (file modification times)
        self._metadata: Dict[str, Any] = {}

    def _log(self, message: str):
        """Print message if verbose mode enabled."""
        if self.verbose:
            print(message)

    def _create_chunk_text(self, symbol: Symbol) -> str:
        """
        Create embedding text from a symbol.

        Format:
            # filepath:line
            type: name

            signature

            docstring

            code_snippet

            Imports: ...
            Calls: ...

        Args:
            symbol: Symbol to convert to chunk text

        Returns:
            Formatted text suitable for embedding
        """
        parts = [
            f"# {symbol.filepath}:{symbol.line_start}",
            f"{symbol.type.value}: {symbol.name}",
            "",
        ]

        if symbol.signature:
            parts.append(symbol.signature)
            parts.append("")

        if symbol.docstring:
            parts.append(symbol.docstring)
            parts.append("")

        if symbol.code_snippet:
            parts.append(symbol.code_snippet)
            parts.append("")

        if symbol.imports:
            imports_str = ", ".join(symbol.imports[:10])
            if len(symbol.imports) > 10:
                imports_str += f", ... ({len(symbol.imports) - 10} more)"
            parts.append(f"Imports: {imports_str}")

        if symbol.calls:
            calls_str = ", ".join(symbol.calls[:10])
            if len(symbol.calls) > 10:
                calls_str += f", ... ({len(symbol.calls) - 10} more)"
            parts.append(f"Calls: {calls_str}")

        return "\n".join(parts)

    def _create_chunk(self, symbol: Symbol) -> CodeChunk:
        """
        Create a CodeChunk from a Symbol.

        Args:
            symbol: Symbol to convert

        Returns:
            CodeChunk ready for embedding
        """
        chunk_id = f"{symbol.filepath}:{symbol.name}:{symbol.line_start}"
        chunk_text = self._create_chunk_text(symbol)

        # Get language from filepath for filtering support
        language = get_language_for_file(symbol.filepath) or "unknown"

        return CodeChunk(
            id=chunk_id,
            text=chunk_text,
            filepath=symbol.filepath,
            symbol_name=symbol.name,
            symbol_type=symbol.type.value,
            line_start=symbol.line_start,
            metadata={
                "filepath": symbol.filepath,
                "name": symbol.name,
                "type": symbol.type.value,
                "line": symbol.line_start,
                "signature": symbol.signature,
                "parent": symbol.parent or "",
                "language": language,  # NEW: enables language filtering in search
            },
        )

    def index_codebase(
        self,
        file_extensions: Optional[List[str]] = None,
        progress_callback: Optional[callable] = None,
    ):
        """
        Index the entire codebase.

        This is the one-time expensive operation that:
        1. Parses all files (Phase 1)
        2. Creates chunks (Phase 2)
        3. Generates embeddings (Phase 4)
        4. Stores in ChromaDB (Phase 2)

        Args:
            file_extensions: List of extensions to parse (default: all supported)
            progress_callback: Optional callback function for progress updates.
                Called with (event_type, data) where event_type is one of:
                - "files_found": data = {"files": List[Path], "codebase_path": Path}
                - "parsing_start": data = {"total": int}
                - "file_parsed": data = {"path": str, "symbols": int, "index": int, "total": int}
                - "parse_error": data = {"path": str, "error": str}
                - "chunks_created": data = {"total": int}
                - "embedding_start": data = {"total": int, "eta_minutes": float}
                - "embedding_progress": data = {"current": int, "total": int}
                - "complete": data = {"stats": dict}
        """
        if file_extensions is None:
            file_extensions = SUPPORTED_EXTENSIONS

        def emit(event_type: str, data: dict):
            """Emit progress event to callback if provided."""
            if progress_callback:
                progress_callback(event_type, data)

        # Legacy logging for when no callback provided
        if not progress_callback:
            self._log("\n" + "=" * 80)
            self._log("INDEXING CODEBASE")
            self._log("=" * 80)
            self._log(f"Codebase: {self.codebase_path}")
            self._log(f"Extensions: {file_extensions}")
            self._log(f"Embedding model: {self.embedding_model}")

        start_time = time.time()

        # Step 1: Find all files (single-pass discovery - 30x faster)
        if not progress_callback:
            self._log("\nStep 1: Finding files...")

        extensions_set = set(file_extensions)
        all_files = discover_files(
            self.codebase_path, extensions_set, progress_callback=progress_callback
        )

        self.stats["total_files"] = len(all_files)

        # Store file modification times for incremental reindexing
        file_mtimes = {}
        for filepath in all_files:
            try:
                file_mtimes[str(filepath)] = filepath.stat().st_mtime
            except OSError:
                pass  # Skip files that can't be stat'd
        self._metadata["file_mtimes"] = file_mtimes

        emit("files_found", {"files": all_files, "codebase_path": self.codebase_path})

        if not progress_callback:
            self._log(f"Found {len(all_files)} files")

        # Step 2+3: Parse files and create chunks (memory-optimized)
        # Symbols are converted to chunks per batch then freed, so we never
        # hold both a large all_symbols list and a large chunks list simultaneously.
        if not progress_callback:
            self._log("\nStep 2: Parsing files and creating chunks...")

        emit("parsing_start", {"total": len(all_files)})
        chunks = []
        total_symbols = 0

        # Use parallel parsing if enabled and there are enough files
        use_parallel = (
            self.parallel and len(all_files) > 50
        )  # Threshold increased for small projects

        if use_parallel:
            from codegrok_mcp.indexing.parallel_indexer import parallel_parse_files

            # Cap workers to limit memory (each holds a tree-sitter parser instance)
            effective_workers = self.max_workers
            if effective_workers is None:
                cpu_count = os.cpu_count() or 4
                effective_workers = max(1, min(cpu_count - 1, MAX_PARSE_WORKERS))

            if not progress_callback:
                self._log(f"  Using parallel parsing with {effective_workers} workers...")

            # Process files in batches to limit peak memory usage
            for batch_start in range(0, len(all_files), FILE_BATCH_SIZE):
                file_batch = all_files[batch_start : batch_start + FILE_BATCH_SIZE]
                batch_symbols, batch_errors = parallel_parse_files(
                    files=file_batch,
                    max_workers=effective_workers,
                    progress_callback=progress_callback,
                )
                self.stats["parse_errors"] += batch_errors
                total_symbols += len(batch_symbols)

                # Convert symbols to chunks immediately, then free symbol memory
                for symbol in batch_symbols:
                    chunks.append(self._create_chunk(symbol))
                del batch_symbols
                gc.collect()
        else:
            # Sequential parsing
            for i, filepath in enumerate(all_files, 1):
                symbols_count = 0
                try:
                    parsed = self.parser.parse_file(str(filepath))
                    symbols_count = len(parsed.symbols)
                    total_symbols += symbols_count

                    # Convert to chunks immediately (don't accumulate symbols)
                    for symbol in parsed.symbols:
                        chunks.append(self._create_chunk(symbol))

                    emit(
                        "file_parsed",
                        {
                            "path": str(filepath),
                            "symbols": symbols_count,
                            "index": i,
                            "total": len(all_files),
                        },
                    )
                except Exception as e:
                    self.stats["parse_errors"] += 1
                    emit("parse_error", {"path": str(filepath), "error": str(e)})
                    if not progress_callback and self.verbose and self.stats["parse_errors"] <= 5:
                        self._log(f"  Error parsing {filepath}: {e}")

                # Legacy progress for no callback
                if not progress_callback and self.verbose and i % 100 == 0:
                    print(f"  Parsed {i}/{len(all_files)} files...", end="\r")

        self.stats["total_symbols"] = total_symbols
        self.stats["total_chunks"] = len(chunks)

        emit("chunks_created", {"total": len(chunks)})

        if not progress_callback:
            self._log(f"\nParsed {total_symbols:,} symbols → {len(chunks):,} chunks")

        # Step 4: Get or create ChromaDB collection (supports resumption)
        if not progress_callback:
            self._log("\nStep 4: Creating vector database...")

        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": f"Code embeddings for {self.codebase_path.name}"},
        )

        # Track new chunk IDs for stale chunk cleanup after embedding
        new_chunk_ids = set(chunk.id for chunk in chunks)

        # Step 5: Generate embeddings and store (with checkpointing)
        eta_minutes = len(chunks) / 50 / 60  # ~50 embeddings/sec native

        emit("embedding_start", {"total": len(chunks), "eta_minutes": eta_minutes})

        if not progress_callback:
            self._log(f"\nStep 5: Generating embeddings (ETA: ~{eta_minutes:.1f} minutes)...")
            self._log("(You can interrupt and resume later)")

        # Load checkpoint if available (resume from interrupted indexing)
        checkpoint_path = (
            Path(self.persist_path).parent / "checkpoint.json" if self.persist_path else None
        )
        start_chunk_idx = 0
        if checkpoint_path:
            checkpoint = _load_checkpoint(checkpoint_path)
            if checkpoint and checkpoint.get("total_chunks") == len(chunks):
                resume_idx = checkpoint.get("chunks_completed", 0)
                if self.collection.count() >= resume_idx:
                    start_chunk_idx = resume_idx
                    if not progress_callback:
                        self._log(f"  Resuming from chunk {start_chunk_idx} (checkpoint found)")

        batch_size = 100
        embedding_start_time = time.time()
        chunks_per_second = None  # Will be calibrated after first batch

        for i in range(start_chunk_idx, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            current_count = i + len(batch)
            elapsed = time.time() - embedding_start_time

            # Calibrate speed after first batch, then update continuously
            chunks_processed = current_count - start_chunk_idx
            if elapsed > 0 and chunks_processed > 0:
                chunks_per_second = chunks_processed / elapsed

            # Calculate remaining time estimate
            remaining_seconds = None
            if chunks_per_second and chunks_per_second > 0:
                remaining_chunks = len(chunks) - current_count
                remaining_seconds = remaining_chunks / chunks_per_second

            emit(
                "embedding_progress",
                {
                    "current": current_count,
                    "total": len(chunks),
                    "elapsed_seconds": elapsed,
                    "remaining_seconds": remaining_seconds,
                    "chunks_per_second": chunks_per_second,
                },
            )

            # Legacy progress for no callback
            if not progress_callback and self.verbose and i % 500 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                remaining = (len(chunks) - i) / rate if rate > 0 else 0
                print(
                    f"  Embedded {i:,}/{len(chunks):,} chunks "
                    f"({i/len(chunks)*100:.1f}%) "
                    f"- ETA: {remaining/60:.1f}m",
                    end="\r",
                )

            try:
                # Generate embeddings for batch
                texts = [chunk.text for chunk in batch]

                # Native batch embedding (10-20x faster)
                embeddings = self.embedding_service.embed_batch(texts)

                # Upsert to ChromaDB (idempotent - safe for resumption)
                self.collection.upsert(
                    ids=[chunk.id for chunk in batch],
                    embeddings=embeddings,
                    documents=[chunk.text for chunk in batch],
                    metadatas=[chunk.metadata for chunk in batch],
                )
            except Exception as e:
                if not progress_callback:
                    self._log(f"\n  Error embedding batch {i}: {e}")
                continue

            # Save checkpoint every 1000 chunks
            if checkpoint_path and current_count % 1000 == 0:
                _save_checkpoint(checkpoint_path, current_count, len(chunks))

        # Free chunks list now that embedding is complete
        del chunks
        gc.collect()

        # Remove stale chunks (from deleted/renamed files)
        try:
            existing = self.collection.get(include=[])
            existing_ids = set(existing["ids"])
            stale_ids = list(existing_ids - new_chunk_ids)
            if stale_ids:
                # ChromaDB delete supports batching
                for j in range(0, len(stale_ids), 500):
                    self.collection.delete(ids=stale_ids[j : j + 500])
                if not progress_callback:
                    self._log(f"  Removed {len(stale_ids)} stale chunks")
        except Exception as e:
            if not progress_callback:
                self._log(f"  Warning: Could not clean stale chunks: {e}")

        # Clean up checkpoint on success
        if checkpoint_path and checkpoint_path.exists():
            checkpoint_path.unlink()

        self.stats["indexing_time"] = time.time() - start_time

        emit("complete", {"stats": self.stats.copy()})

        # Legacy summary for no callback
        if not progress_callback:
            self._log("\n\n" + "=" * 80)
            self._log("INDEXING COMPLETE")
            self._log("=" * 80)
            self._log(f"Files parsed:      {self.stats['total_files']:,}")
            self._log(f"Symbols extracted: {self.stats['total_symbols']:,}")
            self._log(f"Chunks created:    {self.stats['total_chunks']:,}")
            self._log(f"Parse errors:      {self.stats['parse_errors']}")
            self._log(
                f"Time elapsed:      {self.stats['indexing_time']:.1f}s ({self.stats['indexing_time']/60:.1f}m)"
            )
            self._log(f"Ready for retrieval!")
            self._log("=" * 80 + "\n")

    def get_sources_for_question(
        self,
        question: str,
        n_results: int = 10,
        language: Optional[str] = None,
        symbol_type: Optional[str] = None,
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        """
        Get source references and documents for a question.

        This performs semantic search to find relevant code chunks.

        Args:
            question: The question to search for
            n_results: Number of results to retrieve
            language: Optional language filter (e.g., 'python', 'javascript')
            symbol_type: Optional symbol type filter (e.g., 'function', 'class')

        Returns:
            Tuple of (documents, sources) where:
                - documents: List of document texts with metadata
                - sources: List of formatted source strings for display
        """
        if self.collection is None:
            return [], []

        # Embed the question
        query_embedding = self.embedding_service.embed(question, is_query=True)

        # Build optional metadata filter
        where_filter = None
        if language or symbol_type:
            where_filter = {}
            if language:
                where_filter["language"] = language
            if symbol_type:
                where_filter["type"] = symbol_type

        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=n_results, where=where_filter
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        # Format sources for display
        sources = []
        for metadata in metadatas:
            filepath = metadata["filepath"]
            try:
                filepath = str(Path(filepath).relative_to(self.codebase_path))
            except ValueError:
                pass
            sources.append(f"{filepath}:{metadata['line']} - {metadata['name']}()")

        # Build document list with metadata
        doc_results = []
        for doc, metadata in zip(documents, metadatas):
            doc_results.append({"text": doc, "metadata": metadata})

        return doc_results, sources

    def get_stats(self) -> Dict[str, Any]:
        """Get indexing statistics."""
        return self.stats.copy()

    def load_existing_index(self) -> bool:
        """
        Load an existing indexed collection from persistent storage.

        Returns:
            True if collection was loaded successfully, False otherwise
        """
        if not self.persist_path:
            self._log("No persistent storage configured")
            return False

        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
            count = self.collection.count()
            self._log(f"Loaded existing index with {count:,} chunks")
            return True
        except Exception as e:
            self._log(f"No existing index found: {e}")
            return False

    def save_metadata(self, metadata_path: str) -> None:
        """
        Save indexing metadata to JSON file.

        Args:
            metadata_path: Path to save metadata JSON
        """
        metadata = {
            "codebase_path": str(self.codebase_path),
            "embedding_model": self.embedding_model,
            "collection_name": self.collection_name,
            "indexed_at": datetime.now().isoformat(),
            "stats": self.stats,
            "file_mtimes": self._metadata.get("file_mtimes", {}),
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        self._log(f"Metadata saved to {metadata_path}")

    def load_metadata(self, metadata_path: str) -> Optional[Dict[str, Any]]:
        """
        Load indexing metadata from JSON file.

        Args:
            metadata_path: Path to metadata JSON file

        Returns:
            Metadata dictionary or None if not found
        """
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            # Restore stats
            if "stats" in metadata:
                self.stats = metadata["stats"]
            # Restore file modification times for incremental reindexing
            if "file_mtimes" in metadata:
                self._metadata["file_mtimes"] = metadata["file_mtimes"]
            return metadata
        except FileNotFoundError:
            return None

    def incremental_reindex(
        self,
        file_extensions: Optional[List[str]] = None,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Re-index only files that changed since last indexing.

        This method provides efficient incremental updates by:
        1. Comparing file modification times (mtime) against stored metadata
        2. Identifying new, modified, and deleted files
        3. Removing stale chunks from ChromaDB for modified/deleted files
        4. Parsing and indexing only new/modified files
        5. Updating metadata with new mtimes

        Args:
            file_extensions: List of extensions to consider (default: all supported)
            progress_callback: Optional callback function for progress updates.
                Called with (event_type, data) where event_type is one of:
                - "changes_detected": data = {"new": int, "modified": int, "deleted": int}
                - "parsing_start": data = {"total": int}
                - "embedding_start": data = {"total": int}
                - "complete": data = {"chunks_added": int}

        Returns:
            Dict with keys:
                - files_added: Number of new files indexed
                - files_modified: Number of modified files re-indexed
                - files_deleted: Number of deleted files removed
                - chunks_added: Number of new chunks added to ChromaDB
                - chunks_removed: Number of stale chunks removed from ChromaDB
                - time_seconds: Total time taken for incremental reindex
        """
        start_time = time.time()

        def emit(event_type: str, data: dict):
            """Emit progress event to callback if provided."""
            if progress_callback:
                progress_callback(event_type, data)

        extensions = file_extensions or SUPPORTED_EXTENSIONS

        # 1. Get stored file_mtimes from metadata
        stored_mtimes = self._metadata.get("file_mtimes", {})

        # 2. Scan current files and collect modification times (single-pass - 30x faster)
        extensions_set = set(extensions)
        all_current_paths = discover_files(Path(self.codebase_path), extensions_set)

        current_files: Dict[str, float] = {}
        for filepath in all_current_paths:
            try:
                current_files[str(filepath)] = filepath.stat().st_mtime
            except OSError:
                pass  # Skip files that can't be stat'd

        # 3. Categorize changes by comparing current vs stored
        stored_paths = set(stored_mtimes.keys())
        current_paths = set(current_files.keys())

        new_files = current_paths - stored_paths
        deleted_files = stored_paths - current_paths
        modified_files = {
            p for p in (current_paths & stored_paths) if current_files[p] > stored_mtimes.get(p, 0)
        }

        files_to_reindex = new_files | modified_files
        files_to_remove = deleted_files | modified_files

        # Emit changes detected event
        emit(
            "changes_detected",
            {"new": len(new_files), "modified": len(modified_files), "deleted": len(deleted_files)},
        )

        self._log(
            f"Incremental reindex: {len(new_files)} new, {len(modified_files)} modified, {len(deleted_files)} deleted"
        )

        chunks_removed = 0
        chunks_added = 0

        # 4. Remove stale chunks from ChromaDB for modified/deleted files
        if files_to_remove and self.collection:
            for filepath in files_to_remove:
                try:
                    # ChromaDB delete by metadata filter
                    self.collection.delete(where={"filepath": filepath})
                    chunks_removed += 1
                except Exception as e:
                    if self.verbose:
                        self._log(f"Warning: Could not remove chunks for {filepath}: {e}")

        # 5. Parse and index new/modified files
        if files_to_reindex:
            # Emit parsing start event
            emit("parsing_start", {"total": len(files_to_reindex)})

            chunks = []
            for filepath in files_to_reindex:
                try:
                    parsed = self.parser.parse_file(filepath)
                    for symbol in parsed.symbols:
                        chunk = self._create_chunk(symbol)
                        chunks.append(chunk)
                except Exception as e:
                    if self.verbose:
                        self._log(f"Warning: Could not parse {filepath}: {e}")
                    continue

            if chunks:
                # Emit embedding start event
                emit("embedding_start", {"total": len(chunks)})

                # Generate embeddings for new chunks
                texts = [chunk.text for chunk in chunks]

                embeddings = self.embedding_service.embed_batch(texts)

                # Add to ChromaDB
                self.collection.add(
                    ids=[chunk.id for chunk in chunks],
                    embeddings=embeddings,
                    documents=[chunk.text for chunk in chunks],
                    metadatas=[chunk.metadata for chunk in chunks],
                )
                chunks_added = len(chunks)

        # 6. Update metadata with new file modification times
        self._metadata["file_mtimes"] = current_files

        # Persist metadata if we have a persist path
        if self.persist_path:
            metadata_path = Path(self.persist_path).parent / "metadata.json"
            self.save_metadata(str(metadata_path))

        elapsed_time = round(time.time() - start_time, 2)

        result = {
            "files_added": len(new_files),
            "files_modified": len(modified_files),
            "files_deleted": len(deleted_files),
            "chunks_added": chunks_added,
            "chunks_removed": chunks_removed,
            "time_seconds": elapsed_time,
        }

        # Emit complete event
        emit("complete", {"chunks_added": chunks_added})

        self._log(
            f"Incremental reindex complete in {elapsed_time}s: "
            f"+{chunks_added} chunks, -{chunks_removed} files processed"
        )

        return result
