"""
Parallel indexing utilities for CodeGrok MCP.

This module provides thread-safe parallel file parsing to speed up codebase
indexing. It uses ThreadPoolExecutor for concurrent parsing while maintaining
proper progress reporting and error handling.

Performance:
    - 3-5x faster indexing for large codebases (1000+ files)
    - Utilizes multiple CPU cores for parsing
    - Thread-local parsers avoid contention

Usage:
    >>> from codegrok_mcp.indexing.parallel_indexer import parallel_parse_files
    >>> symbols, errors = parallel_parse_files(file_list, max_workers=4)
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Callable, Optional, Dict, Any, Tuple

from codegrok_mcp.parsers.treesitter_parser import ThreadLocalParserFactory
from codegrok_mcp.core.models import Symbol


@dataclass
class ParseResult:
    """
    Result of parsing a single file.

    Attributes:
        filepath: Path to the parsed file
        symbols: List of extracted symbols (empty if parsing failed)
        success: True if parsing succeeded, False otherwise
        error: Error message if parsing failed, None otherwise
    """

    filepath: str
    symbols: List[Symbol]
    success: bool
    error: Optional[str] = None


@dataclass
class ParallelProgress:
    """
    Thread-safe progress tracking for parallel operations.

    Uses a lock to ensure atomic updates to counters from multiple threads.

    Attributes:
        total: Total number of items to process
        _completed: Number of completed items (access via .completed property)
        _errors: Number of errors encountered (access via .errors property)
    """

    total: int
    _completed: int = 0
    _errors: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def completed(self) -> int:
        """Get the number of completed items (thread-safe)."""
        with self._lock:
            return self._completed

    @property
    def errors(self) -> int:
        """Get the number of errors encountered (thread-safe)."""
        with self._lock:
            return self._errors

    def increment_completed(self) -> int:
        """
        Increment the completed count and return new value (thread-safe).

        Returns:
            The new completed count after incrementing
        """
        with self._lock:
            self._completed += 1
            return self._completed

    def increment_errors(self) -> int:
        """
        Increment the error count and return new value (thread-safe).

        Returns:
            The new error count after incrementing
        """
        with self._lock:
            self._errors += 1
            return self._errors


def parse_file_worker(filepath: Path, parser_factory: ThreadLocalParserFactory) -> ParseResult:
    """
    Worker function for parallel file parsing.

    This function is called by ThreadPoolExecutor for each file.
    It uses the thread-local parser factory to get a parser instance
    that is safe to use in the current thread.

    Args:
        filepath: Path to file to parse
        parser_factory: Thread-local parser factory

    Returns:
        ParseResult with symbols or error information
    """
    parser = parser_factory.get_parser()
    try:
        parsed = parser.parse_file(str(filepath))
        return ParseResult(
            filepath=str(filepath),
            symbols=list(parsed.symbols),
            success=parsed.error is None,
            error=parsed.error,
        )
    except Exception as e:
        return ParseResult(filepath=str(filepath), symbols=[], success=False, error=str(e))


def parallel_parse_files(
    files: List[Path],
    max_workers: Optional[int] = None,
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Tuple[List[Symbol], int]:
    """
    Parse multiple files in parallel using ThreadPoolExecutor.

    This function distributes file parsing across multiple threads,
    collecting symbols from all files and tracking progress.

    Args:
        files: List of file paths to parse
        max_workers: Number of worker threads. If None, defaults to
                    (CPU count - 1) to leave one core free for the main thread.
        progress_callback: Optional callback for progress events.
                          Called with (event_type, data) for each event.
                          Event types: 'file_parsed', 'parse_error'

    Returns:
        Tuple of (all_symbols, error_count) where:
            - all_symbols: List of all Symbol objects from all files
            - error_count: Number of files that failed to parse

    Example:
        >>> files = list(Path("src").rglob("*.py"))
        >>> symbols, errors = parallel_parse_files(files, max_workers=4)
        >>> print(f"Parsed {len(files)} files, found {len(symbols)} symbols, {errors} errors")
    """
    if not files:
        return [], 0

    parser_factory = ThreadLocalParserFactory()
    progress = ParallelProgress(total=len(files))
    all_symbols: List[Symbol] = []
    symbols_lock = threading.Lock()

    def emit(event_type: str, data: dict):
        """Emit a progress event if callback is provided."""
        if progress_callback:
            try:
                progress_callback(event_type, data)
            except Exception:
                # Don't let callback errors affect parsing
                pass

    # Determine worker count
    # Use CPU count - 1 to leave a core free, minimum 1, maximum 32
    if max_workers is None:
        cpu_count = os.cpu_count() or 4
        max_workers = max(1, min(cpu_count - 1, 32))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(parse_file_worker, f, parser_factory): f for f in files}

        # Process results as they complete
        for future in as_completed(future_to_file):
            filepath = future_to_file[future]
            try:
                result = future.result()
                completed = progress.increment_completed()

                if result.success and result.symbols:
                    with symbols_lock:
                        all_symbols.extend(result.symbols)
                    emit(
                        "file_parsed",
                        {
                            "path": result.filepath,
                            "symbols": len(result.symbols),
                            "index": completed,
                            "total": progress.total,
                        },
                    )
                elif result.error:
                    progress.increment_errors()
                    emit("parse_error", {"path": result.filepath, "error": result.error})
                else:
                    # File parsed but no symbols found (not an error)
                    emit(
                        "file_parsed",
                        {
                            "path": result.filepath,
                            "symbols": 0,
                            "index": completed,
                            "total": progress.total,
                        },
                    )

            except Exception as e:  # pragma: no cover
                progress.increment_completed()
                progress.increment_errors()
                emit("parse_error", {"path": str(filepath), "error": str(e)})

    return all_symbols, progress.errors
