import pytest
from pathlib import Path
from codegrok_mcp.indexing.parallel_indexer import (
    ParallelProgress,
    parallel_parse_files,
    parse_file_worker,
)
from codegrok_mcp.parsers.treesitter_parser import ThreadLocalParserFactory


def test_parallel_progress_increment():
    progress = ParallelProgress(total=10)
    assert progress.completed == 0
    assert progress.errors == 0

    new_val = progress.increment_completed()
    assert new_val == 1
    assert progress.completed == 1

    new_err = progress.increment_errors()
    assert new_err == 1
    assert progress.errors == 1


def test_parallel_parse_files_empty_list():
    symbols, errors = parallel_parse_files([])
    assert symbols == []
    assert errors == 0


def test_parallel_parse_files_single_file(tmp_path):
    f = tmp_path / "test.py"
    f.write_text("def foo(): pass")

    symbols, errors = parallel_parse_files([f])

    assert errors == 0
    assert len(symbols) >= 1
    assert symbols[0].name == "foo"
    assert symbols[0].filepath == str(f)


def test_parse_file_worker_success(tmp_path):
    f = tmp_path / "test.py"
    f.write_text("def worker_test(): pass")
    factory = ThreadLocalParserFactory()

    result = parse_file_worker(f, factory)

    assert result.success
    assert result.filepath == str(f)
    assert len(result.symbols) >= 1
    assert result.symbols[0].name == "worker_test"


def test_parse_file_worker_error(tmp_path):
    # Depending on parser implementation, syntax error might not raise exception but return ParseResult with error
    # Or might parse partially.
    # Let's try a directory which should fail open() or similar if parser tries to read it as file
    d = tmp_path / "subdir"
    d.mkdir()
    factory = ThreadLocalParserFactory()

    # parse_file usually expects a file path string. If it fails, worker catches exception.
    # TreeSitterParser.parse_file reads content.

    result = parse_file_worker(d, factory)

    assert not result.success
    assert result.error is not None


def test_parallel_parse_files_with_error(tmp_path):
    d = tmp_path / "subdir"
    d.mkdir()
    # Pass a directory which should fail in worker
    symbols, errors = parallel_parse_files([d])
    assert errors == 1
    assert symbols == []
