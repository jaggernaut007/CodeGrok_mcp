# Indexing Improvements (v0.2.1)

Fixes for the `learn` tool hanging on large codebases with many folders/subfolders.

## Changes

### 1. `.gitignore` Support

`discover_files()` now respects `.gitignore` patterns using the `pathspec` library.

- Uses `os.walk()` instead of `Path.rglob("*")` for directory pruning
- Loads root `.gitignore` and stacks nested `.gitignore` files as it descends
- Prunes ignored directories in-place (never descends into `node_modules/`, `build/`, etc.)
- Uses `followlinks=False` to prevent symlink loops
- Backward-compatible: `respect_gitignore=True` by default, can be disabled

### 2. Safety Limits

- `max_files=200_000` circuit breaker stops file discovery if exceeded
- Emits a warning when the limit is hit
- Addresses **SECURITY_REVIEW HIGH-003** (Unbounded Resource Consumption / DoS)

### 3. Upsert-Based Indexing

- `index_codebase()` now uses `get_or_create_collection()` + `collection.upsert()` instead of deleting and recreating the collection
- Chunk IDs are deterministic (`filepath:name:line_start`), making upsert idempotent
- Stale chunks (from deleted/renamed files) are cleaned up after the embedding loop

### 4. Resumable Checkpointing

- Saves progress to `.codegrok/checkpoint.json` every 1000 chunks
- Atomic writes via `os.replace()` (POSIX-safe)
- On restart, detects checkpoint and resumes from where it left off
- Checkpoint is deleted on successful completion

### 5. Improved Progress Reporting

- New `"discovery_progress"` event emitted every 1000 files during file traversal
- ETA added to embedding progress messages (e.g., "Embedding... (5000/10000 chunks, ~2.3m remaining)")
- MCP client now shows progress during the file discovery phase (0-5% range)

## New Dependencies

- `pathspec>=0.11.0` — Pure Python `.gitignore` pattern matching (used by `black`, `flake8`, etc.)

## Security Alignment

| Security Finding | How Addressed |
|-----------------|---------------|
| HIGH-003: Unbounded Resource Consumption | `max_files` limit + `.gitignore` filtering |
| LOW-009: Symlink Following | `followlinks=False` in `os.walk()` |

## MCP Tools Used in Development

This feature was planned and implemented using the following MCP tools:

| MCP Tool | How It Was Used |
|----------|----------------|
| **Sequential Thinking** (`mcp__sequential-thinking__sequentialthinking`) | 7-step chain-of-thought to plan execution order, identify risks (backward compatibility, atomic writes, symlink loops), decide to skip async writer thread, and design test strategy |
| **Snyk Code Scan** (`mcp__Snyk__snyk_code_scan`) | SAST scan on all modified files (`source_retriever.py`, `server.py`) — 0 issues found |

## Test Coverage

### New Unit Tests (`tests/unit/test_discover_files.py`)

| Test | What it verifies |
|------|-----------------|
| `test_discover_files_basic` | Finds .py files in simple directory |
| `test_discover_files_skip_dirs` | Skips `node_modules/`, `__pycache__/`, `.git/` even when nested |
| `test_discover_files_gitignore` | Respects root `.gitignore` patterns |
| `test_discover_files_nested_gitignore` | Handles `.gitignore` in subdirectories |
| `test_discover_files_max_files_limit` | Stops at `max_files` and returns partial results |
| `test_discover_files_no_gitignore` | Works when no `.gitignore` exists |
| `test_discover_files_respect_gitignore_false` | Opt-out disables gitignore filtering |
| `test_discover_files_progress_callback` | Callback mechanism works correctly |

### New Integration Tests (`tests/integration/test_source_retriever.py`)

| Test | What it verifies |
|------|-----------------|
| `test_index_codebase_upsert_idempotent` | Re-indexing produces same chunk count |
| `test_stale_chunk_removal` | Old chunks removed after file deletion |
| `test_checkpoint_save_and_load` | Checkpoint round-trip |
| `test_checkpoint_load_missing_file` | Handles missing checkpoint |
| `test_checkpoint_load_corrupted` | Handles corrupted JSON |
| `test_checkpoint_cleanup_on_success` | Checkpoint deleted after success |
| `test_checkpoint_load_none_path` | Handles None path |
