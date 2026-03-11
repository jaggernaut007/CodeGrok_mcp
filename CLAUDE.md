# CodeGrok MCP

MCP server for semantic code search. No CLI, no LLM - pure indexing and search service via MCP protocol.

## Structure

```
src/codegrok_mcp/
├── core/models.py           # Symbol, ParsedFile, CodebaseIndex + Memory, MemoryType
├── core/interfaces.py       # IParser abstract interface
├── parsers/treesitter_parser.py  # Multi-language AST parsing (9 languages)
├── parsers/language_configs.py   # Tree-sitter node → SymbolType mappings
├── indexing/embedding_service.py # nomic-ai/CodeRankEmbed wrapper
├── indexing/parallel_indexer.py  # Thread-pool file processing
├── indexing/source_retriever.py  # Code indexing + semantic search
├── indexing/memory_retriever.py  # Memory storage + semantic recall
└── mcp/
    ├── state.py             # Global singleton (SourceRetriever + MemoryRetriever)
    └── server.py            # FastMCP server (8 tools)
```

## MCP Tools (8 total)

### Code Search
| Tool | Purpose |
|------|---------|
| `learn` | Index codebase. Modes: `auto` (incremental), `full`, `load_only`. **Required first.** |
| `get_sources` | Semantic search. Returns ranked code chunks with filters (language, symbol_type) |
| `get_stats` | Index statistics (files, symbols, chunks) |
| `list_supported_languages` | Supported language extensions |

### Memory Layer
| Tool | Purpose |
|------|---------|
| `remember` | Store memories (conversation, status, decision, preference, doc, note) |
| `recall` | Semantic search over memories. Filters: type, tags, time_range |
| `forget` | Delete memories by ID, type, tags, or age. **Destructive.** |
| `memory_stats` | Memory statistics by type |

## Key Constants

- **Embedding Model**: `nomic-ai/CodeRankEmbed` (768 dims, 8192 max tokens)
- **Chunk Strategy**: Symbol-based (each function/class/method = 1 chunk)
- **Max Chunk Size**: 4000 chars (~1000-1300 tokens)
- **Storage**: `.codegrok/` (chromadb/ + metadata.json + memory_metadata.json + checkpoint.json)
- **Parallelism**: CPU count - 1 workers (min 1, max 4 for parsing)
- **File Batch Size**: 500 files per parse batch (memory optimization)
- **Default Timeout**: 600s (configurable via `CODEGROK_TIMEOUT` env var or `timeout_seconds` param)
- **Memory TTLs**: session (24h), day, week, month, permanent

## Commands

```bash
pip install -e .          # Install
codegrok-mcp              # Run server
pytest                    # Test
black src/ --line-length 100  # Format
mypy src/                 # Type check
```

## Gotchas

1. **State is global singleton** - `state.py` holds SourceRetriever + MemoryRetriever across MCP calls
2. **Incremental reindex uses mtime** - File modification time comparison for change detection
3. **ChromaDB collections**: `codebase_chunks` (code) and `memories` (memory layer)
4. **No LLM code** - Removed from parent CodeGrok; source_retriever.py has no ask/rerank methods
5. **Tree-sitter node names vary by language** - language_configs.py normalizes them
6. **Embedding is cached** - LRU(1000) + batch processing in embedding_service.py
7. **Memory tags stored as CSV** - ChromaDB doesn't support list metadata; tags joined with commas
8. **All tools require `learn` first** - Except `list_supported_languages` (static data)
9. **discover_files respects .gitignore** - Uses `pathspec` + `os.walk()` with directory pruning; also respects nested `.gitignore` files
10. **Indexing uses upsert** - `collection.upsert()` instead of delete-recreate; stale chunks cleaned after embedding
11. **Checkpointing** - `.codegrok/checkpoint.json` saves progress every 1000 chunks; atomic writes via `os.replace()`; deleted on success
12. **max_files safety limit** - `discover_files()` stops at 200K files to prevent DoS (addresses SECURITY_REVIEW HIGH-003)
13. **Memory-optimized parsing** - Symbols converted to chunks per file batch (500 files), then freed; `gc.collect()` between batches; chunks list freed after embedding
14. **Worker cap** - Parallel parse workers capped at `MAX_PARSE_WORKERS=4` to limit memory from tree-sitter instances
15. **Configurable timeout** - `learn` tool has `timeout_seconds` param; also reads `CODEGROK_TIMEOUT` env var; defaults to 600s
16. **Background indexing** - `learn` returns immediately; indexing runs in `threading.Thread(daemon=True)`; client polls `get_stats()` for progress; `IndexingStatus` in `state.py` tracks progress thread-safely
17. **learn stateful responses** - Returns `indexing_started` (new), `indexing_in_progress` (already running), `complete` (done, clears result), or raises `ToolError` (failed, clears error for retry)

## Adding Languages

1. Verify tree-sitter-languages support
2. Add node mappings to `language_configs.py`
3. Add extensions to `SUPPORTED_LANGUAGES` in treesitter_parser.py

## Adding MCP Tools

1. Add `@mcp.tool` function in `server.py`
2. Use `state.get_state()` for retriever access
3. Add `ToolAnnotations` (readOnlyHint, destructiveHint, idempotentHint)
4. Return dict with results or raise `ToolError` on failure

## Memory Types

| Type | Use Case |
|------|----------|
| `conversation` | Chat history, Q&A exchanges |
| `status` | Project blockers, progress notes |
| `decision` | Architecture choices, rationale |
| `preference` | User coding style, settings |
| `doc` | Documentation snippets |
| `note` | General notes, TODOs |
