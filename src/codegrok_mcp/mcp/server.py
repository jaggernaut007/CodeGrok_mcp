"""
MCP Server for CodeGrok - Semantic code search and source retrieval.

This module provides an MCP server that exposes CodeGrok's semantic search
functionality through the Model Context Protocol using stdio transport.

Usage:
    codegrok-mcp  # Run as stdio MCP server

Tools:
    - learn: Index a codebase (smart mode: auto-detects existing index)
        - mode='auto': Incremental reindex if exists, full index if new
        - mode='full': Force complete re-index
        - mode='load_only': Just load existing index
    - get_sources: Semantic search for code
    - get_stats: Get indexing statistics
    - list_supported_languages: List supported file extensions
"""

from typing import Optional, List, Dict, Any, Annotated, Callable
from pathlib import Path
import asyncio
import os
import threading

from fastmcp import FastMCP, Context
from fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from codegrok_mcp.mcp.state import get_state, IndexingStatus

# Lazy import SourceRetriever to avoid heavy startup cost
# from codegrok_mcp.indexing.source_retriever import SourceRetriever, SUPPORTED_EXTENSIONS
from codegrok_mcp.parsers.language_configs import EXTENSION_MAP

# Derived supported extensions list (lightweight)
SUPPORTED_EXTENSIONS = sorted(list(EXTENSION_MAP.keys()))


# Storage constants (same as CLI)
CODEGROK_DIR = ".codegrok"
CHROMA_DIR = "chroma"
METADATA_FILE = "metadata.json"

# Default timeout for indexing operations (seconds).
# Configurable via CODEGROK_TIMEOUT environment variable in MCP client config.
# Example in claude_desktop_config.json:
#   "env": {"CODEGROK_TIMEOUT": "1200"}
DEFAULT_TIMEOUT_SECONDS = 600  # 10 minutes


def _get_timeout(override: Optional[int] = None) -> int:
    """Get the effective timeout in seconds.

    Priority: per-call override > CODEGROK_TIMEOUT env var > DEFAULT_TIMEOUT_SECONDS.
    """
    if override is not None and override > 0:
        return override
    env_timeout = os.environ.get("CODEGROK_TIMEOUT")
    if env_timeout:
        try:
            val = int(env_timeout)
            if val > 0:
                return val
        except ValueError:
            pass
    return DEFAULT_TIMEOUT_SECONDS


# Initialize FastMCP server
mcp = FastMCP(
    name="CodeGrok",
    instructions="""CodeGrok: Semantic code search + project memory.

🚀 START: Call 'learn' with codebase path to enable all features.

CODE SEARCH (after learn):
• get_sources - Find relevant code using natural language queries
• get_stats - Check index status (files, symbols, chunks indexed)
• list_supported_languages - See supported file extensions

PROJECT MEMORY (after learn):
• remember - Store decisions, preferences, notes, status updates
• recall - Retrieve memories by semantic search
• forget - Remove outdated information (⚠️ destructive)
• memory_stats - View memory statistics

MEMORY REPLACES .md FILES FOR:
- Architecture decisions ("We chose PostgreSQL because...")
- User preferences ("Uses 4-space indent, prefers functional style")
- Project status ("Auth module blocked on DB schema review")
- Key discussions ("Discussed auth flow on 2025-01-15")
- Documentation notes ("API returns paginated results, max 100")

TYPICAL WORKFLOW:
1. learn(path="/project") - Index codebase (required first step)
2. recall("user preferences") - Check existing context
3. remember("Decision: Using Redis for caching", memory_type="decision")
4. get_sources("authentication flow") - Find relevant code""",
)


def _get_codegrok_paths(codebase_path: Path) -> Dict[str, Path]:
    """Get paths to .codegrok storage locations."""
    codegrok_dir = codebase_path / CODEGROK_DIR
    return {
        "codegrok_dir": codegrok_dir,
        "chroma_path": codegrok_dir / CHROMA_DIR,
        "metadata_path": codegrok_dir / METADATA_FILE,
    }


def _has_valid_index(paths: Dict[str, Path]) -> bool:
    """Check if a valid CodeGrok index exists at the given paths."""
    return (
        paths["codegrok_dir"].exists()
        and paths["metadata_path"].exists()
        and paths["chroma_path"].exists()
    )


def _create_bg_progress_callback(indexing_status: IndexingStatus) -> Callable:
    """Create a progress callback that updates IndexingStatus (thread-safe)."""

    def callback(event_type: str, data: dict):
        progress = 0
        message = ""

        if event_type == "discovery_progress":
            progress = min(4, int(data.get("files_found", 0) / 1000))
            message = f"Discovering files... ({data.get('files_found', 0)} found)"
        elif event_type == "files_found":
            progress = 5
            message = f"Found {len(data['files'])} files..."
        elif event_type == "parsing_start":
            progress = 10
            message = f"Parsing {data['total']} files..."
        elif event_type == "chunks_created":
            progress = 30
            message = f"Created {data['total']} code chunks..."
        elif event_type == "embedding_start":
            progress = 35
            message = f"Generating embeddings for {data['total']} chunks..."
        elif event_type == "embedding_progress":
            pct = data["current"] / data["total"] if data["total"] > 0 else 1
            progress = 35 + int(pct * 60)
            remaining = data.get("remaining_seconds")
            if remaining and remaining > 0:
                eta_str = (
                    f", ~{remaining / 60:.1f}m remaining"
                    if remaining >= 60
                    else f", ~{remaining:.0f}s remaining"
                )
            else:
                eta_str = ""
            message = f"Embedding... ({data['current']}/{data['total']} chunks{eta_str})"
        elif event_type == "changes_detected":
            progress = 10
            message = f"Found {data['new']} new, {data['modified']} modified files..."
        elif event_type == "complete":
            progress = 99
            message = "Finishing up..."

        if progress > 0:
            indexing_status.update(progress, message)

    return callback


def _run_full_index_bg(
    codebase_path: Path,
    paths: Dict[str, Path],
    state,
    file_extensions: Optional[List[str]],
    embedding_model: str,
):
    """Run full indexing in a background thread. Updates state on completion."""
    from codegrok_mcp.indexing.source_retriever import SourceRetriever

    try:
        retriever = SourceRetriever(
            codebase_path=str(codebase_path),
            embedding_model=embedding_model,
            verbose=False,
            persist_path=str(paths["chroma_path"]),
        )

        extensions = file_extensions if file_extensions else SUPPORTED_EXTENSIONS
        progress_callback = _create_bg_progress_callback(state.indexing)

        retriever.index_codebase(file_extensions=extensions, progress_callback=progress_callback)
        retriever.save_metadata(str(paths["metadata_path"]))

        state.retriever = retriever
        state.codebase_path = codebase_path

        state.indexing.complete(
            {"success": True, "mode_used": "full", "stats": retriever.get_stats()}
        )
    except Exception as e:
        state.indexing.fail(str(e))


def _run_incremental_reindex_bg(
    codebase_path: Path,
    paths: Dict[str, Path],
    state,
    embedding_model: str,
):
    """Run incremental reindex in a background thread."""
    from codegrok_mcp.indexing.source_retriever import SourceRetriever

    try:
        retriever = SourceRetriever(
            codebase_path=str(codebase_path),
            embedding_model=embedding_model,
            verbose=False,
            persist_path=str(paths["chroma_path"]),
        )

        if not retriever.load_existing_index():
            state.indexing.fail(f"Failed to load existing index from {paths['chroma_path']}")
            return

        retriever.load_metadata(str(paths["metadata_path"]))

        progress_callback = _create_bg_progress_callback(state.indexing)
        result = retriever.incremental_reindex(progress_callback=progress_callback)

        retriever.save_metadata(str(paths["metadata_path"]))

        state.retriever = retriever
        state.codebase_path = codebase_path

        state.indexing.complete({"success": True, "mode_used": "incremental", **result})
    except Exception as e:
        state.indexing.fail(str(e))


@mcp.tool(
    name="learn",
    description="""Index a codebase for semantic search. REQUIRED FIRST STEP before using any other tool.

Modes:
- auto (default): Smart detection. If index exists, updates incrementally. If new, does full index.
- full: Force complete re-index (destroys existing index).
- load_only: Just load existing index without any indexing.

Indexing runs in the background — this tool returns immediately.
Call get_stats() to check progress. Once complete, search tools become available.

Creates a .codegrok/ folder in the codebase directory.""",
    annotations=ToolAnnotations(
        readOnlyHint=False,  # Creates/modifies .codegrok/ directory
        destructiveHint=False,  # Doesn't destroy user data (only index data)
        idempotentHint=True,  # Safe to re-run on same path
        openWorldHint=False,  # Only accesses local filesystem
    ),
)
async def learn(
    path: Annotated[str, Field(description="Absolute path to the codebase directory to index")],
    mode: Annotated[
        str,
        Field(
            description="Indexing mode: 'auto' (smart detection), 'full' (force re-index), 'load_only' (just load)"
        ),
    ] = "auto",
    file_extensions: Annotated[
        Optional[List[str]],
        Field(
            description="Optional list of file extensions to include (e.g., ['.py', '.js']). Defaults to all supported extensions."
        ),
    ] = None,
    embedding_model: Annotated[
        str, Field(description="Embedding model to use (default: coderankembed)")
    ] = "coderankembed",
    timeout_seconds: Annotated[
        Optional[int],
        Field(
            description="Timeout in seconds for indexing. Overrides CODEGROK_TIMEOUT env var. Default: 600 (10 min)."
        ),
    ] = None,
    ctx: Context = None,
) -> Dict[str, Any]:
    """Index a codebase with smart mode detection. Returns immediately; indexing runs in background."""
    state = get_state()

    # If indexing is already in progress, return current status
    if state.indexing.active:
        return {
            "success": True,
            "status": "indexing_in_progress",
            "message": "Indexing is already running. Call get_stats() to check progress.",
            **state.indexing.to_dict(),
        }

    # If last indexing completed, return the result and clear it
    if state.indexing.result is not None:
        result = state.indexing.result.copy()
        result["status"] = "complete"
        result["message"] = "Indexing completed successfully."
        state.indexing.result = None  # Clear so next call can re-index
        return result

    # If last indexing failed, return the error
    if state.indexing.error is not None:
        error = state.indexing.error
        state.indexing.error = None  # Clear so next call can retry
        raise ToolError(f"Previous indexing failed: {error}. Retrying...")

    # Validate mode
    valid_modes = ("auto", "full", "load_only")
    if mode not in valid_modes:
        raise ToolError(f"Invalid mode '{mode}'. Must be one of: {valid_modes}")

    # Validate path
    codebase_path = Path(path).resolve()
    if not codebase_path.exists():
        raise ToolError(f"Path does not exist: {codebase_path}")
    if not codebase_path.is_dir():
        raise ToolError(f"Path is not a directory: {codebase_path}")

    # Get storage paths
    paths = _get_codegrok_paths(codebase_path)
    has_existing = _has_valid_index(paths)

    # Handle load_only mode (fast, no background needed)
    if mode == "load_only":
        if not has_existing:
            raise ToolError(
                f"No existing index found at {codebase_path}. "
                "Use mode='auto' or mode='full' to create one."
            )
        return _load_existing_index_sync(codebase_path, paths, state, embedding_model)

    # Start background indexing
    paths["codegrok_dir"].mkdir(parents=True, exist_ok=True)

    if mode == "auto" and has_existing:
        state.indexing.start("Starting incremental reindex...")
        thread = threading.Thread(
            target=_run_incremental_reindex_bg,
            args=(codebase_path, paths, state, embedding_model),
            daemon=True,
        )
    else:
        state.indexing.start("Starting full index...")
        thread = threading.Thread(
            target=_run_full_index_bg,
            args=(codebase_path, paths, state, file_extensions, embedding_model),
            daemon=True,
        )

    thread.start()

    return {
        "success": True,
        "status": "indexing_started",
        "message": (
            f"Indexing started for {codebase_path.name}. "
            "Call get_stats() to check progress. "
            "Search tools will be available once indexing completes."
        ),
        **state.indexing.to_dict(),
    }


def _load_existing_index_sync(
    codebase_path: Path, paths: Dict[str, Path], state, embedding_model: str
) -> Dict[str, Any]:
    """Load an existing index without any reindexing (synchronous, fast)."""
    from codegrok_mcp.indexing.source_retriever import SourceRetriever

    retriever = SourceRetriever(
        codebase_path=str(codebase_path),
        embedding_model=embedding_model,
        verbose=False,
        persist_path=str(paths["chroma_path"]),
    )

    if not retriever.load_existing_index():
        raise ToolError(f"Failed to load index from {paths['chroma_path']}")

    metadata = retriever.load_metadata(str(paths["metadata_path"]))
    stats = metadata.get("stats", {}) if metadata else {}
    indexed_at = metadata.get("indexed_at") if metadata else None

    state.retriever = retriever
    state.codebase_path = codebase_path

    return {
        "success": True,
        "status": "complete",
        "mode_used": "load_only",
        "message": f"Loaded existing index for {codebase_path.name}",
        "stats": stats,
        "indexed_at": indexed_at,
    }


@mcp.tool(
    name="get_sources",
    description="""Get source code references relevant to a query using semantic search. Requires 'learn' first.

Supports optional filtering by language and symbol type. Returns ranked code chunks with file paths, line numbers, and relevance scores.

Examples:
- Find auth code: get_sources(question="authentication login flow")
- Find Python classes: get_sources(question="user model", language="python", symbol_type="class")""",
    annotations=ToolAnnotations(
        readOnlyHint=True,  # Only reads from index
        idempotentHint=True,  # Same query returns same results
        openWorldHint=False,  # Only accesses local ChromaDB
    ),
)
def get_sources(
    question: Annotated[str, Field(description="Natural language question or search query")],
    n_results: Annotated[
        int, Field(description="Number of source references to return (default: 10)", ge=1, le=50)
    ] = 10,
    language: Annotated[
        Optional[str],
        Field(description="Filter by language (e.g., 'python', 'javascript', 'typescript')"),
    ] = None,
    symbol_type: Annotated[
        Optional[str],
        Field(description="Filter by symbol type (e.g., 'function', 'class', 'method')"),
    ] = None,
) -> Dict[str, Any]:
    """Get source references for a question with optional filters."""
    state = get_state()

    if not state.is_loaded:
        raise ToolError("No codebase loaded. Use 'learn' first.")

    try:
        # get_sources_for_question returns tuple: (doc_results, formatted_sources)
        doc_results, formatted_sources = state.retriever.get_sources_for_question(
            question, n_results=n_results, language=language, symbol_type=symbol_type
        )

        return {
            "documents": doc_results,  # Full document data with metadata
            "sources": formatted_sources,  # Formatted source references for display
        }

    except Exception as e:
        raise ToolError(f"Failed to get sources: {str(e)}")


@mcp.tool(
    name="get_stats",
    description="""Get statistics about the currently loaded codebase index and indexing progress.

Returns: files indexed, total chunks, symbols by type, languages detected, index creation time.
If indexing is in progress, also returns progress percentage and ETA.""",
    annotations=ToolAnnotations(
        readOnlyHint=True,  # Only reads metadata
        idempotentHint=True,  # Same state = same results
        openWorldHint=False,  # Only accesses local state
    ),
)
def get_stats() -> Dict[str, Any]:
    """Get indexing statistics and progress."""
    state = get_state()

    result: Dict[str, Any] = {}

    # Include indexing status if active or recently completed/failed
    indexing_info = state.indexing.to_dict()
    if indexing_info["active"] or indexing_info["error"]:
        result["indexing"] = indexing_info

    if not state.is_loaded:
        result["loaded"] = False
        result["codebase_path"] = None
        result["stats"] = None
        return result

    result["loaded"] = True
    result["codebase_path"] = str(state.codebase_path)
    result["stats"] = state.retriever.get_stats()
    return result


@mcp.tool(
    name="list_supported_languages",
    description="""List all programming languages and file extensions supported by CodeGrok.

Returns extensions grouped by language. Currently supports: Python, JavaScript, TypeScript, Go, Rust, Java, C, C++, Ruby, and more.""",
    annotations=ToolAnnotations(
        readOnlyHint=True,  # Returns static data
        idempotentHint=True,  # Always same result
        openWorldHint=False,  # No external access
    ),
)
def list_supported_languages() -> Dict[str, Any]:
    """List supported file extensions and languages."""
    # Group extensions by language
    languages: Dict[str, List[str]] = {}
    for ext, lang in EXTENSION_MAP.items():
        if lang not in languages:
            languages[lang] = []
        languages[lang].append(ext)

    # Sort extensions within each language
    for lang in languages:
        languages[lang] = sorted(languages[lang])

    return {"extensions": sorted(EXTENSION_MAP.keys()), "languages": languages}


# =============================================================================
# Memory Tools (Universal Memory Layer)
# =============================================================================


@mcp.tool(
    name="remember",
    description="""Store a memory for later retrieval. Requires 'learn' first to set project context.

Memory types:
- conversation: Chat history, Q&A exchanges, discussion summaries
- status: Project status updates, blockers, progress notes
- decision: Architecture decisions, technical choices, rationale
- preference: User preferences, coding style, tool settings
- doc: Documentation snippets, README content, API docs
- note: General notes, reminders, TODOs

TTL options:
- session: Cleared after 24 hours
- day: Kept for 1 day
- week: Kept for 1 week
- month: Kept for 30 days
- permanent: Never expires (default)

Examples:
- Remember user prefers tabs: remember(content="User prefers tabs over spaces", memory_type="preference")
- Remember a blocker: remember(content="Auth blocked on DB schema", memory_type="status", tags=["auth", "blocked"])
- Remember a decision: remember(content="Using JWT with refresh tokens for auth", memory_type="decision", tags=["auth"])
""",
    annotations=ToolAnnotations(
        readOnlyHint=False,  # Writes to ChromaDB
        destructiveHint=False,  # Adds data, doesn't delete
        idempotentHint=False,  # Creates new memory each call
        openWorldHint=False,  # Only local storage
    ),
)
def remember(
    content: Annotated[str, Field(description="The memory content to store")],
    memory_type: Annotated[
        str, Field(description="Type: conversation, status, decision, preference, doc, note")
    ],
    tags: Annotated[
        Optional[List[str]],
        Field(description="Optional tags for filtering (e.g., ['auth', 'backend'])"),
    ] = None,
    ttl: Annotated[
        str, Field(description="Time-to-live: session, day, week, month, permanent (default)")
    ] = "permanent",
) -> Dict[str, Any]:
    """Store a new memory with automatic embedding."""
    from codegrok_mcp.indexing.memory_retriever import MemoryRetriever
    from codegrok_mcp.core.models import MemoryType

    state = get_state()

    if not state.codebase_path:
        raise ToolError("No codebase loaded. Use 'learn' first to set project context.")

    # Initialize memory retriever if needed
    if state.memory_retriever is None:
        paths = _get_codegrok_paths(state.codebase_path)
        state.memory_retriever = MemoryRetriever(
            project_path=str(state.codebase_path),
            persist_path=str(paths["chroma_path"]),
            verbose=False,
        )

    # Validate memory_type
    valid_types = [t.value for t in MemoryType]
    if memory_type not in valid_types:
        raise ToolError(f"Invalid memory_type '{memory_type}'. Must be one of: {valid_types}")

    # Store memory
    memory = state.memory_retriever.remember(
        content=content, memory_type=memory_type, tags=tags or [], ttl=ttl
    )

    return {
        "success": True,
        "memory_id": memory.id,
        "message": f"Stored {memory_type} memory",
        "tags": memory.tags,
    }


@mcp.tool(
    name="recall",
    description="""Retrieve memories using semantic search. Requires 'learn' first to set project context.

Searches across all stored memories (conversations, status, decisions, etc.)
and returns the most relevant matches.

Filters:
- memory_type: Filter by type (conversation, status, decision, preference, doc, note)
- tags: Filter by tags (returns memories matching ANY of the specified tags)
- time_range: Filter by age (today, week, month, all)
- n_results: Number of results to return (default: 5, max: 20)

Examples:
- What did we discuss about auth: recall(query="authentication implementation")
- Get current blockers: recall(query="what is blocked", memory_type="status")
- User preferences: recall(query="coding preferences", memory_type="preference")
- Recent decisions: recall(query="architecture decisions", time_range="week")
""",
    annotations=ToolAnnotations(
        readOnlyHint=True,  # Only reads from memory store
        idempotentHint=True,  # Same query = same results
        openWorldHint=False,  # Only accesses local ChromaDB
    ),
)
def recall(
    query: Annotated[str, Field(description="Natural language search query")],
    memory_type: Annotated[
        Optional[str],
        Field(description="Filter by type: conversation, status, decision, preference, doc, note"),
    ] = None,
    tags: Annotated[Optional[List[str]], Field(description="Filter by tags (matches any)")] = None,
    n_results: Annotated[int, Field(description="Number of results (default: 5)", ge=1, le=20)] = 5,
    time_range: Annotated[
        Optional[str], Field(description="Time filter: today, week, month, all")
    ] = None,
) -> Dict[str, Any]:
    """Retrieve memories using semantic search."""
    from codegrok_mcp.indexing.memory_retriever import MemoryRetriever
    from codegrok_mcp.core.models import MemoryType

    state = get_state()

    if not state.codebase_path:
        raise ToolError("No codebase loaded. Use 'learn' first to set project context.")

    # Initialize memory retriever if needed
    if state.memory_retriever is None:
        paths = _get_codegrok_paths(state.codebase_path)
        state.memory_retriever = MemoryRetriever(
            project_path=str(state.codebase_path),
            persist_path=str(paths["chroma_path"]),
            verbose=False,
        )

    # Validate memory_type if provided
    if memory_type:
        valid_types = [t.value for t in MemoryType]
        if memory_type not in valid_types:
            raise ToolError(f"Invalid memory_type '{memory_type}'. Must be one of: {valid_types}")

    # Search memories
    memories = state.memory_retriever.recall(
        query=query, memory_type=memory_type, tags=tags, n_results=n_results, time_range=time_range
    )

    return {"success": True, "count": len(memories), "memories": memories}


@mcp.tool(
    name="forget",
    description="""⚠️ DESTRUCTIVE: Remove memories matching specified criteria. Requires 'learn' first.

Can delete by:
- Specific memory ID
- Memory type (all of that type)
- Tags (any matching tag)
- Age (older than specified duration)

Examples:
- Forget specific memory: forget(memory_id="abc-123")
- Clear old conversations: forget(memory_type="conversation", older_than="30d")
- Remove by tag: forget(tags=["deprecated", "outdated"])
""",
    annotations=ToolAnnotations(
        readOnlyHint=False,  # Deletes from ChromaDB
        destructiveHint=True,  # ⚠️ PERMANENTLY DELETES data
        idempotentHint=True,  # Re-calling same filter is safe
        openWorldHint=False,  # Only local storage
    ),
)
def forget(
    memory_id: Annotated[Optional[str], Field(description="Specific memory ID to delete")] = None,
    memory_type: Annotated[
        Optional[str], Field(description="Delete all memories of this type")
    ] = None,
    tags: Annotated[
        Optional[List[str]], Field(description="Delete memories with any of these tags")
    ] = None,
    older_than: Annotated[
        Optional[str], Field(description="Delete memories older than: 1d, 7d, 30d, 1y")
    ] = None,
) -> Dict[str, Any]:
    """Remove memories matching criteria."""
    from codegrok_mcp.indexing.memory_retriever import MemoryRetriever

    state = get_state()

    if not state.codebase_path:
        raise ToolError("No codebase loaded. Use 'learn' first to set project context.")

    if state.memory_retriever is None:
        paths = _get_codegrok_paths(state.codebase_path)
        state.memory_retriever = MemoryRetriever(
            project_path=str(state.codebase_path),
            persist_path=str(paths["chroma_path"]),
            verbose=False,
        )

    if not any([memory_id, memory_type, tags, older_than]):
        raise ToolError(
            "Must specify at least one filter: memory_id, memory_type, tags, or older_than"
        )

    result = state.memory_retriever.forget(
        memory_id=memory_id, memory_type=memory_type, tags=tags, older_than=older_than
    )

    return {
        "success": True,
        "deleted": result["deleted"],
        "message": f"Deleted {result['deleted']} memories",
    }


@mcp.tool(
    name="memory_stats",
    description="""Get statistics about stored memories for the current project.

Returns: total memories, count by type, count by TTL, oldest/newest memory dates.""",
    annotations=ToolAnnotations(
        readOnlyHint=True,  # Only reads metadata
        idempotentHint=True,  # Same state = same results
        openWorldHint=False,  # Only accesses local state
    ),
)
def memory_stats() -> Dict[str, Any]:
    """Get memory statistics."""
    from codegrok_mcp.indexing.memory_retriever import MemoryRetriever

    state = get_state()

    if not state.codebase_path:
        return {"loaded": False, "message": "No codebase loaded. Use 'learn' first."}

    if state.memory_retriever is None:
        paths = _get_codegrok_paths(state.codebase_path)
        state.memory_retriever = MemoryRetriever(
            project_path=str(state.codebase_path),
            persist_path=str(paths["chroma_path"]),
            verbose=False,
        )

    stats = state.memory_retriever.get_stats()

    return {"loaded": True, "project": str(state.codebase_path), **stats}


def main():  # pragma: no cover
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
