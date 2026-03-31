# chronos/tools.py
# Responsibility: MCP tool orchestration — memory tools, stats resource,
#                 and the register() entry point that wires all sub-modules.
#
# Module split rationale (all files kept under 400-line hard limit):
#   graph_tools.py    — add_event, query_similar, add_constraint
#   analysis_tools.py — analyze_causal, suggest_next_tasks, analyze_structure
#   memory_tools.py   — update_memory, query_similar_memories
#   tools.py (this)   — remember, recall, forget, query_at, stats + registration

from datetime import datetime
from typing import List

from mcp.server.fastmcp import FastMCP

from chronos.analysis_tools import register_analysis_tools
from chronos.db import get_db
from chronos.geometry import calculate_dimension
from chronos.graph_tools import register_graph_tools
from chronos.memory_tools import register_memory_tools

# ---------------------------------------------------------------------------
# Singletons — injected by chronos_mcp.py after init_db() + load_from_db()
# ---------------------------------------------------------------------------
# These module-level refs are used only by get_stats (resource handler below)
# which needs embedder.dim and mem_store.tfidf.doc_count().
# All tool handlers use closure-injected singletons via their sub-modules.

_embedder  = None   # HyperbolicEmbedder — for stats only
_mem_store = None   # MemoryStore — for memory tools + stats

# Exposed publicly so memory_tools.py (old pattern) tests can still reference mcp
mcp = None


def register(
    fastmcp: FastMCP,
    emb, csl, slv, stru, mem, mem_emb=None,
) -> None:
    """
    Called once from chronos_mcp.py after all singletons are initialized.
    Delegates tool registration to each sub-module.

    fastmcp: the shared FastMCP server instance
    emb:     HyperbolicEmbedder
    csl:     CausalAnalyzer
    slv:     ConstraintSolver
    stru:    StructureAnalyzer
    mem:     MemoryStore
    mem_emb: MemoryEmbedder (optional — enables query_similar_memories)
    """
    global mcp, _embedder, _mem_store
    mcp        = fastmcp
    _embedder  = emb
    _mem_store = mem

    register_graph_tools(fastmcp, emb)
    register_analysis_tools(fastmcp, csl, slv, stru)
    _register_memory_tools(fastmcp, mem)
    register_memory_tools(fastmcp, mem, mem_emb)


# ---------------------------------------------------------------------------
# Memory tools — registered inline to keep mem_store in closure scope
# ---------------------------------------------------------------------------

def _register_memory_tools(mcp_inst: FastMCP, mem_store) -> None:

    @mcp_inst.tool()
    async def remember(
        content: str,
        project: str = "default",
        tags: List[str] = None,
    ) -> dict:
        """
        Store a memory. Use this to save anything worth remembering across
        sessions: decisions, code snippets, findings, context, summaries.

        content: free-text string — no structure required.
        project: logical grouping (e.g. 'auth-service', 'sprint-12').
                 Defaults to 'default'. Use consistent names to enable
                 project-scoped recall.
        tags:    optional list of keyword labels for future filtering.

        Returns: {id, project, token_estimate, indexed_terms, embedded}
        token_estimate is how many tokens this memory will consume when recalled.
        embedded=True means the memory was also added to the vector index.
        """
        return mem_store.remember(content, project=project, tags=tags or [])

    @mcp_inst.tool()
    async def recall(
        query: str,
        project: str = None,
        k: int = 5,
        recency_weight: float = 0.3,
    ) -> dict:
        """
        Retrieve the most relevant memories for a query.
        Call this before starting work on any topic to load relevant context.

        query:          natural language question or topic, e.g. 'authentication flow'
        project:        optional — restrict to memories from this project only.
        k:              number of results to return (default 5, max 20).
                        Check total_tokens in the response before requesting more.
        recency_weight: 0.0–1.0, default 0.3. Controls how much recent memories
                        are boosted. 0.0 = pure TF-IDF ranking. 1.0 = strong
                        recency preference. Adjust down when you want the most
                        historically relevant memories regardless of age.

        Returns:
          results:      ranked list of {id, project, content, score, token_estimate}
          total_tokens: estimated tokens all results will consume in context
          count:        number of results returned
        """
        k              = max(1, min(k, 20))
        recency_weight = max(0.0, min(1.0, recency_weight))
        return mem_store.recall(query, project=project, k=k,
                                recency_weight=recency_weight)

    @mcp_inst.tool()
    async def forget(memory_id: str, reason: str = "manual") -> dict:
        """
        Soft-delete a memory so it no longer appears in recall results.
        The record is retained for audit and time-travel queries.

        memory_id: the id returned by remember() or recall().
        reason:    optional explanation for the deletion.

        Returns: {id, status, reason}
        status is 'forgotten', 'not_found', or 'already_forgotten'.
        """
        return mem_store.forget(memory_id, reason=reason)

    @mcp_inst.tool()
    async def query_at(
        query: str,
        timestamp: str,
        project: str = None,
        k: int = 5,
    ) -> dict:
        """
        Time-travel recall: retrieve memories as they existed at a past timestamp.
        Useful for understanding what was known at the start of a sprint,
        before a major decision, or at any specific point in a project.

        query:     natural language question, same as recall().
        timestamp: ISO 8601 datetime string, e.g. '2026-03-01T00:00:00'.
                   Memories created after this time are excluded.
        project:   optional project filter.
        k:         number of results (default 5).

        Returns same shape as recall() plus 'as_of' field confirming the snapshot time.
        """
        k = max(1, min(k, 20))
        return mem_store.query_at(query, timestamp=timestamp, project=project, k=k)

    @mcp_inst.resource("chronos://stats")
    async def get_stats() -> str:
        """System statistics: memory counts, embedding config, schema version."""
        with get_db() as db:
            n_events      = db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            n_nodes       = db.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            n_causal      = db.execute("SELECT COUNT(*) FROM causal_results").fetchone()[0]
            n_constraints = db.execute("SELECT COUNT(*) FROM constraints").fetchone()[0]
            n_tombstones  = db.execute("SELECT COUNT(*) FROM tombstones").fetchone()[0]
            n_memories    = db.execute(
                "SELECT COUNT(*) FROM memories WHERE forgotten = 0"
            ).fetchone()[0]
            n_forgotten   = db.execute(
                "SELECT COUNT(*) FROM memories WHERE forgotten = 1"
            ).fetchone()[0]
            n_mem_vectors = db.execute(
                "SELECT COUNT(*) FROM memory_vectors"
            ).fetchone()[0]

        target_dim = calculate_dimension(n_nodes)
        return (
            f"Memories (active):    {n_memories}\n"
            f"Memories (forgotten): {n_forgotten}\n"
            f"Memory vectors:       {n_mem_vectors}\n"
            f"TF-IDF indexed:       {mem_store.tfidf.doc_count()}\n"
            f"Events:               {n_events}\n"
            f"Active nodes:         {n_nodes - n_tombstones}\n"
            f"Tombstoned nodes:     {n_tombstones}\n"
            f"Causal analyses:      {n_causal}\n"
            f"Constraints:          {n_constraints}\n"
            f"Embedding dim:        {_embedder.dim} (target: {target_dim})\n"
            f"Schema version:       3.1"
        )
