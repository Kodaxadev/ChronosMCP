# chronos/memory_tools.py
# Responsibility: Overflow MCP tool registrations for advanced memory operations.
#
# Separated from tools.py because tools.py reached the 400-line module limit.
# This module owns: update_memory, query_similar_memories.
#
# Registration: called once from chronos_mcp.py via register_memory_tools()
# after all singletons are initialised.
#
# Design note: tools are registered via closure over injected singletons —
# no module-level globals, no circular imports, no fragile lazy imports.

from mcp.server.fastmcp import FastMCP


def register_memory_tools(fastmcp: FastMCP, ms, me) -> None:
    """
    Register extended memory MCP tools on the given FastMCP instance.
    Called once from chronos_mcp.py after init_db() and all loads complete.

    fastmcp: the shared FastMCP server instance (same object as in tools.py)
    ms:      MemoryStore instance
    me:      MemoryEmbedder instance (or None if running without vector index)
    """
    _register(fastmcp, ms, me)


def _register(mcp: FastMCP, mem_store, mem_embedder) -> None:
    """
    Close over injected singletons. This avoids all module-level state and
    eliminates the circular-import trap from the previous implementation.
    """

    @mcp.tool()
    async def update_memory(memory_id: str, content: str) -> dict:
        """
        Replace the content of an existing memory and re-index it.
        Use this to correct, expand, or clarify a previously stored memory
        without losing its original creation timestamp or breaking time-travel.

        memory_id: the id returned by remember() or recall().
        content:   new free-text content to replace the existing entry.

        Returns: {id, status, token_estimate}
        status is 'updated' on success, 'error' if memory not found or forgotten.
        Forgotten memories cannot be updated (call remember() with corrected
        content instead — this preserves audit integrity).
        """
        try:
            return mem_store.update(memory_id, content)
        except ValueError as exc:
            return {"id": memory_id, "status": "error", "detail": str(exc)}

    @mcp.tool()
    async def query_similar_memories(
        memory_id: str,
        k: int = 5,
        project: str = None,
    ) -> dict:
        """
        Find memories that are structurally similar to a given memory using
        hyperbolic distance in the content embedding space.

        This bridges the remember/recall (TF-IDF) and add_event/query_similar
        (hyperbolic) pipelines. Use it to find related memories when you know
        one memory and want to discover others with similar content structure
        and project affinity.

        memory_id: id of a stored memory (from remember() or recall()).
        k:         number of similar memories to return (default 5, max 20).
        project:   optional — restrict results to this project only.

        Returns: {results: [{memory_id, distance, content_preview}], count, source_id}

        Note: similarity is structural (content length, tag count, project,
        recency) — NOT semantic. Use recall() with a descriptive query for
        semantic/keyword similarity.
        """
        if mem_embedder is None:
            return {
                "results":   [],
                "count":     0,
                "source_id": memory_id,
                "error": "Vector index not available — server started without MemoryEmbedder",
            }

        k       = max(1, min(k, 20))
        results = mem_embedder.nearest(memory_id, k=k, project=project)

        enriched = []
        for r in results:
            preview = mem_store.tfidf.get_text(r["memory_id"])
            enriched.append({
                "memory_id":       r["memory_id"],
                "distance":        r["distance"],
                "content_preview": (preview[:120] + "…") if len(preview) > 120 else preview,
            })

        return {
            "results":   enriched,
            "count":     len(enriched),
            "source_id": memory_id,
        }
