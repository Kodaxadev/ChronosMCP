# chronos/memory_tools.py
# Responsibility: Overflow MCP tool registrations for advanced memory operations.
# Owns: update_memory, related_memories.
#
# Separated from tools.py to keep modules under the 400-line limit.
# Registration: called once from tools.register() after singletons exist.
#
# v4.0: related_memories replaces the v3.x query_similar_memories tool.
# The old tool compared structural features (length, tag count, project
# hash) in hyperbolic space — which surfaced same-length memories, not
# related ones. related_memories uses the memory's own content as a BM25
# query, which finds memories that share distinctive vocabulary.

from mcp.server.fastmcp import FastMCP

from chronos.db import get_db
from chronos.search import search_memories


def register_memory_tools(fastmcp: FastMCP, mem_store) -> None:
    """
    Register extended memory MCP tools on the given FastMCP instance.
    mem_store: MemoryStore singleton.
    """

    @fastmcp.tool()
    async def update_memory(memory_id: str, content: str) -> dict:
        """
        Replace the content of an existing memory and re-index it.
        Use this to correct, expand, or clarify a previously stored memory
        without losing its original creation timestamp or breaking time-travel
        (the prior content is snapshotted automatically).

        memory_id: the id returned by remember() or recall().
        content:   new free-text content to replace the existing entry.

        Returns: {id, status, token_estimate}
        status is 'updated' on success, 'error' if not found or forgotten
        (restore_memory first, or remember() the correction as a new entry).
        """
        try:
            return mem_store.update(memory_id, content)
        except ValueError as exc:
            return {"id": memory_id, "status": "error", "detail": str(exc)}

    @fastmcp.tool()
    async def related_memories(
        memory_id: str,
        k: int = 5,
        project: str = None,
    ) -> dict:
        """
        Find memories related to a given memory — 'more like this'.

        Uses the memory's own content as a relevance query (BM25 over
        distinctive terms), so results share vocabulary and topic with the
        source memory. Use recall() when you have a question; use this when
        you have a memory and want its neighbours.

        memory_id: id of a stored memory (from remember() or recall()).
        k:         number of related memories (default 5, max 20).
        project:   optional — restrict results to this project only.

        Returns: {results: [{memory_id, project, score, content_preview}],
                  count, source_id}
        """
        k = max(1, min(k, 20))

        with get_db() as db:
            row = db.execute(
                "SELECT content, forgotten FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
            if not row:
                return {
                    "results": [], "count": 0, "source_id": memory_id,
                    "error": f"Memory '{memory_id}' not found",
                }
            if row["forgotten"]:
                return {
                    "results": [], "count": 0, "source_id": memory_id,
                    "error": f"Memory '{memory_id}' is forgotten — restore it first",
                }

            rows = search_memories(
                db, row["content"], project=project, k=k, exclude_id=memory_id
            )

        results = []
        for r in rows:
            preview = r["content"]
            if len(preview) > 120:
                preview = preview[:120] + "…"
            results.append({
                "memory_id":       r["id"],
                "project":         r["project"],
                "score":           round(r["score"], 4),
                "content_preview": preview,
            })

        return {
            "results":   results,
            "count":     len(results),
            "source_id": memory_id,
        }
