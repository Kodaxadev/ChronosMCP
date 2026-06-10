# chronos/consolidation_tools.py
# Responsibility: MCP tool registration for dream consolidation.
# Owns: consolidate_memories
#
# Registration: called from tools.register() via register_consolidation_tools().
# Design: closure-injected singleton — no module globals.

from mcp.server.fastmcp import FastMCP


def register_consolidation_tools(mcp: FastMCP, consolidation_engine) -> None:
    """
    Register consolidation MCP tools on the given FastMCP instance.
    consolidation_engine: ConsolidationEngine singleton.
    """

    @mcp.tool()
    async def consolidate_memories(
        project: str = None,
        auto_merge: bool = False,
        auto_prune: bool = False,
    ) -> dict:
        """
        Run a four-phase dream consolidation pass over stored memories.

        Inspired by how biological memory consolidates during REM sleep. Each
        phase builds on the previous one:

        **Phase 1 — Orient:** Snapshot current memory health before changes.
        Returns total active count, average confidence, retention distribution
        (high/medium/low/critical buckets), and number of stale memories.
        Use this as a baseline to measure consolidation effectiveness.

        **Phase 2 — Gather:** Find near-duplicate memory pairs using TF-IDF
        cosine similarity > 85%. Reports pairs with previews and similarity
        scores. No action taken — findings feed into Phase 3.

        **Phase 3 — Consolidate:** Three sub-actions:
          - Merge: if auto_merge=True, merges duplicate pairs (keeps the
            higher-confidence memory, soft-deletes the other, boosts survivor).
          - Decay: memories not reviewed in 30+ days lose 0.05 confidence per
            pass, preventing unverified memories from staying artificially high.
          - Flag: identifies memories with FSRS retention < 30%, meaning they
            are at risk of becoming stale. Review them via review_memory().

        **Phase 4 — Prune:** Identifies memories with BOTH confidence < 0.10
        AND FSRS retention < 0.15. These are effectively abandoned — the system
        has lost faith in them AND they've nearly been forgotten. If
        auto_prune=True, soft-deletes them. If False, reports candidates only.

        Args:
            project:    Scope to a single project name. Omit for all projects.
            auto_merge: False (default) = dry run for duplicates.
                        True = merge detected duplicates automatically.
            auto_prune: False (default) = report prune candidates only.
                        True = auto-forget critically degraded memories.

        Recommended workflow:
        1. Run with auto_merge=False, auto_prune=False to review the report.
        2. If satisfied, re-run with auto_merge=True to clean duplicates.
        3. For aggressive maintenance, add auto_prune=True to also prune.

        Returns a full report with sub-reports keyed by phase name:
        {timestamp, project, phases_run, orient, gather, consolidate, prune}
        """
        return consolidation_engine.consolidate(
            project=project,
            auto_merge=auto_merge,
            auto_prune=auto_prune,
        )
