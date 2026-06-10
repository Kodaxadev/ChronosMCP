# chronos/belief_tools.py
# Responsibility: MCP tool registrations for the cognitive belief subsystem.
# Owns: set_confidence, boost_confidence, weaken_confidence, review_memory,
#        log_search_feedback, search_effectiveness
#
# Registration: called from tools.register() via register_belief_tools().
# Design: closure-injected singletons — no module globals.

from mcp.server.fastmcp import FastMCP


def register_belief_tools(mcp: FastMCP, belief_engine) -> None:
    """
    Register belief/confidence MCP tools on the given FastMCP instance.
    belief_engine: BeliefEngine singleton (from chronos/beliefs.py)
    """

    @mcp.tool()
    async def set_confidence(
        memory_id: str,
        confidence: float,
        reason: str = "manual_adjustment",
    ) -> dict:
        """
        Set the confidence score for a memory directly.

        Use this when you have strong evidence about how trustworthy a memory is.
        Confidence ranges from 0.01 (almost certainly wrong) to 0.99 (almost
        certainly correct). Default for new memories is 0.5 (uncertain).

        memory_id:  id from remember() or recall().
        confidence: target score (0.01–0.99). Values outside range are clamped.
        reason:     why the confidence is being changed (logged for audit).

        Returns: {memory_id, old_confidence, new_confidence, reason}
        """
        return belief_engine.set_confidence(memory_id, confidence, reason)

    @mcp.tool()
    async def boost_confidence(
        memory_id: str,
        reason: str = "confirmed",
    ) -> dict:
        """
        Increase a memory's confidence because evidence confirms it.

        Call this when you encounter information that validates a stored memory.
        Adds +0.10 to the current confidence (clamped at 0.99).

        memory_id: id from remember() or recall().
        reason:    why — e.g. 'confirmed by test results', 'user verified'.

        Returns: {memory_id, old_confidence, new_confidence, reason}
        """
        return belief_engine.boost_confidence(memory_id, reason)

    @mcp.tool()
    async def weaken_confidence(
        memory_id: str,
        reason: str = "refuted",
    ) -> dict:
        """
        Decrease a memory's confidence because evidence contradicts it.

        Call this when you encounter information that challenges a stored memory.
        Subtracts 0.15 from the current confidence (clamped at 0.01).
        The asymmetric delta (weaken > boost) reflects Bayesian caution:
        it's harder to build confidence than to lose it.

        memory_id: id from remember() or recall().
        reason:    why — e.g. 'contradicted by new findings', 'outdated info'.

        Returns: {memory_id, old_confidence, new_confidence, reason}
        """
        return belief_engine.weaken_confidence(memory_id, reason)

    @mcp.tool()
    async def review_memory(
        memory_id: str,
        quality: str = "good",
    ) -> dict:
        """
        Record that a memory was accessed and found useful (FSRS review).

        This strengthens the memory's stability (how long before it fades)
        and adjusts its difficulty score. Call this after using a recalled
        memory in your work — it prevents the memory from decaying.

        memory_id: id from remember() or recall().
        quality:   how useful was this memory?
          - 'easy'  — trivially useful, no effort needed to apply it
          - 'good'  — useful, standard recall (default)
          - 'hard'  — useful but required significant effort to apply

        Returns: {memory_id, quality, old_stability, new_stability,
                  old_difficulty, new_difficulty, review_count}
        """
        return belief_engine.record_review(memory_id, quality)

    @mcp.tool()
    async def get_memory_health(memory_id: str) -> dict:
        """
        Get the full cognitive state of a memory: confidence, FSRS retention,
        stability, difficulty, review history.

        Use this to check whether a memory is still trustworthy before relying
        on it. Low retention means it may be stale; low confidence means
        it may be inaccurate.

        Returns: {memory_id, confidence, stability, difficulty, retention,
                  days_since_review, review_count, forgotten}
        """
        result = belief_engine.get_confidence(memory_id)
        if result is None:
            return {"error": f"Memory '{memory_id}' not found"}
        return result

    @mcp.tool()
    async def log_search_feedback(
        query: str,
        memory_id: str,
        used: bool = True,
    ) -> dict:
        """
        Log whether a recall result was actually used in your work.

        This feeds the meta-learning system. Over time, Chronos uses this
        feedback to understand which types of search results are helpful
        and can recommend search parameter adjustments.

        query:     the recall query that produced this result.
        memory_id: the id of the result.
        used:      True if you used this result, False if you skipped it.

        If used=True, the memory also gets an FSRS review (stability boost).

        Returns: {feedback_id, memory_id, used}
        """
        return belief_engine.log_feedback(query, memory_id, used)

    @mcp.tool()
    async def search_effectiveness(days: int = 30) -> dict:
        """
        Check how well recall has been performing based on search feedback.

        Analyzes feedback from log_search_feedback() over the given time window
        and reports hit rate (what fraction of recalled results were actually used).

        days: number of days to look back (default 30).

        Returns: {total_searches, results_used, hit_rate, window_days, recommendation}
        recommendation is None until 20+ feedback entries exist.
        """
        days = max(1, min(365, days))
        return belief_engine.get_feedback_stats(days)
