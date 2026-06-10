# chronos/compression.py
# Responsibility: Token-budget compression for recall results.
#
# Three tiers of compression, independently derived from Chronos's own
# token_estimate data and typical MCP tool response budgets:
#
#   Tier 1 — Trim:     Truncate individual memories exceeding a per-result
#                       token budget. Preserves the first N words + ellipsis.
#   Tier 2 — Compact:  When total_tokens exceeds a response budget, drop
#                       the lowest-scored results until the budget fits.
#   Tier 3 — Summarize: Replace long content with a truncated extract +
#                        metadata tag, preserving id/score/confidence for
#                        the caller to request full content if needed.
#
# Threshold derivation (NOT from any external source):
#   - Per-result trim: 150 tokens (~200 words). Rationale: most useful memory
#     content is in the first 1-2 sentences. Long memories are often paste dumps.
#   - Response budget: 4000 tokens. Rationale: MCP tool responses should leave
#     room for Claude to reason about the results. 4K is ~30% of a typical
#     8K working context window. Adjustable via parameter.
#   - Summarize threshold: 300 tokens per result. At this size, the content
#     is long enough that a truncated extract saves meaningful space.
#
# Design: Stateless functions operating on recall result dicts. No DB access,
# no side effects. Called from the recall pipeline in memory.py.

from typing import List, Optional

# --- Independently derived thresholds ---
# Per-result: trim content over this many estimated tokens
TRIM_TOKENS = 150
# Response-level: total tokens budget for a recall response
RESPONSE_BUDGET_DEFAULT = 4000
# Per-result: above this, replace with extract + metadata stub
SUMMARIZE_TOKENS = 300
# Approximate words per token (inverse of TFIDFIndex.TOKENS_PER_WORD)
WORDS_PER_TOKEN = 1.33  # 1/0.75


def _estimate_tokens(text: str) -> int:
    """Quick token estimate matching TFIDFIndex.estimate_tokens()."""
    return max(1, round(len(text.split()) * 0.75))


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximately max_tokens, ending at a word boundary."""
    max_words = int(max_tokens * WORDS_PER_TOKEN)
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " [...]"


# ---------------------------------------------------------------------------
# Tier 1: Trim — truncate individual oversized results
# ---------------------------------------------------------------------------

def tier1_trim(results: List[dict], max_tokens_per_result: int = TRIM_TOKENS) -> List[dict]:
    """
    Trim individual memory content that exceeds the per-result token limit.
    Modifies content in-place (on copies). Adds 'trimmed' flag if truncated.
    Preserves all other fields (id, score, confidence, etc.).
    """
    trimmed = []
    for r in results:
        entry = dict(r)  # shallow copy
        tok = _estimate_tokens(entry.get("content", ""))
        if tok > max_tokens_per_result:
            entry["content"] = _truncate_to_tokens(entry["content"], max_tokens_per_result)
            entry["token_estimate"] = _estimate_tokens(entry["content"])
            entry["trimmed"] = True
        trimmed.append(entry)
    return trimmed


# ---------------------------------------------------------------------------
# Tier 2: Compact — drop lowest-scored results to fit budget
# ---------------------------------------------------------------------------

def tier2_compact(
    results: List[dict],
    budget: int = RESPONSE_BUDGET_DEFAULT,
    overhead: int = 40,
) -> List[dict]:
    """
    Drop the lowest-scored results until total_tokens fits within budget.
    Results must already be sorted by score descending (recall guarantees this).

    Returns a (possibly shorter) list. Does NOT modify content — only drops
    entire results from the tail. If even the first result exceeds budget,
    returns just that one result (never returns empty).
    """
    if not results:
        return results

    # Calculate running total from highest-scored down
    running = overhead
    kept = []
    for r in results:
        tok = r.get("token_estimate", _estimate_tokens(r.get("content", "")))
        if running + tok > budget and kept:
            break  # Budget exceeded — stop adding
        running += tok
        kept.append(r)

    return kept


# ---------------------------------------------------------------------------
# Tier 3: Summarize — replace long content with extract + stub
# ---------------------------------------------------------------------------

def tier3_summarize(
    results: List[dict],
    threshold: int = SUMMARIZE_TOKENS,
) -> List[dict]:
    """
    For results whose content exceeds the summarize threshold, replace
    with a short extract preserving the first ~50 tokens + metadata stub.
    The stub includes the original token count so the caller can request
    the full memory via its id if needed.

    This is the most aggressive compression tier — use when the response
    is still too large after Tier 1 + Tier 2.
    """
    summarized = []
    for r in results:
        entry = dict(r)
        tok = _estimate_tokens(entry.get("content", ""))
        if tok > threshold:
            original_tokens = tok
            entry["content"] = _truncate_to_tokens(entry["content"], 50)
            entry["token_estimate"] = _estimate_tokens(entry["content"])
            entry["summarized"] = True
            entry["original_tokens"] = original_tokens
        summarized.append(entry)
    return summarized


# ---------------------------------------------------------------------------
# Unified compression pipeline
# ---------------------------------------------------------------------------

def compress_results(
    results: List[dict],
    budget: Optional[int] = None,
    overhead: int = 40,
) -> dict:
    """
    Apply compression tiers progressively until the response fits budget.

    1. Always apply Tier 1 (trim oversized individual results)
    2. If total still exceeds budget, apply Tier 2 (drop lowest-scored)
    3. If total STILL exceeds budget, apply Tier 3 (summarize remaining)

    budget: total token budget for the response. None = no compression
            beyond Tier 1 trimming (which always runs for hygiene).

    Returns: {
        results: compressed result list,
        total_tokens: new total,
        compression_applied: list of tier names applied,
    }
    """
    if not results:
        return {
            "results": results,
            "total_tokens": overhead,
            "compression_applied": [],
        }

    applied = []
    effective_budget = budget or RESPONSE_BUDGET_DEFAULT

    # Tier 1: Always trim oversized individual results
    compressed = tier1_trim(results)
    applied.append("trim")

    # Check total
    total = overhead + sum(
        r.get("token_estimate", _estimate_tokens(r.get("content", "")))
        for r in compressed
    )

    # Tier 2: If over budget, drop tail results
    if budget is not None and total > effective_budget:
        compressed = tier2_compact(compressed, effective_budget, overhead)
        applied.append("compact")
        total = overhead + sum(
            r.get("token_estimate", _estimate_tokens(r.get("content", "")))
            for r in compressed
        )

    # Tier 3: If STILL over budget, summarize remaining
    if budget is not None and total > effective_budget:
        compressed = tier3_summarize(compressed)
        applied.append("summarize")
        total = overhead + sum(
            r.get("token_estimate", _estimate_tokens(r.get("content", "")))
            for r in compressed
        )

    return {
        "results": compressed,
        "total_tokens": total,
        "compression_applied": applied,
    }
