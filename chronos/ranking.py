# chronos/ranking.py
# Responsibility: the recall pipeline — candidate retrieval (BM25, optionally
# hybrid with semantic neighbors), Reciprocal Rank Fusion, FSRS-aware
# re-ranking, and response assembly.
#
# Extracted from memory.py (which delegates here) to keep modules under the
# 400-line limit, following the lifecycle.py / time_travel.py pattern.
#
# Fusion: Reciprocal Rank Fusion (RRF), score = Σ 1/(60 + rank) across the
# retriever lists that contain the document. RRF is the standard for hybrid
# lexical+vector search because it needs no score normalization — BM25 and
# cosine values never have to share a scale. k=60 per Cormack et al.

from collections import defaultdict
from datetime import datetime
from typing import List, Optional, Tuple

from chronos.beliefs import CONFIDENCE_DEFAULT, STABILITY_DEFAULT, BeliefEngine
from chronos.compression import compress_results
from chronos.db import get_db
from chronos.search import estimate_tokens, search_memories

# Approximate tokens consumed by the recall response wrapper itself
RECALL_OVERHEAD_TOKENS = 40

# Default recency/retention boost weight (see recall docstring in tools.py)
DEFAULT_RECENCY_WEIGHT = 0.3

# Candidates fetched from each retriever before fusion in hybrid mode.
# Matches semantic.SEARCH_CANDIDATES so neither list dominates by length.
HYBRID_CANDIDATES = 50

_RRF_K = 60


def _recency_factor(created_at_iso: str) -> float:
    """
    1/(1+days_old) — approaches 1 for brand-new, 0 for old. Falls back to
    0.0 on unparseable timestamps. Legacy path when FSRS data is missing.
    """
    try:
        created = datetime.fromisoformat(created_at_iso)
        days = max(
            0.0,
            (datetime.now() - created.replace(tzinfo=None)).total_seconds() / 86400,
        )
        return 1.0 / (1.0 + days)
    except (ValueError, TypeError):
        return 0.0


def _fsrs_boost(stability: float, last_reviewed: str, created_at: str,
                confidence: float) -> float:
    """
    FSRS-aware ranking boost in [0, 1]: retention × confidence_factor.
    High when the memory is both confident AND well-retained; confidence
    alone doesn't override staleness, and vice versa.
    """
    review_ts = last_reviewed or created_at
    days = BeliefEngine.days_since(review_ts)
    stab = stability if stability and stability > 0 else STABILITY_DEFAULT
    retention = BeliefEngine.compute_retention(stab, days)
    conf = confidence if confidence is not None else CONFIDENCE_DEFAULT
    return retention * (0.5 + 0.5 * conf)


def _fetch_rows_by_id(db, memory_ids: List[str]) -> dict:
    """Fetch recall metadata for ids the BM25 pass didn't return."""
    if not memory_ids:
        return {}
    placeholders = ",".join("?" * len(memory_ids))
    rows = db.execute(
        f"""SELECT id, project, content, created_at, confidence,
                   stability, last_reviewed, source
            FROM memories
            WHERE forgotten = 0 AND id IN ({placeholders})""",
        memory_ids,
    ).fetchall()
    return {r["id"]: r for r in rows}


def _fuse_rrf(db, bm25_rows, sem_hits) -> List[Tuple[object, float, Optional[float]]]:
    """
    Merge the BM25 list and the semantic list with Reciprocal Rank Fusion.
    Returns [(row, rrf_score, semantic_similarity|None)] sorted best-first.
    """
    rrf: dict = defaultdict(float)
    for rank, row in enumerate(bm25_rows):
        rrf[row["id"]] += 1.0 / (_RRF_K + rank + 1)
    for rank, (mid, _sim) in enumerate(sem_hits):
        rrf[mid] += 1.0 / (_RRF_K + rank + 1)

    sims = dict(sem_hits)
    rows = {r["id"]: r for r in bm25_rows}
    rows.update(_fetch_rows_by_id(db, [m for m in rrf if m not in rows]))

    fused = [
        (rows[mid], score, sims.get(mid))
        for mid, score in rrf.items()
        if mid in rows  # ids that vanished mid-flight are dropped
    ]
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused


def recall(
    query: str,
    project: Optional[str] = None,
    k: int = 5,
    recency_weight: float = DEFAULT_RECENCY_WEIGHT,
    token_budget: Optional[int] = None,
    semantic=None,
) -> dict:
    """
    Retrieve the k most relevant memories for a query.

    Lexical mode (semantic=None, the default): BM25 over the FTS5 index,
    identical to v4.0 behavior.

    Hybrid mode (semantic=SemanticSearch): BM25 candidates ∪ embedding
    nearest-neighbors, fused with RRF — so "car" can surface a memory about
    an automobile even with zero lexical overlap.

    Both modes then apply the FSRS retention×confidence boost (when
    recency_weight > 0) and opt-in token-budget compression.
    """
    if not query or not query.strip():
        return {"results": [], "total_tokens": 0, "count": 0, "query": query}

    with get_db() as db:
        if semantic is not None:
            bm25_rows = search_memories(db, query, project=project,
                                        k=HYBRID_CANDIDATES)
            sem_hits = semantic.search(db, query, project=project,
                                       k=HYBRID_CANDIDATES)
            candidates = _fuse_rrf(db, bm25_rows, sem_hits)
        else:
            fetch_k = k * 3 if recency_weight > 0 else k
            rows = search_memories(db, query, project=project, k=fetch_k)
            candidates = [(r, r["score"], None) for r in rows]

    boosted = []
    for row, base_score, sim in candidates:
        score = base_score
        if recency_weight > 0:
            if row["stability"] is not None:
                boost = _fsrs_boost(row["stability"], row["last_reviewed"],
                                    row["created_at"], row["confidence"])
            else:
                boost = _recency_factor(row["created_at"])
            score = score * (1.0 + recency_weight * boost)
        boosted.append((row, score, sim))

    boosted.sort(key=lambda x: x[1], reverse=True)
    boosted = boosted[:k]

    results = []
    total_tokens = RECALL_OVERHEAD_TOKENS
    for row, score, sim in boosted:
        tok_est = estimate_tokens(row["content"])
        total_tokens += tok_est
        entry = {
            "id":             row["id"],
            "project":        row["project"],
            "content":        row["content"],
            "score":          round(score, 5),
            "token_estimate": tok_est,
            "source":         row["source"] or "user",
        }
        if row["confidence"] is not None:
            entry["confidence"] = round(row["confidence"], 4)
        if sim is not None:
            entry["semantic_similarity"] = round(sim, 4)
        results.append(entry)

    # Compression is opt-in. No budget → full content, always.
    if token_budget is not None:
        compressed = compress_results(
            results, budget=token_budget, overhead=RECALL_OVERHEAD_TOKENS
        )
        out = {
            "results":             compressed["results"],
            "total_tokens":        compressed["total_tokens"],
            "count":               len(compressed["results"]),
            "query":               query,
            "compression_applied": compressed["compression_applied"],
        }
    else:
        out = {
            "results":      results,
            "total_tokens": total_tokens,
            "count":        len(results),
            "query":        query,
        }

    if semantic is not None:
        out["retrieval"] = "hybrid_rrf"
    return out
