# chronos/consolidation_scan.py
# Responsibility: read-only analysis phases of dream consolidation —
# Orient (health snapshot), Gather (duplicate detection), and the scans
# that feed Flag and Prune decisions.
#
# Extracted from consolidation.py to keep modules under the 400-line limit.
# Nothing in this module writes to the database; ConsolidationEngine
# (consolidation.py) owns all mutations.

from typing import List, Optional

from chronos.beliefs import CONFIDENCE_DEFAULT, STABILITY_DEFAULT, BeliefEngine
from chronos.consolidation_config import (
    DUPLICATE_THRESHOLD,
    PRUNE_CONFIDENCE_THRESHOLD,
    PRUNE_RETENTION_THRESHOLD,
    RETENTION_WARNING_THRESHOLD,
    STALE_DAYS_THRESHOLD,
)
from chronos.db import get_db
from chronos.search import cosine_similarity, token_counter


def _active_rows(db, columns: str, project: Optional[str]):
    """Fetch the given columns for all non-forgotten memories, optionally
    scoped to a project."""
    if project:
        return db.execute(
            f"SELECT {columns} FROM memories WHERE forgotten = 0 AND project = ?",
            (project,),
        ).fetchall()
    return db.execute(
        f"SELECT {columns} FROM memories WHERE forgotten = 0"
    ).fetchall()


def orient(beliefs: BeliefEngine, project: Optional[str] = None) -> dict:
    """
    Phase 1 — snapshot current memory health metrics before consolidation.
    Returns counts, average confidence, retention distribution, and staleness.
    """
    with get_db() as db:
        rows = _active_rows(
            db, "id, confidence, stability, last_reviewed, created_at", project
        )

    if not rows:
        return {
            "total_active": 0,
            "avg_confidence": 0.0,
            "retention_buckets": {"high": 0, "medium": 0, "low": 0, "critical": 0},
            "stale_count": 0,
        }

    confidences = []
    retention_buckets = {"high": 0, "medium": 0, "low": 0, "critical": 0}
    stale_count = 0

    for row in rows:
        conf = row[1] if row[1] is not None else CONFIDENCE_DEFAULT
        confidences.append(conf)

        stability = row[2] if row[2] is not None else STABILITY_DEFAULT
        review_ts = row[3] or row[4]
        days = beliefs.days_since(review_ts)
        retention = beliefs.compute_retention(stability, days)

        if retention >= 0.7:
            retention_buckets["high"] += 1
        elif retention >= 0.4:
            retention_buckets["medium"] += 1
        elif retention >= PRUNE_RETENTION_THRESHOLD:
            retention_buckets["low"] += 1
        else:
            retention_buckets["critical"] += 1

        if days > STALE_DAYS_THRESHOLD:
            stale_count += 1

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    return {
        "total_active": len(rows),
        "avg_confidence": round(avg_conf, 4),
        "retention_buckets": retention_buckets,
        "stale_count": stale_count,
    }


def gather_duplicates(project: Optional[str] = None) -> List[dict]:
    """
    Phase 2 — find pairs of memories with cosine similarity >= DUPLICATE_THRESHOLD.
    Token counters are precomputed once per memory; the comparison is O(N²)
    pairs, acceptable for on-demand maintenance at personal scale
    (1,000 memories ≈ 500k cheap counter dot-products).

    Returns list of {id_a, id_b, similarity, preview_a, preview_b}.
    """
    with get_db() as db:
        rows = _active_rows(db, "id, content", project)

    if len(rows) < 2:
        return []

    entries = [(r[0], r[1], token_counter(r[1])) for r in rows]

    pairs = []
    for i in range(len(entries)):
        id_a, content_a, counter_a = entries[i]
        for j in range(i + 1, len(entries)):
            id_b, content_b, counter_b = entries[j]
            sim = cosine_similarity(counter_a, counter_b)
            if sim >= DUPLICATE_THRESHOLD:
                pairs.append({
                    "id_a": id_a,
                    "id_b": id_b,
                    "similarity": round(sim, 4),
                    "preview_a": content_a[:80],
                    "preview_b": content_b[:80],
                })

    return pairs


def flag_low_retention(
    beliefs: BeliefEngine, project: Optional[str] = None
) -> List[dict]:
    """
    Find memories whose FSRS retention has dropped below the warning threshold.
    Read-only — feeds the 'needs_review' section of the Consolidate report.
    """
    flagged = []
    with get_db() as db:
        rows = _active_rows(
            db, "id, stability, last_reviewed, created_at, content", project
        )

    for row in rows:
        mem_id = row[0]
        stability = row[1] if row[1] is not None else STABILITY_DEFAULT
        review_ts = row[2] or row[3]
        days = beliefs.days_since(review_ts)
        retention = beliefs.compute_retention(stability, days)

        if retention < RETENTION_WARNING_THRESHOLD:
            flagged.append({
                "memory_id": mem_id,
                "retention": round(retention, 4),
                "stability": round(stability, 4),
                "days_since_review": round(days, 1),
                "preview": row[4][:80] if row[4] else "",
            })

    return flagged


def find_prune_candidates(
    beliefs: BeliefEngine, project: Optional[str] = None
) -> List[dict]:
    """
    Find memories that are BOTH low-confidence AND low-retention — effectively
    abandoned. Both thresholds must be met, which prevents pruning high-value
    memories that simply haven't been reviewed recently.
    """
    candidates = []
    with get_db() as db:
        rows = _active_rows(
            db,
            "id, confidence, stability, last_reviewed, created_at, content",
            project,
        )

    for row in rows:
        mem_id = row[0]
        conf = row[1] if row[1] is not None else CONFIDENCE_DEFAULT
        stability = row[2] if row[2] is not None else STABILITY_DEFAULT
        review_ts = row[3] or row[4]
        days = beliefs.days_since(review_ts)
        retention = beliefs.compute_retention(stability, days)

        if conf < PRUNE_CONFIDENCE_THRESHOLD and retention < PRUNE_RETENTION_THRESHOLD:
            candidates.append({
                "memory_id": mem_id,
                "confidence": round(conf, 4),
                "retention": round(retention, 4),
                "days_since_review": round(days, 1),
                "preview": row[5][:80] if row[5] else "",
            })

    return candidates
