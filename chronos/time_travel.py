# chronos/time_travel.py
# Responsibility: Time-travel queries — reconstruct memory state at a past timestamp.
#
# This module owns query_at() and the version-resolution logic needed to
# show content as it existed at any point. Snapshot ranking uses the same
# BM25 + porter pipeline as live recall (search.rank_snapshot), so scores
# are comparable across the two tools.

from datetime import datetime
from typing import List, Optional

from chronos.db import get_db
from chronos.search import estimate_tokens, rank_snapshot

# Approximate tokens consumed by the recall response wrapper
_RECALL_OVERHEAD_TOKENS = 40

# Max IDs per SQLite IN clause batch (stay under SQLITE_MAX_VARIABLE_NUMBER)
_BATCH = 900


def query_at(
    query: str,
    timestamp: str,
    project: Optional[str] = None,
    k: int = 5,
) -> dict:
    """
    Reconstruct which memories existed at `timestamp` and rank them
    against `query` using an ephemeral BM25 index over that snapshot.

    timestamp: ISO 8601 string, e.g. '2026-03-01T00:00:00'
    Memories created after `timestamp` are excluded.
    Memories forgotten at or before `timestamp` are excluded.
    Content edited since is resolved back through memory_versions.

    Returns same shape as recall() plus `as_of` field.
    """
    if not query or not query.strip():
        return {
            "results": [], "total_tokens": 0,
            "count": 0, "query": query, "as_of": timestamp,
        }

    # Validate timestamp is parseable ISO 8601 before using in SQL.
    try:
        parsed_ts = datetime.fromisoformat(timestamp)
        # Strip timezone info for consistent lexicographic comparison
        # against tz-naive stored timestamps.
        if parsed_ts.tzinfo is not None:
            timestamp = parsed_ts.replace(tzinfo=None).isoformat()
    except (ValueError, TypeError):
        return {
            "results": [], "total_tokens": 0, "count": 0,
            "query": query, "as_of": timestamp,
            "error": f"Invalid ISO 8601 timestamp: '{timestamp}'",
        }

    with get_db() as db:
        if project:
            rows = db.execute(
                """SELECT id, project, content, forgotten, updated_at
                   FROM memories
                   WHERE created_at <= ?
                     AND project = ?""",
                (timestamp, project),
            ).fetchall()
        else:
            rows = db.execute(
                """SELECT id, project, content, forgotten, updated_at
                   FROM memories
                   WHERE created_at <= ?""",
                (timestamp,),
            ).fetchall()

        # Load version history for candidate memories only.
        # Batched to stay under SQLite's variable number limit.
        candidate_ids = [r[0] for r in rows]
        version_rows = _fetch_versions_batched(db, candidate_ids, timestamp)

    # Build version lookup: memory_id -> list of (content, valid_from, valid_to)
    versions: dict = {}
    for vr in version_rows:
        vid, vcontent, vfrom, vto = vr
        versions.setdefault(vid, []).append((vcontent, vfrom, vto))

    # Exclude memories that were forgotten at or before the timestamp
    snapshot: List[tuple] = []
    for r in rows:
        mem_id, proj, content, forgotten, updated_at = r
        if forgotten and updated_at <= timestamp:
            continue
        # Resolve historical content: find the version active at timestamp.
        if mem_id in versions:
            for vcontent, vfrom, vto in versions[mem_id]:
                if vfrom <= timestamp < vto:
                    content = vcontent
                    break
        snapshot.append((mem_id, content, proj))

    if not snapshot:
        return {
            "results": [], "total_tokens": _RECALL_OVERHEAD_TOKENS,
            "count": 0, "query": query, "as_of": timestamp,
        }

    # Rank the snapshot with an ephemeral BM25 index
    ranked = rank_snapshot([(s[0], s[1]) for s in snapshot], query, k=k)

    meta         = {s[0]: {"content": s[1], "project": s[2]} for s in snapshot}
    results      = []
    total_tokens = _RECALL_OVERHEAD_TOKENS

    for doc_id, score in ranked:
        content  = meta[doc_id]["content"]
        tok_est  = estimate_tokens(content)
        total_tokens += tok_est
        results.append({
            "id":             doc_id,
            "project":        meta[doc_id]["project"],
            "content":        content,
            "score":          round(score, 5),
            "token_estimate": tok_est,
        })

    return {
        "results":      results,
        "total_tokens": total_tokens,
        "count":        len(results),
        "query":        query,
        "as_of":        timestamp,
    }


def _fetch_versions_batched(db, candidate_ids: list, timestamp: str) -> list:
    """Fetch memory_versions for a list of IDs, batched for SQLite safety."""
    version_rows = []
    for i in range(0, len(candidate_ids), _BATCH):
        batch = candidate_ids[i:i + _BATCH]
        placeholders = ",".join("?" * len(batch))
        version_rows.extend(db.execute(
            f"""SELECT memory_id, content, valid_from, valid_to
                FROM memory_versions
                WHERE memory_id IN ({placeholders})
                ORDER BY valid_from ASC""",
            batch,
        ).fetchall())
    return version_rows
