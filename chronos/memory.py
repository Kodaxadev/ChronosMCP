# chronos/memory.py
# Responsibility: High-level memory operations — remember, recall, get, update.
# Lifecycle transitions (forget/restore/purge) live in lifecycle.py;
# time-travel lives in time_travel.py. This is the primary interface
# between MCP tools and the storage layer.
#
# Architecture (v4.0):
#   - memories table   → persistent text storage (db.py)
#   - memories_fts     → BM25 full-text index, trigger-synced (db.py, search.py)
#
# The v3.x in-memory TF-IDF index and structural memory vectors are gone:
# the index now lives in the same SQLite file and commits atomically with
# content writes. MemoryStore holds no state of its own.
#
# Token budget design:
#   recall() reports a token_estimate per result. Compression is applied
#   ONLY when the caller passes token_budget — results are never silently
#   truncated (v3.3 trimmed anything over 150 tokens unconditionally, and
#   offered no way to fetch full content; both fixed in v4.0).

import json
from datetime import datetime
from typing import List, Optional

from chronos import lifecycle
from chronos.beliefs import CONFIDENCE_DEFAULT, STABILITY_DEFAULT, BeliefEngine
from chronos.compression import compress_results
from chronos.db import get_db
from chronos.search import estimate_tokens, search_memories
from chronos.time_travel import query_at as _time_travel_query_at
from chronos.uuid7 import uuid7

# Approximate tokens consumed by the recall response wrapper itself
_RECALL_OVERHEAD_TOKENS = 40

# Recency decay: score multiplier = 1 + recency_weight * boost.
# Default 0.3 gives a mild boost to fresh/confident memories without
# inverting rankings on high-scoring older ones. Set to 0.0 to disable.
_DEFAULT_RECENCY_WEIGHT = 0.3


def _recency_factor(created_at_iso: str) -> float:
    """
    Returns 1/(1+days_old) — approaches 1 for brand-new, approaches 0 for old.
    Falls back to 0.0 if the timestamp cannot be parsed.
    Legacy fallback used when FSRS data is unavailable.
    """
    try:
        created = datetime.fromisoformat(created_at_iso)
        now = datetime.now()
        days = max(0.0, (now - created.replace(tzinfo=None)).total_seconds() / 86400)
        return 1.0 / (1.0 + days)
    except (ValueError, TypeError):
        return 0.0


def _fsrs_boost(stability: float, last_reviewed: str, created_at: str,
                confidence: float) -> float:
    """
    FSRS-aware ranking boost. Combines retention probability with confidence.

    Returns a value in [0, 1]. Higher when the memory is both confident AND
    well-retained:
      boost = retention * confidence_factor
      retention = (1 + days/(9*S))^(-1)        [FSRS forgetting curve]
      confidence_factor = 0.5 + 0.5*confidence [0.01→0.505, 0.99→0.995]

    Confidence alone doesn't override staleness, and vice versa.
    """
    review_ts = last_reviewed or created_at
    days = BeliefEngine.days_since(review_ts)
    stab = stability if stability and stability > 0 else STABILITY_DEFAULT
    retention = BeliefEngine.compute_retention(stab, days)

    conf = confidence if confidence is not None else CONFIDENCE_DEFAULT
    confidence_factor = 0.5 + 0.5 * conf

    return retention * confidence_factor


class MemoryStore:
    """
    Full lifecycle of free-text memories. Stateless — all persistence and
    indexing live in SQLite (see db.py). Constructed once at startup and
    injected into tool handlers via closure.
    """

    # ------------------------------------------------------------------
    # remember
    # ------------------------------------------------------------------

    def remember(
        self,
        content: str,
        project: str = "default",
        tags: Optional[List[str]] = None,
        source: str = "user",
    ) -> dict:
        """
        Store a free-text memory. The FTS trigger indexes it in the same
        transaction as the insert.

        source: provenance label — where this memory's content came from
        (e.g. 'user', 'claude', 'web', 'document'). Recall results echo it
        so the consumer can weigh trust accordingly.

        Returns: {id, project, source, token_estimate}
        """
        if not content or not content.strip():
            raise ValueError("content must be a non-empty string")

        mem_id = uuid7()
        now    = datetime.now().isoformat()
        tags   = tags or []

        with get_db() as db:
            db.execute(
                """INSERT INTO memories
                   (id, project, content, tags, created_at, updated_at,
                    forgotten, source)
                   VALUES (?, ?, ?, ?, ?, ?, 0, ?)""",
                (mem_id, project, content.strip(), json.dumps(tags),
                 now, now, source),
            )
            db.commit()

        return {
            "id":             mem_id,
            "project":        project,
            "source":         source,
            "token_estimate": estimate_tokens(content),
        }

    # ------------------------------------------------------------------
    # recall
    # ------------------------------------------------------------------

    def recall(
        self,
        query: str,
        project: Optional[str] = None,
        k: int = 5,
        recency_weight: float = _DEFAULT_RECENCY_WEIGHT,
        token_budget: Optional[int] = None,
    ) -> dict:
        """
        Retrieve the k most relevant memories for a query.

        Ranking: BM25 over the FTS5 index (porter-stemmed), then re-ranked
        with an FSRS retention*confidence boost when recency_weight > 0.
        Scores are relative ranking values, not normalised [0,1].

        Compression runs ONLY when token_budget is set. Without a budget,
        content is returned in full.

        Returns:
          {
            results: [{id, project, content, score, token_estimate,
                       source, confidence?}],
            total_tokens: int,
            count: int,
            query: str,
            compression_applied: [..]   (only when token_budget was set)
          }
        """
        if not query or not query.strip():
            return {"results": [], "total_tokens": 0, "count": 0, "query": query}

        # Over-fetch when re-ranking may reorder the top-k
        fetch_k = k * 3 if recency_weight > 0 else k
        with get_db() as db:
            rows = search_memories(db, query, project=project, k=fetch_k)

        boosted = []
        for r in rows:
            score = r["score"]
            if recency_weight > 0:
                if r["stability"] is not None:
                    boost = _fsrs_boost(
                        r["stability"], r["last_reviewed"],
                        r["created_at"], r["confidence"],
                    )
                else:
                    boost = _recency_factor(r["created_at"])
                score = score * (1.0 + recency_weight * boost)
            boosted.append((r, score))

        boosted.sort(key=lambda x: x[1], reverse=True)
        boosted = boosted[:k]

        results      = []
        total_tokens = _RECALL_OVERHEAD_TOKENS

        for r, score in boosted:
            tok_est = estimate_tokens(r["content"])
            total_tokens += tok_est
            entry = {
                "id":             r["id"],
                "project":        r["project"],
                "content":        r["content"],
                "score":          round(score, 5),
                "token_estimate": tok_est,
                "source":         r["source"] or "user",
            }
            if r["confidence"] is not None:
                entry["confidence"] = round(r["confidence"], 4)
            results.append(entry)

        # v4.0: compression is opt-in. No budget → full content, always.
        if token_budget is not None:
            compressed = compress_results(
                results,
                budget=token_budget,
                overhead=_RECALL_OVERHEAD_TOKENS,
            )
            return {
                "results":             compressed["results"],
                "total_tokens":        compressed["total_tokens"],
                "count":               len(compressed["results"]),
                "query":               query,
                "compression_applied": compressed["compression_applied"],
            }

        return {
            "results":      results,
            "total_tokens": total_tokens,
            "count":        len(results),
            "query":        query,
        }

    # ------------------------------------------------------------------
    # get — full single-memory retrieval (no truncation, ever)
    # ------------------------------------------------------------------

    def get(self, memory_id: str) -> dict:
        """
        Fetch one memory by id with full content and metadata.
        Works on forgotten memories too (the forgotten field says so).

        Returns: {id, project, content, tags, source, created_at, updated_at,
                  confidence, forgotten, forget_reason, version_count,
                  token_estimate}
        or {id, status: 'not_found'}.
        """
        with get_db() as db:
            row = db.execute(
                """SELECT id, project, content, tags, source, created_at,
                          updated_at, confidence, forgotten, forget_reason
                   FROM memories WHERE id = ?""",
                (memory_id,),
            ).fetchone()
            if not row:
                return {"id": memory_id, "status": "not_found"}
            version_count = db.execute(
                "SELECT COUNT(*) FROM memory_versions WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()[0]

        return {
            "id":             row["id"],
            "project":        row["project"],
            "content":        row["content"],
            "tags":           json.loads(row["tags"] or "[]"),
            "source":         row["source"] or "user",
            "created_at":     row["created_at"],
            "updated_at":     row["updated_at"],
            "confidence":     round(row["confidence"], 4)
                              if row["confidence"] is not None else None,
            "forgotten":      bool(row["forgotten"]),
            "forget_reason":  row["forget_reason"],
            "version_count":  version_count,
            "token_estimate": estimate_tokens(row["content"]),
        }

    # ------------------------------------------------------------------
    # update
    # ------------------------------------------------------------------

    def update(self, memory_id: str, content: str) -> dict:
        """
        Replace the content of an existing memory. The old content is
        snapshotted into memory_versions (time-travel), and the FTS trigger
        re-indexes the new content in the same transaction.
        Raises ValueError if the memory does not exist or is forgotten.

        Returns: {id, status, token_estimate}
        """
        if not content or not content.strip():
            raise ValueError("content must be a non-empty string")

        with get_db() as db:
            row = db.execute(
                "SELECT forgotten, content, updated_at"
                " FROM memories WHERE id = ?",
                (memory_id,)
            ).fetchone()
            if not row:
                raise ValueError(f"Memory '{memory_id}' not found")
            if row["forgotten"] == 1:
                raise ValueError(
                    f"Memory '{memory_id}' is forgotten — call restore_memory"
                    " first, or remember() the corrected content as a new entry"
                )

            old_content = row["content"]
            old_updated = row["updated_at"]
            now         = datetime.now().isoformat()

            # Snapshot old content into memory_versions before overwriting.
            # valid_from = when this version became current; valid_to = now.
            db.execute(
                """INSERT INTO memory_versions
                   (id, memory_id, content, valid_from, valid_to)
                   VALUES (?, ?, ?, ?, ?)""",
                (uuid7(), memory_id, old_content, old_updated, now),
            )
            db.execute(
                "UPDATE memories SET content = ?, updated_at = ? WHERE id = ?",
                (content.strip(), now, memory_id),
            )
            db.commit()

        return {
            "id":             memory_id,
            "status":         "updated",
            "token_estimate": estimate_tokens(content),
        }

    # ------------------------------------------------------------------
    # Delegated operations — lifecycle.py and time_travel.py own the logic
    # ------------------------------------------------------------------

    def forget(self, memory_id: str, reason: str = "manual") -> dict:
        """Soft-delete. See lifecycle.forget()."""
        return lifecycle.forget(memory_id, reason)

    def restore(self, memory_id: str) -> dict:
        """Un-forget. See lifecycle.restore()."""
        return lifecycle.restore(memory_id)

    def purge(self, memory_id: str) -> dict:
        """Hard-delete, irreversible. See lifecycle.purge()."""
        return lifecycle.purge(memory_id)

    def query_at(
        self,
        query: str,
        timestamp: str,
        project: Optional[str] = None,
        k: int = 5,
    ) -> dict:
        """Time-travel recall. See time_travel.query_at()."""
        return _time_travel_query_at(query, timestamp, project, k)
