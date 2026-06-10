# chronos/memory.py
# Responsibility: High-level memory operations — remember, get, update —
# and the MemoryStore facade that tool handlers close over.
# Delegation map (each module owns its logic, this file stays thin):
#   recall pipeline            → ranking.py
#   forget / restore / purge   → lifecycle.py
#   time-travel                → time_travel.py
#   semantic vectors (opt-in)  → semantic.py
#
# Architecture (v4.x):
#   - memories table   → persistent text storage (db.py)
#   - memories_fts     → BM25 full-text index, trigger-synced (db.py, search.py)
#   - memory_embeddings→ semantic vectors, write-through ONLY when a
#                        SemanticSearch is injected (CHRONOS_SEMANTIC=1)
#
# MemoryStore holds no state of its own beyond the optional semantic handle.

import json
from datetime import datetime
from typing import List, Optional

from chronos import lifecycle, ranking
from chronos.db import get_db
from chronos.search import estimate_tokens
from chronos.time_travel import query_at as _time_travel_query_at
from chronos.uuid7 import uuid7


class MemoryStore:
    """
    Full lifecycle of free-text memories. All persistence and indexing live
    in SQLite (see db.py). Constructed once at startup and injected into
    tool handlers via closure.

    semantic: optional SemanticSearch. When provided, remember() and
    update() write-through to the embedding index, and recall() runs in
    hybrid (BM25 ∪ vector, RRF-fused) mode. None = pure lexical v4.0
    behavior.
    """

    def __init__(self, semantic=None) -> None:
        self.semantic = semantic

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
        transaction as the insert; the semantic index (when enabled) is
        updated write-through immediately after.

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

        if self.semantic is not None:
            self.semantic.embed_memory(mem_id, content.strip())

        return {
            "id":             mem_id,
            "project":        project,
            "source":         source,
            "token_estimate": estimate_tokens(content),
        }

    # ------------------------------------------------------------------
    # recall — delegated to ranking.py
    # ------------------------------------------------------------------

    def recall(
        self,
        query: str,
        project: Optional[str] = None,
        k: int = 5,
        recency_weight: float = ranking.DEFAULT_RECENCY_WEIGHT,
        token_budget: Optional[int] = None,
    ) -> dict:
        """
        Retrieve the k most relevant memories. Lexical BM25 by default;
        hybrid BM25 ∪ semantic with RRF fusion when semantic search is
        enabled. See ranking.recall() for the full pipeline.
        """
        return ranking.recall(
            query,
            project=project,
            k=k,
            recency_weight=recency_weight,
            token_budget=token_budget,
            semantic=self.semantic,
        )

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
        snapshotted into memory_versions (time-travel), the FTS trigger
        re-indexes the new content in the same transaction, and the
        semantic vector (when enabled) is refreshed write-through.
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

        if self.semantic is not None:
            self.semantic.embed_memory(memory_id, content.strip())

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
