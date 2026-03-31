# chronos/memory.py
# Responsibility: High-level memory operations — remember, recall, forget, query_at.
# This is the primary interface between MCP tools and the storage/indexing layer.
#
# Architecture:
#   - memories table  → persistent text storage (db.py)
#   - TFIDFIndex      → in-memory content retrieval (tfidf.py)
#   - HyperbolicEmbedder → structural/relational similarity (geometry.py)
#
# Token budget design:
#   Every recall response includes a `token_estimate` field so Claude can
#   decide how many results to request without blindly inflating context.

import json
from datetime import datetime
from typing import List, Optional

from chronos.db import get_db
from chronos.tfidf import TFIDFIndex
from chronos.uuid7 import uuid7

# Approximate tokens consumed by the recall response wrapper itself
_RECALL_OVERHEAD_TOKENS = 40

# Recency decay: score multiplier = 1 + recency_weight * (1 / (1 + days_old))
# Default 0.3 gives ~23% boost for same-day memories, ~15% boost for 3-day-old,
# ~10% boost for 7-day-old.  Set to 0.0 to disable.
_DEFAULT_RECENCY_WEIGHT = 0.3


def _recency_factor(created_at_iso: str) -> float:
    """
    Returns 1/(1+days_old) — approaches 1 for brand-new, approaches 0 for old.
    Falls back to 0.0 if the timestamp cannot be parsed.
    """
    try:
        created = datetime.fromisoformat(created_at_iso)
        # Make both timezone-naive for diff
        now = datetime.now()
        days = max(0.0, (now - created.replace(tzinfo=None)).total_seconds() / 86400)
        return 1.0 / (1.0 + days)
    except (ValueError, TypeError):
        return 0.0


class MemoryStore:
    """
    Full lifecycle of free-text memories: remember, recall, forget, update, query_at.

    mem_embedder: optional MemoryEmbedder. When provided, remember() and update()
    also store content vectors enabling query_similar_memories(). Pass None for
    TF-IDF-only mode (e.g. lightweight tests).
    """

    def __init__(self, tfidf: TFIDFIndex, mem_embedder=None) -> None:
        self.tfidf       = tfidf
        self.mem_embedder = mem_embedder  # MemoryEmbedder | None

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def load(self) -> int:
        """
        Load all non-forgotten memories from DB into the TF-IDF index.
        If a MemoryEmbedder is attached, also loads memory vectors from DB.
        Call once at server startup after init_db().
        Returns number of memories loaded.
        """
        with get_db() as db:
            rows = db.execute(
                "SELECT id, content FROM memories WHERE forgotten = 0"
            ).fetchall()
        docs = [(r[0], r[1]) for r in rows]
        self.tfidf.load_documents(docs)
        if self.mem_embedder is not None:
            self.mem_embedder.load_from_db()
        return len(docs)

    # ------------------------------------------------------------------
    # remember
    # ------------------------------------------------------------------

    def remember(
        self,
        content: str,
        project: str = "default",
        tags: Optional[List[str]] = None,
    ) -> dict:
        """
        Store a free-text memory and index it for retrieval.

        Returns: {id, project, token_estimate, indexed_terms}
        """
        if not content or not content.strip():
            raise ValueError("content must be a non-empty string")

        mem_id = uuid7()
        now    = datetime.now().isoformat()
        tags   = tags or []

        with get_db() as db:
            db.execute(
                """INSERT INTO memories
                   (id, project, content, tags, created_at, updated_at, forgotten)
                   VALUES (?, ?, ?, ?, ?, ?, 0)""",
                (mem_id, project, content.strip(), json.dumps(tags), now, now),
            )
            db.commit()

        self.tfidf.add_document(mem_id, content)

        embedded = False
        if self.mem_embedder is not None:
            self.mem_embedder.embed_and_store(mem_id, content, tags, project, now)
            embedded = True

        return {
            "id":             mem_id,
            "project":        project,
            "token_estimate": self.tfidf.estimate_tokens(content),
            "indexed_terms":  self.tfidf.doc_count(),
            "embedded":       embedded,
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
    ) -> dict:
        """
        Retrieve the k most relevant memories for a query.

        Ranking: TF-IDF cosine similarity over stored content, with optional
        recency decay multiplier. Final score = tfidf_score * (1 + recency_weight
        * recency_factor) where recency_factor = 1/(1+days_old).

        recency_weight=0.0 disables decay (pure TF-IDF ranking).
        recency_weight=0.3 (default) gives a mild boost to recent memories
        without inverting rankings on high-scoring older ones.

        Returns:
          {
            results: [{id, project, content, score, token_estimate}],
            total_tokens: int,
            count: int,
            query: str,
          }
        """
        if not query or not query.strip():
            return {"results": [], "total_tokens": 0, "count": 0, "query": query}

        # Get forgotten IDs to exclude
        with get_db() as db:
            forgotten = {
                r[0] for r in db.execute(
                    "SELECT id FROM memories WHERE forgotten = 1"
                ).fetchall()
            }
            # Fetch all candidate metadata in one shot (include created_at for decay)
            if project:
                rows = db.execute(
                    """SELECT id, project, content, created_at FROM memories
                       WHERE project = ? AND forgotten = 0""",
                    (project,),
                ).fetchall()
            else:
                rows = db.execute(
                    """SELECT id, project, content, created_at FROM memories
                       WHERE forgotten = 0"""
                ).fetchall()

        meta = {r[0]: {"project": r[1], "content": r[2], "created_at": r[3]}
                for r in rows}

        # Restrict TF-IDF search to this project's IDs if filtered.
        # exclude = all forgotten IDs plus all IDs NOT in this project's metadata.
        # The meta dict already contains only the target project's docs (fetched
        # with WHERE project = ?), so the difference gives us all non-project docs.
        exclude = forgotten
        if project:
            exclude = exclude | (self.tfidf.doc_ids() - set(meta.keys()))

        # Pull more candidates than k if decay is on (re-rank may change top-k)
        fetch_k = k * 3 if recency_weight > 0 else k
        ranked  = self.tfidf.query(query, k=fetch_k, exclude=exclude)

        # Apply recency boost and re-rank
        boosted = []
        for doc_id, score in ranked:
            if doc_id not in meta:
                continue
            if recency_weight > 0:
                rf    = _recency_factor(meta[doc_id]["created_at"])
                score = score * (1.0 + recency_weight * rf)
            boosted.append((doc_id, score))

        boosted.sort(key=lambda x: x[1], reverse=True)
        boosted = boosted[:k]

        results      = []
        total_tokens = _RECALL_OVERHEAD_TOKENS

        for doc_id, score in boosted:
            content  = meta[doc_id]["content"]
            tok_est  = self.tfidf.estimate_tokens(content)
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
        }

    # ------------------------------------------------------------------
    # forget
    # ------------------------------------------------------------------

    def forget(self, memory_id: str, reason: str = "manual") -> dict:
        """
        Soft-delete a memory. Sets forgotten=1 in DB and removes from
        the TF-IDF index. The record is retained for audit and query_at().

        Returns: {id, status, reason}
        """
        with get_db() as db:
            row = db.execute(
                "SELECT id, forgotten FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()

            if not row:
                return {"id": memory_id, "status": "not_found", "reason": reason}
            if row[1] == 1:
                return {"id": memory_id, "status": "already_forgotten", "reason": reason}

            db.execute(
                """UPDATE memories SET forgotten = 1,
                   updated_at = ? WHERE id = ?""",
                (datetime.now().isoformat(), memory_id),
            )
            db.commit()

        self.tfidf.remove_document(memory_id)
        if self.mem_embedder is not None:
            self.mem_embedder.remove(memory_id)
        return {"id": memory_id, "status": "forgotten", "reason": reason}

    # ------------------------------------------------------------------
    # update
    # ------------------------------------------------------------------

    def update(self, memory_id: str, content: str) -> dict:
        """
        Replace the content of an existing memory and re-index it.
        Raises ValueError if the memory does not exist or is forgotten.

        Returns: {id, token_estimate}
        """
        if not content or not content.strip():
            raise ValueError("content must be a non-empty string")

        with get_db() as db:
            row = db.execute(
                "SELECT forgotten, project, tags, created_at, content, updated_at"
                " FROM memories WHERE id = ?",
                (memory_id,)
            ).fetchone()
            if not row:
                raise ValueError(f"Memory '{memory_id}' not found")
            if row[0] == 1:
                raise ValueError(f"Memory '{memory_id}' is forgotten — restore first")

            project     = row[1]
            tags        = json.loads(row[2] or "[]")
            created_at  = row[3]
            old_content = row[4]
            old_updated = row[5]
            now         = datetime.now().isoformat()

            # FIX #4: Snapshot old content into memory_versions before overwriting.
            # valid_from = when this version became current (last updated_at or created_at)
            # valid_to   = now (when it's being replaced)
            version_id = uuid7()
            db.execute(
                """INSERT INTO memory_versions
                   (id, memory_id, content, valid_from, valid_to)
                   VALUES (?, ?, ?, ?, ?)""",
                (version_id, memory_id, old_content, old_updated, now),
            )

            db.execute(
                "UPDATE memories SET content = ?, updated_at = ? WHERE id = ?",
                (content.strip(), now, memory_id),
            )
            db.commit()

        self.tfidf.add_document(memory_id, content)  # add_document replaces existing

        if self.mem_embedder is not None:
            self.mem_embedder.embed_and_store(
                memory_id, content, tags, project, created_at
            )

        return {
            "id":             memory_id,
            "status":         "updated",
            "token_estimate": self.tfidf.estimate_tokens(content),
        }

    # ------------------------------------------------------------------
    # query_at — time-travel
    # ------------------------------------------------------------------

    def query_at(
        self,
        query: str,
        timestamp: str,
        project: Optional[str] = None,
        k: int = 5,
    ) -> dict:
        """
        Reconstruct which memories existed at `timestamp` and rank them
        against `query` using a fresh TF-IDF index over that snapshot.

        timestamp: ISO 8601 string, e.g. '2026-03-01T00:00:00'
        Memories created after `timestamp` are excluded.
        Memories forgotten before `timestamp` are excluded.

        Returns same shape as recall() plus `as_of` field.
        """
        if not query or not query.strip():
            return {
                "results": [], "total_tokens": 0,
                "count": 0, "query": query, "as_of": timestamp,
            }

        # FIX: Validate timestamp is parseable ISO 8601 before using in SQL.
        # Without this, garbage strings like "not_a_date" are silently accepted
        # and produce unpredictable results via SQLite string comparison.
        try:
            parsed_ts = datetime.fromisoformat(timestamp)
            # FIX: Strip timezone info for consistent lexicographic comparison
            # against tz-naive stored timestamps. Without this, a tz-aware input
            # like "2026-03-01T00:00:00+05:30" would sort incorrectly against
            # tz-naive DB values due to the +/- suffix.
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

            # FIX #4: Load version history for candidate memories only.
            # Scoped to the memory_ids from the main query (not the full table).
            # FIX: Batch into chunks of 900 to stay under SQLite's
            # SQLITE_MAX_VARIABLE_NUMBER limit (999 on older builds).
            candidate_ids = [r[0] for r in rows]
            version_rows = []
            _BATCH = 900
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
            # If a version's valid_from <= timestamp < valid_to, use that content.
            # If no version matches, the current content was already current then.
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

        # Build a temporary TF-IDF index over the snapshot
        snap_index = TFIDFIndex()
        snap_index.load_documents([(s[0], s[1]) for s in snapshot])
        ranked = snap_index.query(query, k=k)

        meta         = {s[0]: {"content": s[1], "project": s[2]} for s in snapshot}
        results      = []
        total_tokens = _RECALL_OVERHEAD_TOKENS

        for doc_id, score in ranked:
            content  = meta[doc_id]["content"]
            tok_est  = snap_index.estimate_tokens(content)
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
