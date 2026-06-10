# chronos/semantic.py
# Responsibility: optional local semantic search over memory embeddings.
#
# Enabled via CHRONOS_SEMANTIC=1 and `pip install "chronosmcp[semantic]"`.
# When off (the default), nothing in this module runs and recall behavior is
# byte-identical to v4.0 — the semantic stack is injected into MemoryStore
# only when the flag is set (see chronos_mcp.py).
#
# Design:
#   - Embeddings come from a local ONNX model via fastembed (default
#     BAAI/bge-small-en-v1.5, 384-dim, ~70MB downloaded once on first use).
#     No API keys, no cloud, no torch.
#   - Vectors persist in the memory_embeddings table with a content hash
#     (stale detection after edits made while the flag was off) and the
#     model name (clean model switches re-embed instead of mixing spaces).
#   - search() is brute-force normalized-dot-product over all active
#     vectors. At personal scale (10k memories × 384 dims ≈ 15MB) this is
#     single-digit milliseconds in numpy; an ANN index would be premature.
#   - Retrieval is HYBRID, not re-rank-only: semantic neighbors are fused
#     with BM25 candidates by Reciprocal Rank Fusion (ranking.py). Re-rank-
#     only cannot fix the synonym problem — if BM25 finds zero candidates
#     for "car", nothing can re-rank "automobile" into view.
#
# Testability: pass embed_fn to the constructor to avoid model downloads
# (CI uses a deterministic fake; the real model is exercised by an opt-in
# test that skips when the extra isn't installed).

import hashlib
import os
from typing import Callable, List, Optional, Tuple

import numpy as np

from chronos.db import get_db

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# Candidates returned to the RRF fusion stage. Mirrors the BM25 candidate
# count in ranking.py so neither retriever dominates by list length.
SEARCH_CANDIDATES = 50

# Cosine floor — neighbors below this are noise, not candidates. Embedding
# similarity rarely drops this low for related text with bge-small.
MIN_SIMILARITY = 0.30

_INSTALL_HINT = (
    "CHRONOS_SEMANTIC=1 requires the semantic extra. "
    "Fix: pip install \"chronosmcp[semantic]\"  "
    "(downloads a local ~70MB ONNX model on first use; no API keys)"
)


def content_hash(text: str) -> str:
    """Short stable hash used to detect stale vectors after content edits."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class SemanticSearch:
    """
    Embedding store + brute-force similarity search for memories.

    embed_fn: optional Callable[[list[str]], list[vector]] for tests.
    When None, fastembed is required at construction time (fail loud with
    install instructions rather than degrade silently) but the model itself
    loads lazily on first embed.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        embed_fn: Optional[Callable[[List[str]], list]] = None,
    ) -> None:
        self.model_name = model_name or os.environ.get(
            "CHRONOS_SEMANTIC_MODEL", DEFAULT_MODEL
        )
        self._embed_fn = embed_fn
        self._model = None
        if embed_fn is None:
            try:
                import fastembed  # noqa: F401 — availability check only
            except ImportError as exc:
                raise RuntimeError(_INSTALL_HINT) from exc

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts into row-normalized float32 vectors."""
        if self._embed_fn is not None:
            raw = self._embed_fn(texts)
        else:
            if self._model is None:
                from fastembed import TextEmbedding
                self._model = TextEmbedding(model_name=self.model_name)
            raw = self._model.embed(texts)
        return np.vstack(
            [_normalize(np.asarray(v, dtype=np.float32)) for v in raw]
        )

    def embed_memory(self, memory_id: str, content: str) -> None:
        """
        Upsert the vector for one memory. Called from remember()/update()
        when semantic search is enabled, keeping the index write-through.
        """
        vec = self._embed([content])[0]
        with get_db() as db:
            db.execute(
                """INSERT OR REPLACE INTO memory_embeddings
                   (memory_id, vector, dim, model, content_hash)
                   VALUES (?, ?, ?, ?, ?)""",
                (memory_id, vec.tobytes(), len(vec), self.model_name,
                 content_hash(content)),
            )
            db.commit()

    # ------------------------------------------------------------------
    # Backfill
    # ------------------------------------------------------------------

    def backfill(self, check_stale: bool = False, batch_size: int = 64) -> int:
        """
        Embed every active memory that has no vector for the current model —
        and, when check_stale=True, every memory whose stored hash no longer
        matches its content (edited while the flag was off, or model switch).

        Run once at startup when CHRONOS_SEMANTIC=1. Returns count embedded.
        First run on an existing database downloads the model and embeds the
        whole corpus — minutes, once; logged by the caller.
        """
        with get_db() as db:
            rows = db.execute(
                """SELECT m.id, m.content, e.content_hash
                   FROM memories m
                   LEFT JOIN memory_embeddings e
                     ON e.memory_id = m.id AND e.model = ?
                   WHERE m.forgotten = 0""",
                (self.model_name,),
            ).fetchall()

        todo = [
            (r["id"], r["content"]) for r in rows
            if r["content_hash"] is None
            or (check_stale and r["content_hash"] != content_hash(r["content"]))
        ]
        if not todo:
            return 0

        done = 0
        for i in range(0, len(todo), batch_size):
            batch = todo[i:i + batch_size]
            vecs = self._embed([c for _, c in batch])
            with get_db() as db:
                for (mid, c), vec in zip(batch, vecs, strict=True):
                    db.execute(
                        """INSERT OR REPLACE INTO memory_embeddings
                           (memory_id, vector, dim, model, content_hash)
                           VALUES (?, ?, ?, ?, ?)""",
                        (mid, vec.tobytes(), len(vec), self.model_name,
                         content_hash(c)),
                    )
                db.commit()
            done += len(batch)
        return done

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        db,
        query: str,
        project: Optional[str] = None,
        k: int = SEARCH_CANDIDATES,
    ) -> List[Tuple[str, float]]:
        """
        Nearest active memories to the query by embedding similarity.

        db: open connection from get_db() (shares the recall transaction's
        view). Returns [(memory_id, similarity)] best-first, floor-filtered.
        Memories without vectors (embed failures, races) are simply absent —
        BM25 still covers them, so hybrid recall degrades gracefully.
        """
        sql = (
            "SELECT e.memory_id, e.vector FROM memory_embeddings e "
            "JOIN memories m ON m.id = e.memory_id "
            "WHERE m.forgotten = 0 AND e.model = ?"
        )
        params: list = [self.model_name]
        if project:
            sql += " AND m.project = ?"
            params.append(project)
        rows = db.execute(sql, params).fetchall()
        if not rows:
            return []

        matrix = np.vstack(
            [np.frombuffer(r["vector"], dtype=np.float32) for r in rows]
        )
        qvec = self._embed([query])[0]
        sims = matrix @ qvec  # all vectors are normalized → dot = cosine

        order = np.argsort(sims)[::-1][:k]
        return [
            (rows[int(i)]["memory_id"], float(sims[int(i)]))
            for i in order
            if sims[int(i)] >= MIN_SIMILARITY
        ]
