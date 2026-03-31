# chronos/mem_embed.py
# Responsibility: Map free-text memory content into hyperbolic embedding space.
#
# Purpose (Gap 1 fix):
#   The remember/recall pipeline (TF-IDF) and the add_event/query_similar pipeline
#   (hyperbolic embeddings) were previously disconnected silos. This module bridges
#   them by embedding memory content using structural content-derived features so
#   that query_similar_memories() can find related memories by spatial proximity.
#
# Feature vector (5 dimensions, all normalised to [0,1]):
#   0. content_len_norm  — word count / 200 (capped at 1.0)
#   1. unique_term_ratio — unique tokens / total tokens
#   2. tag_count_norm    — min(tag_count, 10) / 10
#   3. project_bucket    — sha256(project) % 10 / 10 (locality within project)
#   4. recency_factor    — 1/(1+days_old) — 1.0 for brand-new, → 0 for old
#
# Limitation acknowledged:
#   These are structural features, not semantic ones. Two memories with
#   identical content but different lengths can be far apart; two memories
#   with different content but similar lengths can be close. This is NOT
#   semantic similarity — it captures content-structure proximity and
#   project locality. Full semantic similarity requires an embedding model.
#   For project memory use-cases the project_bucket feature alone ensures
#   same-project memories cluster together.

import hashlib
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from chronos.db import get_db
from chronos.geometry import HyperbolicEmbedder
from chronos.tfidf import _tokenise  # reuse tokeniser for unique-term ratio


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _project_bucket(project: str) -> float:
    """Hash project name into [0, 1] — keeps same-project memories spatially close."""
    bucket = int(hashlib.sha256(project.encode()).hexdigest(), 16) % 10
    return bucket / 10.0


def _recency_feature(created_at_iso: str) -> float:
    """1/(1+days_old) — brand-new → 1.0, aged → 0.0."""
    try:
        created = datetime.fromisoformat(created_at_iso)
        days = max(0.0, (datetime.now() - created.replace(tzinfo=None)).total_seconds() / 86400)
        return 1.0 / (1.0 + days)
    except (ValueError, TypeError):
        return 0.0


def content_features(
    content: str,
    tags: List[str],
    project: str,
    created_at: str,
) -> List[float]:
    """
    Derive a 5-element normalised feature vector from memory metadata.
    All values are in [0, 1].
    """
    words  = content.split()
    tokens = _tokenise(content)

    content_len_norm  = min(1.0, len(words) / 200.0)
    unique_term_ratio = (len(set(tokens)) / max(1, len(tokens)))
    tag_count_norm    = min(len(tags), 10) / 10.0
    proj_bucket       = _project_bucket(project)
    recency           = _recency_feature(created_at)

    return [content_len_norm, unique_term_ratio, tag_count_norm,
            proj_bucket, recency]


# ---------------------------------------------------------------------------
# MemoryEmbedder — thin wrapper around HyperbolicEmbedder for memory vectors
# ---------------------------------------------------------------------------

class MemoryEmbedder:
    """
    Manages hyperbolic embeddings for free-text memories.
    Backed by the memory_vectors table (separate from the node embeddings table).

    Why separate from HyperbolicEmbedder:
      - Memories use a fixed 5-feature content vector; nodes use 4 payload features.
      - Mixing them into one index would corrupt query_similar (node-only) results.
      - Keeping two indexes lets us add query_similar_memories without side effects.
    """

    def __init__(self, dim: int = 32) -> None:
        self._embedder = HyperbolicEmbedder(dim=dim)
        # memory_id -> project (for project-scoped searches)
        self._projects: Dict[str, str] = {}

    @property
    def dim(self) -> int:
        return self._embedder.dim

    def load_from_db(self) -> int:
        """Load all memory vectors from DB into memory. Returns count loaded."""
        with get_db() as db:
            rows = db.execute(
                "SELECT memory_id, vector, project FROM memory_vectors"
            ).fetchall()
        for row in rows:
            self._embedder.nodes[row[0]] = np.frombuffer(row[1], dtype=np.float32).copy()
            self._projects[row[0]] = row[2]
        return len(rows)

    def embed_and_store(
        self,
        memory_id: str,
        content: str,
        tags: List[str],
        project: str,
        created_at: str,
    ) -> np.ndarray:
        """
        Compute feature vector, embed in hyperbolic space, persist to DB.
        Replaces any existing vector for this memory_id.
        """
        features = content_features(content, tags, project, created_at)
        vec = self._embedder.embed(memory_id, features)
        self._projects[memory_id] = project

        with get_db() as db:
            db.execute(
                """INSERT OR REPLACE INTO memory_vectors
                   (memory_id, vector, dim, project)
                   VALUES (?, ?, ?, ?)""",
                (memory_id, vec.tobytes(), self._embedder.dim, project),
            )
            db.commit()

        return vec

    def remove(self, memory_id: str) -> None:
        """Remove a forgotten memory's vector from index and DB."""
        self._embedder.remove(memory_id)
        self._projects.pop(memory_id, None)
        with get_db() as db:
            db.execute(
                "DELETE FROM memory_vectors WHERE memory_id = ?", (memory_id,)
            )
            db.commit()

    def nearest(
        self,
        query_id: str,
        k: int = 5,
        project: Optional[str] = None,
    ) -> list:
        """
        Hyperbolic nearest-neighbor search over memory vectors.

        query_id: a memory_id whose vector is already in the index.
        project:  if provided, restrict results to this project only.
        Returns:  list of {memory_id, distance}
        """
        if query_id not in self._embedder.nodes:
            return []

        exclude = set()
        if project:
            exclude = {
                mid for mid, proj in self._projects.items()
                if proj != project
            }

        results = self._embedder.nearest(query_id, k=k, tombstoned=exclude)
        return [{"memory_id": nid, "distance": round(float(d), 4)} for nid, d in results]

    def nearest_by_features(
        self,
        content: str,
        tags: List[str],
        project: str,
        created_at: str,
        k: int = 5,
        scope_project: Optional[str] = None,
    ) -> list:
        """
        Ad-hoc similarity search without requiring a stored vector.
        Computes a temporary feature vector, finds neighbors, discards temp entry.

        Used by query_similar_memories() when the caller provides raw text
        rather than an existing memory_id.
        Returns: list of {memory_id, distance}
        """
        import uuid
        tmp_id = f"_tmp_{uuid.uuid4().hex}"
        try:
            self.embed_and_store(tmp_id, content, tags, project, created_at)
            results = self.nearest(tmp_id, k=k + 1, project=scope_project)
            # Exclude the temp entry itself from results
            return [r for r in results if r["memory_id"] != tmp_id][:k]
        finally:
            self.remove(tmp_id)
