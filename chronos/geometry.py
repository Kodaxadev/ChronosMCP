# chronos/geometry.py
# Responsibility: Hyperbolic embedding engine.
# Contains: PoincareBall (geometry ops), HyperbolicEmbedder (node index),
#           calculate_dimension (adaptive dimensionality — §4.1).

import math
from typing import Dict

import numpy as np

from chronos.db import get_db


# ---------------------------------------------------------------------------
# Adaptive dimensionality (§4.1)
# ---------------------------------------------------------------------------

def calculate_dimension(node_count: int) -> int:
    """
    §4.1 formula: 4 * log2(N), rounded up.
    Min: 16. Max: 128. Override: 32 for N < 50 (variance protection).
    """
    if node_count < 50:
        return 32
    return min(128, max(16, math.ceil(4 * math.log2(node_count))))


# ---------------------------------------------------------------------------
# Poincaré ball geometry (§4.5)
# ---------------------------------------------------------------------------

class PoincareBall:
    def __init__(self, dim: int = 32, c: float = 1.0):
        self.dim = dim
        self.c   = c

    def mobius_add(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Möbius addition in the Poincaré ball."""
        x2  = np.sum(x ** 2)
        y2  = np.sum(y ** 2)
        xy  = np.sum(x * y)
        num = (1 + 2 * self.c * xy + self.c * y2) * x + (1 - self.c * x2) * y
        den = 1 + 2 * self.c * xy + self.c ** 2 * x2 * y2
        return num / (den + 1e-10)

    def exponential_map(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Exponential map at x in direction v (for gradient-based updates)."""
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-6:
            return x
        sqrt_c = np.sqrt(self.c)
        return self.mobius_add(
            x, (np.tanh(sqrt_c * v_norm) / (sqrt_c * v_norm)) * v
        )

    def dist(self, x: np.ndarray, y: np.ndarray) -> float:
        # FIX #6: Clip by vector NORM, not per-element. Per-element clipping
        # does not enforce ||x|| < 1 (Poincaré ball constraint) in high dims.
        # project_to_ball uses 0.95; here we use 0.99 as a defensive fallback.
        x = self._clip_norm(x, 0.99)
        y = self._clip_norm(y, 0.99)
        x2, y2 = np.sum(x ** 2), np.sum(y ** 2)
        xy     = np.sum((x - y) ** 2)
        # Clamp arccosh argument to >= 1.0 for float safety
        arg = 1 + 2 * self.c * xy / ((1 - self.c * x2) * (1 - self.c * y2) + 1e-10)
        return np.arccosh(max(1.0, arg)) / np.sqrt(self.c)

    @staticmethod
    def _clip_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
        """Rescale vector so ||v|| <= max_norm. Preserves direction."""
        norm = np.linalg.norm(v)
        if norm > max_norm:
            return v * (max_norm / norm)
        return v

    def project_to_ball(self, x: np.ndarray) -> np.ndarray:
        """Project to ball, preserving relative magnitudes (hierarchy signal)."""
        norm = np.linalg.norm(x)
        if norm >= 0.95:
            x = x * (0.95 / norm)
        return x


# ---------------------------------------------------------------------------
# Node embedding index
# ---------------------------------------------------------------------------

class HyperbolicEmbedder:
    def __init__(self, dim: int = 32):
        self.dim  = dim
        self.ball = PoincareBall(dim)
        self.nodes: Dict[str, np.ndarray] = {}

    def load_from_db(self) -> None:
        """Restore embeddings from DB. Call once at server startup.
        Vectors with mismatched dimensions are padded/truncated to self.dim."""
        with get_db() as db:
            for row in db.execute(
                "SELECT node_id, vector FROM embeddings"
            ).fetchall():
                vec = np.frombuffer(row[1], dtype=np.float32).copy()
                # Defensive: align dimension if DB has stale vectors
                if len(vec) < self.dim:
                    vec = np.pad(vec, (0, self.dim - len(vec)))
                elif len(vec) > self.dim:
                    vec = vec[:self.dim]
                self.nodes[row[0]] = vec

    def maybe_resize(self) -> bool:
        """
        GAP-05: Resize embedding space if node count has grown enough.
        Only grows — never shrinks — to avoid geometric inconsistency with
        existing stored vectors. When a resize occurs, all in-memory vectors
        are zero-padded to the new dimension AND persisted to DB so that
        a server restart loads vectors at the correct dimension.
        Returns True if a resize occurred.
        """
        target_dim = calculate_dimension(len(self.nodes))
        if target_dim <= self.dim:
            return False

        # FIX #3: Re-pad all vectors AND persist to DB in one transaction.
        # Previously only in-memory vectors were resized, causing dimension
        # mismatches after restart (old DB vectors shorter than new ones).
        with get_db() as db:
            for nid, vec in self.nodes.items():
                if len(vec) < target_dim:
                    self.nodes[nid] = np.pad(vec, (0, target_dim - len(vec)))
                elif len(vec) > target_dim:
                    self.nodes[nid] = vec[:target_dim]
                db.execute(
                    "UPDATE embeddings SET vector = ?, dim = ? WHERE node_id = ?",
                    (self.nodes[nid].tobytes(), target_dim, nid),
                )
            db.commit()

        self.dim  = target_dim
        self.ball = PoincareBall(target_dim)
        return True

    def embed(self, node_id: str, features: list) -> np.ndarray:
        """
        Embed node into the Poincaré ball.

        Feature scaling: all raw features are min-max scaled to [0, 1] before
        padding/truncation so no single feature dominates by magnitude.
        The hash-based identity feature is scaled alongside the others — it
        acts as a stable tiebreaker rather than the primary distance signal.
        Magnitude hierarchy is still preserved by project_to_ball().
        """
        vec = np.array(features, dtype=np.float32)

        # Min-max scale to [0, 1] so features contribute equally
        v_min, v_max = vec.min(), vec.max()
        if v_max - v_min > 1e-8:
            vec = (vec - v_min) / (v_max - v_min)
        else:
            vec = np.zeros_like(vec)  # all features identical — no signal

        if len(vec) > self.dim:
            vec = vec[:self.dim]
        elif len(vec) < self.dim:
            vec = np.pad(vec, (0, self.dim - len(vec)))

        vec = self.ball.project_to_ball(vec)
        self.nodes[node_id] = vec
        return vec

    def remove(self, node_id: str) -> None:
        """Remove a tombstoned node from the in-memory index."""
        self.nodes.pop(node_id, None)

    def nearest(self, query_id: str, k: int = 5,
                tombstoned: set = None) -> list:
        """Hyperbolic nearest neighbors. Excludes tombstoned nodes."""
        if query_id not in self.nodes:
            return []
        tombstoned = tombstoned or set()
        q = self.nodes[query_id]
        distances = [
            (nid, self.ball.dist(q, vec))
            for nid, vec in self.nodes.items()
            if nid != query_id and nid not in tombstoned
        ]
        return sorted(distances, key=lambda x: x[1])[:k]
