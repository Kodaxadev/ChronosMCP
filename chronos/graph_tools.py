# chronos/graph_tools.py
# Responsibility: MCP tool registrations for the knowledge graph layer.
# Owns: add_event, query_similar, add_constraint
#
# Separated from tools.py as part of the module split (tools.py exceeded 400 lines).
# Registration: called from tools.register() via register_graph_tools().

import hashlib
import json
from datetime import datetime
from typing import List

import numpy as np

from mcp.server.fastmcp import FastMCP

from chronos.db import get_db, get_tombstoned_ids
from chronos.uuid7 import uuid7
from chronos.validation import validate_event


def _author_bucket(s: str) -> int:
    """
    Hash author string into 0–9 bucket.
    Bounded range prevents any single feature from dominating the
    embedding distance calculation. The aggregate_id is intentionally
    excluded from features — it carries no semantic similarity signal.
    """
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % 10


def register_graph_tools(mcp: FastMCP, embedder) -> None:
    """
    Register graph-layer MCP tools on the given FastMCP instance.
    embedder: HyperbolicEmbedder singleton
    """

    @mcp.tool()
    async def add_event(aggregate_id: str, event_type: str, payload: dict) -> str:
        """
        Add a node/event to the knowledge graph.

        aggregate_id: format '{type}:{project}:{id}', e.g. 'node:myproject:task_001'.
                      The project segment is used by suggest_next_tasks() and
                      analyze_structure() for project scoping.
        event_type:   one of:
          - node_created     — creates node + auto-embeds for similarity search
          - node_updated     — re-embeds with new payload features
          - node_deleted     — tombstones node, removes from similarity search
          - node_restored    — un-tombstones, restores to similarity index
          - relation_added   — creates edge (payload: {source, target})
          - relation_removed — removes edge (payload: {source, target})
          - relation_updated — updates edge metadata
        payload: dict of node features. For node_created/node_updated, embedding
                 uses these keys: priority (int), tags (list), author (str),
                 complexity (int). Missing keys default to 0/empty.
                 For relation_added/removed: must include 'source' and 'target'
                 aggregate_ids.

        Returns: event_id (uuid7 string) — use for audit trail and ordering.
        """
        validate_event(aggregate_id, event_type, payload)
        event_id = uuid7()

        with get_db() as db:
            db.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)",
                (event_id, aggregate_id, event_type,
                 datetime.now().isoformat(), json.dumps(payload), "2.3"),
            )

            if event_type in ("node_created", "node_updated"):
                embedder.maybe_resize()
                features = [
                    payload.get("priority", 0),
                    len(payload.get("tags", [])),
                    _author_bucket(payload.get("author", "")),
                    payload.get("complexity", 5),
                ]
                vec = embedder.embed(aggregate_id, features)
                db.execute(
                    "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?)",
                    (aggregate_id, vec.tobytes(), 1, embedder.dim),
                )

            elif event_type == "node_deleted":
                reason = payload.get("reason", "manual_delete")
                db.execute(
                    "INSERT OR IGNORE INTO tombstones VALUES (?, ?, ?, ?)",
                    (aggregate_id, event_id, datetime.now().isoformat(), reason),
                )
                # Remove from in-memory index; KEEP vector in DB for causal validity
                embedder.remove(aggregate_id)

            elif event_type == "node_restored":
                db.execute(
                    "DELETE FROM tombstones WHERE node_id = ?", (aggregate_id,)
                )
                row = db.execute(
                    "SELECT vector FROM embeddings WHERE node_id = ?",
                    (aggregate_id,),
                ).fetchone()
                if row:
                    vec = np.frombuffer(row[0], dtype=np.float32).copy()
                    # FIX: Pad/truncate to current dim — same as load_from_db().
                    # Without this, a resize between delete and restore would
                    # leave this vector at the old dimension, causing numpy
                    # broadcast errors on distance computation.
                    if len(vec) < embedder.dim:
                        vec = np.pad(vec, (0, embedder.dim - len(vec)))
                    elif len(vec) > embedder.dim:
                        vec = vec[:embedder.dim]
                    embedder.nodes[aggregate_id] = vec

            db.commit()
        return event_id

    @mcp.tool()
    async def query_similar(node_id: str, k: int = 5) -> list:
        """
        Find the k most structurally similar nodes via hyperbolic distance.
        Tombstoned (deleted) nodes are automatically excluded.

        Similarity is based on node payload features (priority, tag count,
        author, complexity) embedded in Poincaré ball space — NOT content
        semantics. For keyword/content similarity, use recall(). For
        content-vector similarity, use query_similar_memories().

        node_id: aggregate_id of the reference node (must exist in graph).
        k:       number of neighbors to return (default 5, max 50).

        Returns: list of {node_id: str, distance: float} sorted by ascending
        distance. Lower distance = more similar. Distance 0.0 = identical
        features. Typical meaningful range: 0.0–2.0.
        """
        k = max(1, min(k, 50))
        with get_db() as db:
            tombstoned = get_tombstoned_ids(db)
        neighbors = embedder.nearest(node_id, k, tombstoned=tombstoned)
        return [{"node_id": nid, "distance": round(float(d), 4)} for nid, d in neighbors]

    @mcp.tool()
    async def add_constraint(
        node_id: str,
        constraint_type: str,
        depends_on: List[str] = None,
        priority: int = 1,
    ) -> dict:
        """
        Add a constraint for the dependency solver.

        constraint_type: ONLY 'dependency' is actively enforced by suggest_next_tasks().
                         'uniqueness', 'temporal', and 'capacity' are accepted and stored
                         but NOT enforced — they require the full §6.2 python-constraint
                         implementation. Storing them now reserves the record for future use.
        depends_on: list of node aggregate_ids this node depends on.
        priority:   lower = higher priority (1 = highest).

        node_id must be a valid aggregate_id (format: 'node:{project}:{id}').

        Returns: {constraint_id, enforced}
        enforced=True  → suggest_next_tasks() will respect this constraint.
        enforced=False → stored only, no effect on current ordering.
        """
        _ENFORCED_TYPES = {"dependency"}
        enforced = constraint_type in _ENFORCED_TYPES

        with get_db() as db:
            constraint_id = uuid7()
            data = {
                "type":       constraint_type,
                "depends_on": depends_on or [],
                "priority":   priority,
            }
            db.execute(
                "INSERT INTO constraints VALUES (?, ?, ?, ?, ?)",
                (constraint_id, node_id, constraint_type, priority, json.dumps(data)),
            )
            db.commit()

        result: dict = {"constraint_id": constraint_id, "enforced": enforced}
        if not enforced:
            result["warning"] = (
                f"constraint_type='{constraint_type}' is stored but NOT enforced. "
                "Only 'dependency' constraints affect suggest_next_tasks() output."
            )
        return result
