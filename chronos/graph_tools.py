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

        aggregate_id format: '{type}:{project}:{id}'  e.g. 'node:myproject:task_001'
        event_type: one of node_created | node_updated | node_deleted | node_restored |
                    relation_added | relation_removed | relation_updated

        Auto-embeds on node_created and node_updated (embedding dimension may grow
        if the node count crosses an adaptive threshold — see §4.1).
        Writes tombstone on node_deleted and removes the node from similarity searches.
        Restores tombstone and embedding index on node_restored.
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

        Similarity is based on node payload features (priority, tag count, author,
        complexity) — not content semantics. For memory content similarity,
        use query_similar_memories() instead.
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
