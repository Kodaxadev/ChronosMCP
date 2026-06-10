# tests/test_graph.py
# Graph layer (opt-in): the v3.3 audit regressions — resize deadlock and
# degenerate per-vector normalization — plus basic graph tool behavior.

import asyncio

from mcp.server.fastmcp import FastMCP

import chronos.geometry as geometry
from chronos.geometry import HyperbolicEmbedder
from chronos.graph_tools import _node_features, register_graph_tools


def _graph_tools(embedder):
    mcp = FastMCP("graph-test")
    register_graph_tools(mcp, embedder)
    return {tool.name: tool.fn for tool in mcp._tool_manager.list_tools()}


def test_add_event_during_resize_does_not_deadlock(monkeypatch):
    """
    v3.3 REGRESSION (verified by repro in the audit): maybe_resize() opened a
    second connection while add_event held the write lock, failing with
    'database is locked' the first time the dimension threshold was crossed
    (~257 nodes). v4.0 resizes before the write transaction opens.
    """
    embedder = HyperbolicEmbedder(dim=32)
    tools = _graph_tools(embedder)

    async def scenario():
        # First node persists a vector so the resize has rows to UPDATE
        await tools["add_event"](
            aggregate_id="node:p:first",
            event_type="node_created",
            payload={"title": "first", "priority": 1, "complexity": 2},
        )
        # Force the next add_event to trigger a resize mid-flight
        monkeypatch.setattr(geometry, "calculate_dimension", lambda n: 40)
        await tools["add_event"](
            aggregate_id="node:p:second",
            event_type="node_created",
            payload={"title": "second", "priority": 2, "complexity": 3},
        )

    asyncio.run(scenario())  # old code: sqlite3.OperationalError after 5s
    assert embedder.dim == 40
    assert all(len(v) == 40 for v in embedder.nodes.values())


def test_proportional_payloads_embed_differently():
    """
    v3.3 REGRESSION: per-vector min-max scaling mapped proportional payloads
    (priority=1/complexity=5 vs priority=2/complexity=10) to IDENTICAL
    embeddings. Fixed per-feature scales must keep them apart.
    """
    embedder = HyperbolicEmbedder(dim=32)
    va = embedder.embed("a", _node_features({"priority": 1, "complexity": 5}))
    vb = embedder.embed("b", _node_features({"priority": 2, "complexity": 10}))
    assert embedder.ball.dist(va, vb) > 1e-4


def test_node_features_fixed_scales_and_clamping():
    f = _node_features({"priority": 99, "complexity": -5, "tags": list(range(50))})
    assert f[0] == 1.0   # priority clamped to 10 → 1.0
    assert f[1] == 1.0   # tag count clamped
    assert f[3] == 0.0   # negative complexity clamped
    assert all(0.0 <= x <= 1.0 for x in f)


def test_tombstone_excludes_from_similarity_and_restore_returns():
    embedder = HyperbolicEmbedder(dim=32)
    tools = _graph_tools(embedder)

    async def scenario():
        for name, prio in [("a", 1), ("b", 2), ("c", 3)]:
            await tools["add_event"](
                aggregate_id=f"node:p:{name}",
                event_type="node_created",
                payload={"title": name, "priority": prio, "complexity": prio},
            )
        await tools["add_event"](
            aggregate_id="node:p:c",
            event_type="node_deleted",
            payload={"reason": "cancelled"},
        )
        similar = await tools["query_similar"](node_id="node:p:a")
        assert "node:p:c" not in {r["node_id"] for r in similar}

        await tools["add_event"](
            aggregate_id="node:p:c",
            event_type="node_restored",
            payload={"restored_by": "test"},
        )
        assert "node:p:c" in embedder.nodes

    asyncio.run(scenario())
