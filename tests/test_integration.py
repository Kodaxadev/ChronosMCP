# tests/test_integration.py
# End-to-end flows through the registered MCP tool layer: core memory tools,
# cognitive tools, consolidation, graph gating, and the stats resource.

import asyncio

from mcp.server.fastmcp import FastMCP

from chronos.analyzers import CausalAnalyzer, ConstraintSolver, StructureAnalyzer
from chronos.analysis_tools import register_analysis_tools
from chronos.beliefs import BeliefEngine
from chronos.consolidation import ConsolidationEngine
from chronos.geometry import HyperbolicEmbedder
from chronos.graph_tools import register_graph_tools
from chronos.memory import MemoryStore
from chronos.tools import register


def _build_core(with_graph: bool = False):
    mcp = FastMCP("chronos-test")
    beliefs = BeliefEngine()
    embedder = None
    if with_graph:
        embedder = HyperbolicEmbedder(dim=32)
        embedder.load_from_db()
        register_graph_tools(mcp, embedder)
        register_analysis_tools(
            mcp, CausalAnalyzer(), ConstraintSolver(), StructureAnalyzer()
        )
    register(
        mcp,
        MemoryStore(),
        belief_engine=beliefs,
        consolidation_engine=ConsolidationEngine(beliefs),
        embedder=embedder,
    )
    tools = {tool.name: tool.fn for tool in mcp._tool_manager.list_tools()}
    resources = {
        str(resource.uri): resource.fn
        for resource in mcp._resource_manager.list_resources()
    }
    return tools, resources


def test_graph_tools_absent_unless_enabled():
    tools, _ = _build_core(with_graph=False)
    assert "remember" in tools and "recall" in tools
    for graph_tool in ("add_event", "query_similar", "add_constraint",
                       "analyze_causal", "suggest_next_tasks"):
        assert graph_tool not in tools

    tools, _ = _build_core(with_graph=True)
    assert "add_event" in tools and "analyze_causal" in tools


def test_memory_belief_consolidation_flow():
    async def scenario():
        tools, resources = _build_core()

        jwt = await tools["remember"](
            content="JWT tokens expire after 24 hours and refresh tokens live in Redis.",
            project="auth", tags=["auth", "jwt"], source="claude",
        )
        await tools["remember"](
            content="Database migrations use Alembic and must run in a transaction.",
            project="auth", tags=["database"],
        )
        devops = await tools["remember"](
            content="GitHub Actions deploys staging after main branch merges.",
            project="devops",
        )

        out = await tools["recall"](query="JWT authentication", project="auth")
        assert out["results"][0]["id"] == jwt["id"]
        assert out["results"][0]["source"] == "claude"

        upd = await tools["update_memory"](
            memory_id=jwt["id"],
            content="JWT tokens expire after 12 hours; refresh tokens live in Redis.",
        )
        assert upd["status"] == "updated"
        refreshed = await tools["recall"](query="JWT expire Redis", project="auth")
        assert "12 hours" in refreshed["results"][0]["content"]

        full = await tools["get_memory"](memory_id=jwt["id"])
        assert full["version_count"] == 1 and "12 hours" in full["content"]

        related = await tools["related_memories"](memory_id=jwt["id"])
        assert related["source_id"] == jwt["id"]
        assert all(r["memory_id"] != jwt["id"] for r in related["results"])

        gone = await tools["forget"](memory_id=devops["id"], reason="obsolete")
        assert gone["status"] == "forgotten"
        assert (await tools["recall"](query="GitHub Actions"))["count"] == 0
        back = await tools["restore_memory"](memory_id=devops["id"])
        assert back["status"] == "restored"
        assert (await tools["recall"](query="GitHub Actions"))["count"] == 1

        purged = await tools["purge_memory"](memory_id=devops["id"])
        assert purged["status"] == "purged"
        assert (await tools["get_memory"](memory_id=devops["id"]))["status"] == "not_found"

        conf = await tools["set_confidence"](
            memory_id=jwt["id"], confidence=0.9, reason="integration test"
        )
        assert conf["new_confidence"] == 0.9
        health = await tools["get_memory_health"](memory_id=jwt["id"])
        assert health["confidence"] == 0.9

        await tools["log_search_feedback"](
            query="JWT authentication", memory_id=jwt["id"], used=True
        )
        eff = await tools["search_effectiveness"]()
        assert eff["total_searches"] == 1

        report = await tools["consolidate_memories"](project="auth")
        assert report["phases_run"] == ["orient", "gather", "consolidate", "prune"]

        stats = await resources["chronos://stats"]()
        assert "Schema version:       4.0" in stats
        assert "Graph layer:          disabled" in stats
        assert "Memories (purged):    1" in stats

    asyncio.run(scenario())


def test_graph_and_analysis_flow():
    async def scenario():
        tools, _ = _build_core(with_graph=True)

        for name, prio, cplx in [("task_a", 1, 2), ("task_b", 2, 5), ("task_c", 3, 1)]:
            await tools["add_event"](
                aggregate_id=f"node:sprint1:{name}",
                event_type="node_created",
                payload={
                    "title": name, "project_id": "sprint1",
                    "priority": prio, "complexity": cplx, "author": "alice",
                },
            )
        await tools["add_event"](
            aggregate_id="node:sprint1:task_a",
            event_type="relation_added",
            payload={"source": "node:sprint1:task_a", "target": "node:sprint1:task_b"},
        )
        await tools["add_constraint"](
            node_id="node:sprint1:task_b",
            constraint_type="dependency",
            depends_on=["node:sprint1:task_a"],
            priority=2,
        )

        order = await tools["suggest_next_tasks"](project_id="sprint1")
        assert order["total_tasks"] == 3
        assert order["suggested_order"].index("task_a") < order[
            "suggested_order"].index("task_b")

        structure = await tools["analyze_structure"](project_id="sprint1")
        assert structure["total_nodes"] == 3

        # causal analysis over a small synthetic cohort
        for i in range(20):
            team = "a" if i < 10 else "b"
            await tools["add_event"](
                aggregate_id=f"node:causal:item_{i:03d}",
                event_type="node_created",
                payload={
                    "project_id": "causal", "team": team,
                    "complexity": i % 5,
                    "score": i * 2 if team == "a" else i,
                },
            )
        causal = await tools["analyze_causal"](
            treatment_filter={"team": "a"},
            outcome_metric="score",
            confounder_keys=["complexity"],
        )
        assert causal["confounder_used"] == "complexity"
        assert causal["n"] > 0

    asyncio.run(scenario())


def test_cycle_detection_in_solver():
    solver = ConstraintSolver()
    ordered = solver.solve_next_actions([
        {"id": "A", "depends_on": ["B"], "priority": 1},
        {"id": "B", "depends_on": ["A"], "priority": 1},
        {"id": "C", "depends_on": [], "priority": 2},
    ])
    warnings = [t for t in ordered if "_cycle_warning" in t]
    assert [t["id"] for t in ordered][:1] == ["C"]
    assert {t["id"] for t in warnings} == {"A", "B"}
