import atexit
import asyncio
import os
import tempfile
import time

_tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp_db.close()
os.environ["CHRONOS_DB_PATH"] = _tmp_db.name
atexit.register(lambda: os.path.exists(_tmp_db.name) and os.unlink(_tmp_db.name))

from mcp.server.fastmcp import FastMCP

from chronos.analyzers import CausalAnalyzer, ConstraintSolver, StructureAnalyzer
from chronos.beliefs import BeliefEngine
from chronos.consolidation import ConsolidationEngine
from chronos.db import get_db, init_db
from chronos.geometry import HyperbolicEmbedder
from chronos.mem_embed import MemoryEmbedder
from chronos.memory import MemoryStore
from chronos.tfidf import TFIDFIndex
from chronos.tools import register
from chronos.uuid7 import uuid7
from chronos.validation import validate_event


def _build_tools():
    init_db()
    mcp = FastMCP("chronos-test")
    embedder = HyperbolicEmbedder(dim=32)
    embedder.load_from_db()
    tfidf = TFIDFIndex()
    mem_embedder = MemoryEmbedder(dim=32)
    mem_store = MemoryStore(tfidf, mem_embedder=mem_embedder)
    mem_store.load()
    beliefs = BeliefEngine()
    consolidation = ConsolidationEngine(beliefs, tfidf)
    register(
        mcp,
        embedder,
        CausalAnalyzer(),
        ConstraintSolver(),
        StructureAnalyzer(),
        mem_store,
        mem_embedder,
        belief_engine=beliefs,
        consolidation_engine=consolidation,
    )
    tools = {tool.name: tool.fn for tool in mcp._tool_manager.list_tools()}
    resources = {
        str(resource.uri): resource.fn
        for resource in mcp._resource_manager.list_resources()
    }
    return tools, resources, embedder


async def _call(tools, name, **kwargs):
    return await tools[name](**kwargs)


def test_schema_uuid_and_validation():
    init_db()
    with get_db() as db:
        tables = {
            row[0]
            for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    expected = {
        "events",
        "embeddings",
        "causal_results",
        "constraints",
        "tombstones",
        "memories",
        "memory_vectors",
        "memory_versions",
        "belief_updates",
        "search_feedback",
    }
    assert expected <= tables

    early = uuid7()
    time.sleep(0.002)
    late = uuid7()
    assert early < late
    assert early.split("-")[2][0] == "7"

    validate_event("node:project:task", "node_created", {"title": "Task"})
    for args in [
        ("bad_id", "node_created", {"x": 1}),
        ("node:p:t", "node_exploded", {"x": 1}),
        ("node:p:t", "node_created", {}),
    ]:
        try:
            validate_event(*args)
        except ValueError:
            continue
        raise AssertionError(f"validate_event accepted invalid args: {args}")


def test_graph_analysis_and_stats_tools():
    async def scenario():
        tools, resources, embedder = _build_tools()
        first = await _call(
            tools,
            "add_event",
            aggregate_id="node:sprint1:task_a",
            event_type="node_created",
            payload={
                "title": "Design schema",
                "project_id": "sprint1",
                "priority": 1,
                "complexity": 2,
                "tags": ["api"],
                "author": "alice",
            },
        )
        assert isinstance(first, str) and "-" in first

        await _call(
            tools,
            "add_event",
            aggregate_id="node:sprint1:task_b",
            event_type="node_created",
            payload={
                "title": "Build API",
                "project_id": "sprint1",
                "priority": 2,
                "complexity": 5,
                "tags": ["api"],
                "author": "alice",
            },
        )
        await _call(
            tools,
            "add_event",
            aggregate_id="node:sprint1:task_c",
            event_type="node_created",
            payload={
                "title": "Write docs",
                "project_id": "sprint1",
                "priority": 3,
                "complexity": 1,
                "author": "bob",
            },
        )
        await _call(
            tools,
            "add_event",
            aggregate_id="node:sprint1:task_a",
            event_type="relation_added",
            payload={"source": "node:sprint1:task_a", "target": "node:sprint1:task_b"},
        )
        await _call(
            tools,
            "add_constraint",
            node_id="node:sprint1:task_b",
            constraint_type="dependency",
            depends_on=["node:sprint1:task_a"],
            priority=2,
        )

        similar = await _call(tools, "query_similar", node_id="node:sprint1:task_a")
        assert {row["node_id"] for row in similar} >= {
            "node:sprint1:task_b",
            "node:sprint1:task_c",
        }

        order = await _call(tools, "suggest_next_tasks", project_id="sprint1")
        assert order["total_tasks"] == 3
        assert order["suggested_order"].index("Design schema") < order[
            "suggested_order"
        ].index("Build API")

        structure = await _call(tools, "analyze_structure", project_id="sprint1")
        assert structure["total_nodes"] == 3
        assert "degree_heuristic" in structure["method"]

        await _call(
            tools,
            "add_event",
            aggregate_id="node:sprint1:task_c",
            event_type="node_deleted",
            payload={"reason": "cancelled"},
        )
        deleted = await _call(tools, "query_similar", node_id="node:sprint1:task_a")
        assert "node:sprint1:task_c" not in {row["node_id"] for row in deleted}

        await _call(
            tools,
            "add_event",
            aggregate_id="node:sprint1:task_c",
            event_type="node_restored",
            payload={"restored_by": "alice"},
        )
        assert "node:sprint1:task_c" in embedder.nodes

        stats = await resources["chronos://stats"]()
        assert "Schema version" in stats
        assert "Active nodes" in stats

    asyncio.run(scenario())


def test_memory_belief_and_consolidation_tools():
    async def scenario():
        tools, _, _ = _build_tools()
        jwt = await _call(
            tools,
            "remember",
            content="JWT tokens expire after 24 hours and refresh tokens live in Redis.",
            project="auth",
            tags=["auth", "jwt"],
        )
        migration = await _call(
            tools,
            "remember",
            content="Database migrations use Alembic and must run in a transaction.",
            project="auth",
            tags=["database"],
        )
        devops = await _call(
            tools,
            "remember",
            content="GitHub Actions deploys staging after main branch merges.",
            project="devops",
        )
        assert jwt["embedded"] is True
        assert migration["token_estimate"] > 0

        recall = await _call(
            tools, "recall", query="JWT authentication", project="auth", k=5
        )
        assert recall["results"][0]["id"] == jwt["id"]
        assert recall["total_tokens"] > 0

        updated = await _call(
            tools,
            "update_memory",
            memory_id=jwt["id"],
            content="JWT tokens expire after 12 hours; refresh tokens live in Redis.",
        )
        assert updated["status"] == "updated"
        refreshed = await _call(
            tools, "recall", query="JWT expire Redis", project="auth", k=3
        )
        assert "12 hours" in refreshed["results"][0]["content"]

        similar = await _call(
            tools, "query_similar_memories", memory_id=jwt["id"], project="auth"
        )
        assert similar["source_id"] == jwt["id"]
        assert all(row["memory_id"] != jwt["id"] for row in similar["results"])

        forgotten = await _call(tools, "forget", memory_id=devops["id"])
        assert forgotten["status"] == "forgotten"
        after_forget = await _call(tools, "recall", query="GitHub Actions")
        assert devops["id"] not in {row["id"] for row in after_forget["results"]}

        invalid_time = await _call(
            tools, "query_at", query="JWT", timestamp="not-a-date"
        )
        assert "error" in invalid_time
        future = await _call(
            tools,
            "query_at",
            query="JWT",
            timestamp="2099-01-01T00:00:00",
            project="auth",
        )
        assert future["count"] > 0

        confidence = await _call(
            tools,
            "set_confidence",
            memory_id=jwt["id"],
            confidence=0.9,
            reason="integration test",
        )
        assert confidence["new_confidence"] == 0.9
        health = await _call(tools, "get_memory_health", memory_id=jwt["id"])
        assert health["confidence"] == 0.9

        consolidation = await _call(
            tools, "consolidate_memories", project="auth", auto_merge=False
        )
        assert consolidation["phases_run"] == [
            "orient",
            "gather",
            "consolidate",
            "prune",
        ]

    asyncio.run(scenario())


def test_causal_analysis_and_cycle_reporting():
    async def scenario():
        tools, _, _ = _build_tools()
        for index in range(20):
            team = "a" if index < 10 else "b"
            await _call(
                tools,
                "add_event",
                aggregate_id=f"node:causal:item_{index:03d}",
                event_type="node_created",
                payload={
                    "project_id": "causal",
                    "team": team,
                    "complexity": index % 5,
                    "score": index * 2 if team == "a" else index,
                },
            )

        causal = await _call(
            tools,
            "analyze_causal",
            treatment_filter={"team": "a"},
            outcome_metric="score",
            confounder_keys=["complexity"],
        )
        assert causal["confounder_used"] == "complexity"
        assert causal["n"] > 0
        assert causal["status"] in {"hypothesis", "observational"}

        missing = await _call(
            tools,
            "analyze_causal",
            treatment_filter={"team": "a"},
            outcome_metric="score",
            confounder_keys=["missing"],
        )
        assert "error" in missing

        solver = ConstraintSolver()
        ordered = solver.solve_next_actions(
            [
                {"id": "A", "depends_on": ["B"], "priority": 1},
                {"id": "B", "depends_on": ["A"], "priority": 1},
                {"id": "C", "depends_on": [], "priority": 2},
            ]
        )
        warnings = [task for task in ordered if "_cycle_warning" in task]
        assert [task["id"] for task in ordered][:1] == ["C"]
        assert {task["id"] for task in warnings} == {"A", "B"}

    asyncio.run(scenario())
