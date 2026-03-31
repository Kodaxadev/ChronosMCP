# test_integration.py
# End-to-end integration test for CHRONOS v2.3
# Drives every MCP tool directly, validates outputs, reports pass/fail.
# Run: python3 test_integration.py

import asyncio
import json
import math
import os
import sys
import tempfile

# ---------------------------------------------------------------------------
# Isolate to a temp DB so tests never touch the real chronos.db
# ---------------------------------------------------------------------------
_tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp_db.close()
os.environ["CHRONOS_DB_PATH"] = _tmp_db.name

# Now import — init_db() will target the temp file
from chronos.db import init_db
from chronos.geometry import HyperbolicEmbedder, calculate_dimension
from chronos.analyzers import CausalAnalyzer, ConstraintSolver, StructureAnalyzer
from chronos.mem_embed import MemoryEmbedder
from chronos.memory import MemoryStore
from chronos.tfidf import TFIDFIndex
from chronos.uuid7 import uuid7
from chronos.validation import validate_event

# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------
PASS = 0
FAIL = 0

def check(label: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        print(f"  ✓  {label}")
        PASS += 1
    else:
        print(f"  ✗  {label}{('  →  ' + detail) if detail else ''}")
        FAIL += 1


# ===========================================================================
# 1. DB INIT
# ===========================================================================
print("\n── 1. DB initialisation ─────────────────────────────────────────────")
init_db()
from chronos.db import get_db
with get_db() as db:
    tables = {r[0] for r in db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}

for tbl in ["events", "embeddings", "causal_results", "constraints", "tombstones", "memories", "memory_vectors"]:
    check(f"Table '{tbl}' exists", tbl in tables)


# ===========================================================================
# 2. UUIDv7
# ===========================================================================
print("\n── 2. UUIDv7 ────────────────────────────────────────────────────────")
ids = [uuid7() for _ in range(5)]
for uid in ids:
    parts = uid.split("-")
    check(f"Format 5 groups: {uid[:23]}…", len(parts) == 5)
    check(f"Version nibble = 7", parts[2][0] == "7")
    check(f"Variant nibble = 8/9/a/b", parts[3][0] in "89ab")

check("IDs are unique", len(set(ids)) == 5)

# Time-ordering only holds across distinct milliseconds.
# Generate two IDs 2ms apart to verify the timestamp portion sorts correctly.
import time as _time
uid_early = uuid7(); _time.sleep(0.002); uid_late = uuid7()
check("IDs are time-ordered across distinct ms", uid_early < uid_late,
      f"early={uid_early[:23]} late={uid_late[:23]}")


# ===========================================================================
# 3. EVENT VALIDATION
# ===========================================================================
print("\n── 3. Event validation ──────────────────────────────────────────────")

# Should pass
try:
    validate_event("node:myproject:task_001", "node_created", {"title": "Test"})
    check("Valid event passes", True)
except ValueError as e:
    check("Valid event passes", False, str(e))

# Bad aggregate_id
try:
    validate_event("bad_id", "node_created", {"x": 1})
    check("Bad aggregate_id raises", False)
except ValueError:
    check("Bad aggregate_id raises", True)

# Unknown event_type
try:
    validate_event("node:p:t", "node_exploded", {"x": 1})
    check("Unknown event_type raises", False)
except ValueError:
    check("Unknown event_type raises", True)

# Empty payload
try:
    validate_event("node:p:t", "node_created", {})
    check("Empty payload raises", False)
except ValueError:
    check("Empty payload raises", True)


# ===========================================================================
# 4. MCP TOOLS — wire up singletons, run tools directly
# ===========================================================================
print("\n── 4. MCP tools (add_event, query_similar, tombstone, restore) ──────")

from mcp.server.fastmcp import FastMCP
import chronos.tools as tools_mod
# MemoryEmbedder already imported at top of file

mcp_instance  = FastMCP("chronos-test")
emb           = HyperbolicEmbedder(dim=32)
emb.load_from_db()
_tfidf        = TFIDFIndex()
_mem_embedder = MemoryEmbedder(dim=32)
_mem          = MemoryStore(_tfidf, mem_embedder=_mem_embedder)
_mem.load()
# register() now wires all sub-modules: graph_tools, analysis_tools,
# memory tools (inline), and memory_tools.py (update_memory, query_similar_memories)
tools_mod.register(mcp_instance, emb, CausalAnalyzer(), ConstraintSolver(),
                   StructureAnalyzer(), _mem, _mem_embedder)

# Grab the registered async functions by name from the FastMCP tool registry
_tool_map = {t.name: t.fn for t in mcp_instance._tool_manager.list_tools()}

async def call(name, **kwargs):
    return await _tool_map[name](**kwargs)


async def run_tool_tests():
    # ---- add_event: node_created ----
    eid1 = await call("add_event",
        aggregate_id="node:testproject:task_001",
        event_type="node_created",
        payload={"title": "Fix login bug", "priority": 1,
                 "complexity": 3, "tags": ["auth"], "author": "alice"}
    )
    check("add_event returns UUIDv7", isinstance(eid1, str) and "-" in eid1)

    eid2 = await call("add_event",
        aggregate_id="node:testproject:task_002",
        event_type="node_created",
        payload={"title": "Write tests", "priority": 1,
                 "complexity": 3, "tags": ["auth"], "author": "alice"}
    )
    eid3 = await call("add_event",
        aggregate_id="node:testproject:task_003",
        event_type="node_created",
        payload={"title": "Deploy to prod", "priority": 9,
                 "complexity": 8, "tags": [], "author": "bob"}
    )

    # Both task_001 and task_002 share author/priority/complexity/tags
    # task_003 is very different — task_002 should be closer to task_001 than task_003 is
    similar = await call("query_similar", node_id="node:testproject:task_001", k=3)
    check("query_similar returns list", isinstance(similar, list))
    check("query_similar returns 2 neighbors (only 2 other nodes)", len(similar) == 2)

    ids_returned = [s["node_id"] for s in similar]
    distances    = {s["node_id"]: s["distance"] for s in similar}
    check("task_002 (similar attrs) closer than task_003 (different)",
          distances.get("node:testproject:task_002", 9999)
          < distances.get("node:testproject:task_003", 9999),
          f"distances: {distances}")

    # ---- relation_added ----
    await call("add_event",
        aggregate_id="node:testproject:task_001",
        event_type="relation_added",
        payload={"source": "node:testproject:task_001",
                 "target": "node:testproject:task_002"}
    )
    check("relation_added accepted", True)

    # ---- node_deleted + tombstone ----
    await call("add_event",
        aggregate_id="node:testproject:task_003",
        event_type="node_deleted",
        payload={"reason": "cancelled"}
    )
    similar_after_delete = await call("query_similar",
                                      node_id="node:testproject:task_001", k=5)
    deleted_ids = [s["node_id"] for s in similar_after_delete]
    check("Deleted node excluded from query_similar",
          "node:testproject:task_003" not in deleted_ids)

    with get_db() as db:
        tomb = db.execute(
            "SELECT * FROM tombstones WHERE node_id='node:testproject:task_003'"
        ).fetchone()
    check("Tombstone record written to DB", tomb is not None)

    # ---- node_restored ----
    await call("add_event",
        aggregate_id="node:testproject:task_003",
        event_type="node_restored",
        payload={"restored_by": "alice"}
    )
    with get_db() as db:
        tomb_after = db.execute(
            "SELECT * FROM tombstones WHERE node_id='node:testproject:task_003'"
        ).fetchone()
    check("Tombstone removed after node_restored", tomb_after is None)
    check("Restored node back in embedder index",
          "node:testproject:task_003" in emb.nodes)


asyncio.run(run_tool_tests())


# ===========================================================================
# 5. CONSTRAINT SOLVER + suggest_next_tasks
# ===========================================================================
print("\n── 5. Constraint solver + suggest_next_tasks ────────────────────────")

async def run_constraint_tests():
    # Add tasks with explicit project_id
    await call("add_event",
        aggregate_id="node:sprint1:task_a",
        event_type="node_created",
        payload={"title": "Design schema", "project_id": "sprint1",
                 "priority": 1, "complexity": 2}
    )
    await call("add_event",
        aggregate_id="node:sprint1:task_b",
        event_type="node_created",
        payload={"title": "Build API", "project_id": "sprint1",
                 "priority": 2, "complexity": 5}
    )
    await call("add_event",
        aggregate_id="node:sprint1:task_c",
        event_type="node_created",
        payload={"title": "Write docs", "project_id": "sprint1",
                 "priority": 3, "complexity": 1}
    )

    # task_b depends on task_a
    await call("add_constraint",
        node_id="node:sprint1:task_b",
        constraint_type="dependency",
        depends_on=["node:sprint1:task_a"],
        priority=2
    )

    result = await call("suggest_next_tasks", project_id="sprint1")
    check("suggest_next_tasks returns dict", isinstance(result, dict))
    check("total_tasks = 3", result.get("total_tasks") == 3, str(result))

    order = result.get("suggested_order", [])
    check("Returns up to 5 suggestions", len(order) <= 5)
    # Design schema must appear before Build API (dependency)
    if "Design schema" in order and "Build API" in order:
        check("Design schema before Build API (dependency respected)",
              order.index("Design schema") < order.index("Build API"))
    else:
        check("Both tasks present in suggestion", False, f"order={order}")


asyncio.run(run_constraint_tests())


# ===========================================================================
# 6. STRUCTURE ANALYZER + analyze_structure
# ===========================================================================
print("\n── 6. Structure analyzer ────────────────────────────────────────────")

async def run_structure_tests():
    result = await call("analyze_structure", project_id="sprint1")
    check("analyze_structure returns dict", isinstance(result, dict))
    check("total_nodes = 3", result.get("total_nodes") == 3, str(result))
    check("connected_components key present", "connected_components" in result)
    check("method field present", "degree_heuristic" in result.get("method", ""))

asyncio.run(run_structure_tests())


# ===========================================================================
# 7. CAUSAL ANALYZER + analyze_causal
# ===========================================================================
print("\n── 7. Causal analyzer ───────────────────────────────────────────────")

async def run_causal_tests():
    # Seed enough nodes with 'complexity' as confounder
    for i in range(20):
        tag = "team_a" if i < 10 else "team_b"
        score = i * 2 if i < 10 else i       # team_a has higher scores
        await call("add_event",
            aggregate_id=f"node:causaltest:node_{i:03d}",
            event_type="node_created",
            payload={"team": tag, "complexity": i % 5,
                     "score": score, "project_id": "causaltest"}
        )

    # Test: team_a vs team_b, outcome = score, confounder = complexity
    result = await call("analyze_causal",
        treatment_filter={"team": "team_a"},
        outcome_metric="score",
        confounder_keys=["complexity"]
    )
    check("analyze_causal returns dict", isinstance(result, dict))
    check("confounder_used = complexity",
          result.get("confounder_used") == "complexity", str(result))
    check("ATE is a number", isinstance(result.get("ate"), (int, float)))
    check("n matched > 0", result.get("n", 0) > 0, str(result))
    check("status is observational or hypothesis (10-19 pairs)",
          result.get("status") in ("observational", "hypothesis"), str(result))
    check("interpretation field present", "interpretation" in result)

    # Test: auto-detection picks 'complexity' when confounder_keys omitted
    result2 = await call("analyze_causal",
        treatment_filter={"team": "team_a"},
        outcome_metric="score"
    )
    check("Auto-detects complexity when no confounder_keys given",
          result2.get("confounder_used") == "complexity", str(result2))

    # Test: __missing__ error path
    result3 = await call("analyze_causal",
        treatment_filter={"team": "team_a"},
        outcome_metric="score",
        confounder_keys=["nonexistent_field"]
    )
    # Should either error gracefully or return hypothesis
    check("Missing confounder handled gracefully",
          "error" in result3 or result3.get("status") == "hypothesis", str(result3))


asyncio.run(run_causal_tests())


# ===========================================================================
# 8. ADAPTIVE DIMENSIONALITY (maybe_resize grow-only)
# ===========================================================================
print("\n── 8. Adaptive dimensionality ───────────────────────────────────────")

async def run_resize_tests():
    emb2 = HyperbolicEmbedder(dim=32)
    emb2.load_from_db()
    initial_dim = emb2.dim
    check(f"Starts at dim=32", initial_dim == 32)

    # Inject 260 synthetic nodes — at N=256 formula=32, at N=300 formula=33
    # so a grow should trigger somewhere past 256
    for i in range(260):
        feat = [float(i % 100), float(i % 10), 1.0, float(i % 7), 3.0]
        emb2.embed(f"synthetic_{i}", feat)

    grew = emb2.maybe_resize()
    check("Dimension grew after 260 nodes", grew and emb2.dim > 32,
          f"dim={emb2.dim}")
    check("All stored vectors match new dim",
          all(len(v) == emb2.dim for v in emb2.nodes.values()),
          f"dim={emb2.dim}")
    check("Dim never shrank below initial 32", emb2.dim >= 32)

asyncio.run(run_resize_tests())


# ===========================================================================
# 9. CHRONOS://STATS RESOURCE
# ===========================================================================
print("\n── 9. chronos://stats resource ──────────────────────────────────────")

# Resources use a different registry in FastMCP — call directly
from chronos.db import get_db as _get_db
from chronos.geometry import calculate_dimension as _calc_dim

async def run_stats_test():
    # Resource URIs are AnyUrl objects — compare via str()
    resource_map = {str(r.uri): r.fn
                    for r in mcp_instance._resource_manager.list_resources()}
    stats_fn = resource_map.get("chronos://stats")
    check("chronos://stats resource registered", stats_fn is not None)
    if stats_fn:
        stats = await stats_fn()
        check("Stats contains 'Schema version'", "Schema version" in stats)
        check("Stats contains 'Active nodes'", "Active nodes" in stats)
        check("Stats contains 'Embedding dim'", "Embedding dim" in stats)

asyncio.run(run_stats_test())


# ===========================================================================
# 10. MEMORY TOOLS — remember / recall / forget / query_at
# ===========================================================================
print("\n── 10. Memory tools ─────────────────────────────────────────────────")

async def run_memory_tests():
    # ---- remember ----
    r1 = await call("remember",
        content="The authentication service uses JWT tokens with a 24h expiry. "
                "Refresh tokens are stored in Redis with a 30-day TTL.",
        project="auth-service",
        tags=["auth", "jwt", "redis"]
    )
    check("remember returns id", "id" in r1 and "-" in r1["id"])
    check("remember returns token_estimate", r1.get("token_estimate", 0) > 0)
    check("remember returns indexed_terms > 0", r1.get("indexed_terms", 0) > 0)

    r2 = await call("remember",
        content="Database migrations run via Alembic. Always run in a transaction. "
                "Never modify existing migration files — create a new revision.",
        project="auth-service",
        tags=["database", "migrations", "alembic"]
    )
    r3 = await call("remember",
        content="The CI pipeline uses GitHub Actions. Tests must pass on PR. "
                "Deploy to staging happens automatically on merge to main.",
        project="devops",
        tags=["ci", "github-actions", "deploy"]
    )

    # ---- recall — content match ----
    results = await call("recall", query="JWT token authentication", project="auth-service", k=5)
    check("recall returns dict with results", isinstance(results.get("results"), list))
    check("recall returns total_tokens", results.get("total_tokens", 0) > 0)
    check("recall count matches results length", results["count"] == len(results["results"]))

    top_ids = [r["id"] for r in results["results"]]
    check("recall: JWT memory ranked first",
          len(top_ids) > 0 and top_ids[0] == r1["id"],
          f"top_ids={top_ids[:2]}, expected={r1['id'][:8]}")

    # ---- recall — project filter ----
    devops_results = await call("recall", query="pipeline deploy", project="devops", k=5)
    devops_projects = {r["project"] for r in devops_results["results"]}
    check("recall project filter: only devops memories returned",
          devops_projects <= {"devops"},
          f"projects={devops_projects}")

    # ---- recall — cross-project without filter ----
    all_results = await call("recall", query="authentication JWT", k=5)
    check("recall without project returns results", all_results["count"] > 0)

    # ---- recall token budget ----
    check("recall total_tokens is int", isinstance(all_results["total_tokens"], int))
    check("each result has token_estimate",
          all(isinstance(r.get("token_estimate"), int) for r in all_results["results"]))

    # ---- forget ----
    forget_result = await call("forget", memory_id=r3["id"], reason="test_cleanup")
    check("forget returns status=forgotten", forget_result.get("status") == "forgotten")

    after_forget = await call("recall", query="GitHub Actions CI pipeline", k=5)
    forgotten_ids = [r["id"] for r in after_forget["results"]]
    check("forgotten memory absent from recall", r3["id"] not in forgotten_ids)

    # ---- forget idempotent ----
    forget_again = await call("forget", memory_id=r3["id"])
    check("forget already_forgotten is graceful",
          forget_again.get("status") == "already_forgotten")

    # ---- forget unknown id ----
    forget_unknown = await call("forget", memory_id="nonexistent-id-xyz")
    check("forget unknown id returns not_found",
          forget_unknown.get("status") == "not_found")

    # ---- query_at time-travel ----
    import time as _t
    past_ts = "2020-01-01T00:00:00"   # before any memory was created
    snap_past = await call("query_at", query="JWT authentication", timestamp=past_ts)
    check("query_at past timestamp returns empty", snap_past["count"] == 0,
          f"count={snap_past['count']}")
    check("query_at returns as_of field", snap_past.get("as_of") == past_ts)

    # Future timestamp should return current memories
    future_ts = "2099-01-01T00:00:00"
    snap_future = await call("query_at", query="JWT authentication", timestamp=future_ts,
                             project="auth-service")
    check("query_at future timestamp returns memories", snap_future["count"] > 0,
          f"count={snap_future['count']}")
    check("query_at returns as_of field", snap_future.get("as_of") == future_ts)

asyncio.run(run_memory_tests())


# ===========================================================================
# 11. EXTENDED MEMORY TOOLS — update_memory, recency, query_similar_memories
# ===========================================================================
print("\n── 11. Extended memory tools ────────────────────────────────────────")

async def run_extended_memory_tests():
    # All tools registered in one register() call — tool map is already complete

    # ---- Store a memory we'll update ----
    r_orig = await call("remember",
        content="The rate limiter is set to 100 requests per minute per user.",
        project="api-service",
        tags=["rate-limit"]
    )
    orig_id = r_orig["id"]
    check("remember for update returns embedded=True", r_orig.get("embedded") is True,
          f"embedded={r_orig.get('embedded')}")

    # Verify vector was stored in DB
    with get_db() as db:
        vec_row = db.execute(
            "SELECT memory_id FROM memory_vectors WHERE memory_id = ?", (orig_id,)
        ).fetchone()
    check("memory_vectors row persisted after remember()", vec_row is not None,
          f"memory_id={orig_id[:12]}")

    # ---- update_memory ----
    update_result = await call("update_memory",
        memory_id=orig_id,
        content="The rate limiter is set to 200 requests per minute per user. "
                "Burst allowance is 50 extra requests.",
    )
    check("update_memory returns status=updated",
          update_result.get("status") == "updated", str(update_result))
    check("update_memory returns token_estimate",
          update_result.get("token_estimate", 0) > 0, str(update_result))

    # Confirm updated content shows up in recall
    updated_recall = await call("recall", query="rate limiter requests per minute",
                                project="api-service", k=3)
    top_content = updated_recall["results"][0]["content"] if updated_recall["results"] else ""
    check("recall returns updated content after update_memory",
          "200 requests" in top_content, f"content={top_content[:60]}")

    # ---- update_memory: error on unknown id ----
    err_result = await call("update_memory",
        memory_id="nonexistent-memory-id",
        content="should fail"
    )
    check("update_memory unknown id returns status=error",
          err_result.get("status") == "error", str(err_result))

    # ---- update_memory: error on forgotten id ----
    r_tmp = await call("remember",
        content="Temporary memory to test forget+update guard.",
        project="api-service"
    )
    await call("forget", memory_id=r_tmp["id"])
    err_forgotten = await call("update_memory",
        memory_id=r_tmp["id"],
        content="trying to update a forgotten memory"
    )
    check("update_memory on forgotten id returns status=error",
          err_forgotten.get("status") == "error", str(err_forgotten))

    # ---- Recency decay: two memories, one newer, query should prefer newer ----
    # We can't fake creation timestamps at insertion time, but we can verify
    # that recall() with recency_weight=0.0 gives identical ranking to TF-IDF
    # (no bias), while recency_weight=1.0 gives the same result for same-age memories.
    r_a = await call("remember",
        content="OAuth2 PKCE flow for mobile clients requires code_verifier.",
        project="auth-recency-test"
    )
    r_b = await call("remember",
        content="OAuth2 PKCE flow for web clients requires redirect_uri validation.",
        project="auth-recency-test"
    )

    results_no_decay = await call("recall",
        query="OAuth2 PKCE flow", project="auth-recency-test",
        k=5, recency_weight=0.0
    )
    check("recall with recency_weight=0 returns results",
          results_no_decay["count"] >= 2, str(results_no_decay["count"]))

    results_with_decay = await call("recall",
        query="OAuth2 PKCE flow", project="auth-recency-test",
        k=5, recency_weight=1.0
    )
    check("recall with recency_weight=1 returns same count",
          results_with_decay["count"] == results_no_decay["count"],
          f"no_decay={results_no_decay['count']} with_decay={results_with_decay['count']}")

    # Scores should be higher with recency boost (recently created, factor ≈ 1)
    if results_no_decay["results"] and results_with_decay["results"]:
        score_no_decay    = results_no_decay["results"][0]["score"]
        score_with_decay  = results_with_decay["results"][0]["score"]
        check("recency boost increases scores for recent memories",
              score_with_decay >= score_no_decay,
              f"no_decay={score_no_decay:.5f} with_decay={score_with_decay:.5f}")

    # ---- query_similar_memories ----
    # Seed a few more memories in same project so neighbor search has candidates
    r_c = await call("remember",
        content="OAuth2 authorization code flow for server-side apps uses client_secret.",
        project="auth-recency-test"
    )
    r_d = await call("remember",
        content="JWT tokens are signed with RS256 in production. Rotate keys every 90 days.",
        project="auth-recency-test"
    )

    similar = await call("query_similar_memories",
        memory_id=r_a["id"], k=3, project="auth-recency-test"
    )
    check("query_similar_memories returns dict with results",
          isinstance(similar.get("results"), list), str(similar))
    check("query_similar_memories count <= 3", similar.get("count", 999) <= 3)
    check("query_similar_memories source_id correct",
          similar.get("source_id") == r_a["id"], str(similar))

    # Results should be within the same project
    if similar["results"]:
        check("query_similar_memories result has distance field",
              "distance" in similar["results"][0])
        check("query_similar_memories result has content_preview",
              "content_preview" in similar["results"][0])
        check("source memory not in its own results",
              all(r["memory_id"] != r_a["id"] for r in similar["results"]))

    # ---- query_similar_memories: unknown id returns empty gracefully ----
    similar_unknown = await call("query_similar_memories",
        memory_id="nonexistent-memory-id", k=3
    )
    check("query_similar_memories unknown id returns empty results",
          similar_unknown.get("count", 999) == 0, str(similar_unknown))

    # ---- verify memory_vectors table grows with remember() calls ----
    with get_db() as db:
        vec_count = db.execute("SELECT COUNT(*) FROM memory_vectors").fetchone()[0]
    check("memory_vectors table has entries after remember() calls",
          vec_count > 0, f"count={vec_count}")

asyncio.run(run_extended_memory_tests())


# ===========================================================================
# Summary
# ===========================================================================
total = PASS + FAIL
print(f"\n{'─'*70}")
print(f"  Results: {PASS}/{total} passed  |  {FAIL} failed")
print(f"{'─'*70}\n")
sys.exit(0 if FAIL == 0 else 1)
