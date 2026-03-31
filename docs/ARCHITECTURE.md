# CHRONOS v3.1 — Architecture

## Overview

ChronosMCP is a single-process MCP server backed by a single SQLite file. It exposes two independent pipelines — a free-text memory layer and a structured event graph — that share the same database, server instance, and singleton lifecycle. There are no background threads, no external services, and no inter-process communication.

The runtime model is a **single-threaded asyncio event loop** via FastMCP. All tool calls are serialized by the event loop; there is no concurrent mutation of shared in-memory state.

---

## Startup Sequence

```
chronos_mcp.py
  1. init_db()                        — Apply DDL, enable WAL mode
  2. HyperbolicEmbedder(dim=32)       — Create node embedding index
  3. embedder.load_from_db()          — Restore node vectors from DB
  4. CausalAnalyzer()                 — Stateless analyzer instance
  5. ConstraintSolver()               — Stateless solver instance
  6. StructureAnalyzer()              — Stateless analyzer instance
  7. TFIDFIndex()                     — Create empty in-memory text index
  8. MemoryEmbedder(dim=32)           — Create memory vector index
  9. MemoryStore(tfidf, mem_embedder) — Bind memory layer
  10. mem_store.load()                — Rebuild TF-IDF from DB + load memory vectors
  11. register(mcp, ...)              — Wire all tool handlers to FastMCP instance
  12. mcp.run()                       — Enter asyncio event loop
```

All singletons are constructed before `mcp.run()`. Tool handlers capture singletons via closure, not module globals. This makes the dependency graph explicit and avoids import-time side effects.

---

## Module Responsibilities

Each file has exactly one stated responsibility. No file may exceed 500 lines (hard limit; 400-line threshold triggers refactor review).

### Entry Point

**`chronos_mcp.py`** (62 lines) — Orchestration only. Constructs singletons in dependency order, calls `register()`, calls `mcp.run()`. No domain logic.

### Infrastructure

**`chronos/db.py`** (131 lines) — SQLite connection context manager (`get_db()`), schema DDL, and `init_db()`. WAL mode is set once at startup. Connections are opened per-operation and closed in `finally` blocks. No connection pooling — acceptable for single-threaded asyncio.

**`chronos/uuid7.py`** (26 lines) — RFC 9562 UUIDv7 generation. Uses `time.time_ns()` (no float precision loss) and `secrets.randbits()` (CSPRNG). All IDs across all tables are UUIDv7.

**`chronos/validation.py`** (57 lines) — Event schema validation. Validates `aggregate_id` format, `event_type` whitelist, and payload type. Strict mode — raises `ValueError` on any violation, no silent coercion.

### Memory Pipeline

**`chronos/memory.py`** (449 lines ⚠️) — `MemoryStore`: the primary interface between MCP tools and the storage/indexing layer. Owns `remember()`, `recall()`, `forget()`, `update()`, and `query_at()`. Coordinates writes between the `memories` DB table, the `TFIDFIndex`, and the `MemoryEmbedder`. Note: approaching 500-line hard limit — `query_at()` is a candidate for extraction into a dedicated `time_travel.py` module.

**`chronos/tfidf.py`** (217 lines) — `TFIDFIndex`: in-memory TF-IDF document index. Pure Python + numpy. Maintains term-frequency counters and cached IDF values. Rebuilt from DB on startup. Updated incrementally on remember/forget/update. No persistence of its own — relies on the `memories` table as the source of truth.

**`chronos/mem_embed.py`** (203 lines) — `MemoryEmbedder`: maps memory content metadata (word count, unique term ratio, tag density, project bucket, recency) into a 5-dimensional hyperbolic vector. Backed by the `memory_vectors` table. Enables `query_similar_memories()`. Structural similarity only — not semantic.

### Graph Pipeline

**`chronos/geometry.py`** (185 lines) — `PoincareBall` (Möbius addition, exponential map, hyperbolic distance) and `HyperbolicEmbedder` (embed, remove, nearest-neighbor, adaptive resize). The Poincaré ball uses curvature `c=1.0`. Distance computation uses norm-based clipping (not per-element) to enforce the `||x|| < 1` constraint in high dimensions.

**`chronos/analyzers.py`** (265 lines) — Three stateless analytical engines:
- `CausalAnalyzer` — greedy propensity-score matching with pooled normalization
- `StructureAnalyzer` — iterative DFS connected components + degree-based bottleneck heuristic
- `ConstraintSolver` — greedy topological sort with cycle detection

### MCP Tool Registration

**`chronos/tools.py`** (190 lines) — Registers `remember`, `recall`, `forget`, `query_at`, the `chronos://stats` resource, and calls `register_graph_tools()`, `register_analysis_tools()`, `register_memory_tools()`. The `register()` function is the single wiring point called from `chronos_mcp.py`.

**`chronos/graph_tools.py`** (169 lines) — Registers `add_event`, `query_similar`, `add_constraint`. Owns the embedding lifecycle for nodes: embed on create/update, tombstone on delete, restore on node_restored.

**`chronos/analysis_tools.py`** (245 lines) — Registers `analyze_causal`, `suggest_next_tasks`, `analyze_structure`. Fetches from DB, delegates to the relevant analyzer, persists results.

**`chronos/memory_tools.py`** (103 lines) — Registers `update_memory` and `query_similar_memories`. Separated from `tools.py` when that file reached the module size limit.

---

## Database Schema

All tables are created by `init_db()` via `CREATE TABLE IF NOT EXISTS`. Schema is idempotent on restart.

```sql
-- Core event log (append-only, event sourcing backbone)
events (
    id             TEXT PRIMARY KEY,   -- UUIDv7
    aggregate_id   TEXT NOT NULL,      -- format: {type}:{project}:{id}
    event_type     TEXT NOT NULL,      -- see validation.py VALID_EVENT_TYPES
    ts             TEXT NOT NULL,      -- ISO 8601 tz-naive datetime
    payload        TEXT NOT NULL,      -- JSON blob
    schema_version TEXT NOT NULL DEFAULT '2.3'
)

-- Node hyperbolic vectors (retained even after tombstone for causal validity)
embeddings (
    node_id  TEXT PRIMARY KEY,
    vector   BLOB NOT NULL,            -- np.float32 tobytes()
    version  INTEGER NOT NULL,         -- always 1 (future: increment on resize)
    dim      INTEGER NOT NULL
)

-- Soft deletes — never removed
tombstones (
    node_id    TEXT PRIMARY KEY,
    event_id   TEXT NOT NULL,
    deleted_at TEXT NOT NULL,
    reason     TEXT
)

-- Causal analysis results
causal_results (
    id         TEXT PRIMARY KEY,
    treatment  TEXT NOT NULL,          -- JSON of treatment_filter
    outcome    TEXT NOT NULL,
    ate        REAL NOT NULL,
    n_samples  INTEGER NOT NULL,
    status     TEXT NOT NULL
)

-- Dependency constraints for the solver
constraints (
    id              TEXT PRIMARY KEY,
    node_id         TEXT NOT NULL,
    constraint_type TEXT NOT NULL,
    priority        INTEGER NOT NULL,
    data            TEXT NOT NULL      -- JSON: {type, depends_on, priority}
)

-- Free-text memory store
memories (
    id         TEXT PRIMARY KEY,
    project    TEXT NOT NULL DEFAULT 'default',
    content    TEXT NOT NULL,
    tags       TEXT NOT NULL DEFAULT '[]',  -- JSON array
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    forgotten  INTEGER NOT NULL DEFAULT 0   -- soft delete flag
)

-- Hyperbolic content-structure vectors for memories
memory_vectors (
    memory_id  TEXT PRIMARY KEY,
    vector     BLOB NOT NULL,
    dim        INTEGER NOT NULL,
    project    TEXT NOT NULL DEFAULT 'default'
)

-- Version history for time-travel queries
memory_versions (
    id         TEXT PRIMARY KEY,       -- UUIDv7
    memory_id  TEXT NOT NULL,
    content    TEXT NOT NULL,          -- old content snapshot
    valid_from TEXT NOT NULL,          -- previous updated_at (or created_at)
    valid_to   TEXT NOT NULL           -- timestamp of the update that replaced it
)
```

Indexes: `idx_memories_project (project, forgotten)`, `idx_memories_created_at (created_at)`, `idx_memory_vectors_project (project)`, `idx_memory_versions_lookup (memory_id, valid_from, valid_to)`.

---

## Data Flow: remember() → recall()

```
remember(content, project, tags)
  │
  ├── INSERT INTO memories (DB commit)
  ├── TFIDFIndex.add_document()         — tokenize + update in-memory tf/df counts
  └── MemoryEmbedder.embed_and_store()  — 5-feature vector → Poincaré ball → memory_vectors

recall(query, project, k, recency_weight)
  │
  ├── SELECT forgotten ids              — build exclusion set
  ├── SELECT memory metadata            — id, project, content, created_at (all active)
  ├── TFIDFIndex.query()                — cosine TF-IDF ranking, returns top k*3 candidates
  ├── Apply recency boost               — score *= (1 + weight * 1/(1+days_old))
  ├── Re-rank and slice to k
  └── Return {results, total_tokens, count, query}
```

## Data Flow: add_event() → query_similar()

```
add_event(aggregate_id, node_created, payload)
  │
  ├── validate_event()                  — format + whitelist checks
  ├── INSERT INTO events
  ├── embedder.maybe_resize()           — grow dim if N crosses threshold
  ├── embedder.embed()                  — 4-feature vector → Poincaré ball
  ├── INSERT OR REPLACE INTO embeddings
  └── db.commit()

query_similar(node_id, k)
  │
  ├── get_tombstoned_ids()              — exclusion set from tombstones table
  └── embedder.nearest()               — O(N) linear scan over in-memory node dict
```

---

## Hyperbolic Geometry Details

The Poincaré ball model with curvature `c=1` is used throughout. Key properties:

**Distance:** `d(x,y) = (1/√c) · arccosh(1 + 2c·||x-y||² / ((1-c·||x||²)(1-c·||y||²)))`

The implementation clips vectors by norm (not per-element) before distance computation to enforce `||x|| < 1`. The arccosh argument is clamped to `≥ 1.0` for float safety.

**Embedding:** Features are min-max scaled to [0,1], zero-padded to the current dimension, then projected to the ball with `||x|| < 0.95`. This preserves relative magnitudes — nodes with more distinctive payloads end up further from the origin (closer to the ball boundary), acting as "hubs" in the hierarchy.

**Adaptive dimension:** `dim = max(32, min(128, ceil(4 * log2(N))))` where N is node count. The minimum of 32 is maintained for all N < 257, so adaptive resizing is effectively inactive until a project accumulates hundreds of nodes.

---

## Causal Analysis

The causal pipeline implements §5.2 greedy propensity-score matching:

1. Fetch all `node_created` events, split into treatment/control by `treatment_filter`
2. Resolve confounder field (caller-specified, or auto-detected from defaults: `size`, `complexity`, `priority`, `effort`, `weight`)
3. Normalize confounders using **pooled** mean/std across both groups (not independent — prevents spurious cross-group matches)
4. Greedy 1:1 matching with caliper of 0.5 standard deviations (Austin 2011)
5. ATE = mean(treatment outcomes) - mean(control outcomes) over matched pairs
6. Status: `hypothesis` (<10 pairs), `observational` (10-29), `counterfactual_validated` (≥30)

Errors (missing confounder, no matches within caliper) are returned as structured error responses, not stored in `causal_results`.

---

## Time-Travel Query

`query_at(query, timestamp)` reconstructs the memory state at an arbitrary past point:

1. Fetch all memories with `created_at <= timestamp` (memories that existed then)
2. Exclude memories with `forgotten = 1` where `updated_at <= timestamp` (they were deleted before the snapshot)
3. For each candidate, check `memory_versions` for a version active at `timestamp`: `valid_from <= timestamp < valid_to`
4. If a version matches, substitute that version's content for the current content
5. Build a temporary `TFIDFIndex` over the snapshot, rank against `query`, return results

This gives true content time-travel, not just existence filtering. A memory that was later edited shows its original content when queried at a past timestamp.

---

## Known Architectural Trade-offs

See [`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md) for a full accounting of where the implementation diverges from the CHRONOS v2.3 specification and what is not implemented.
