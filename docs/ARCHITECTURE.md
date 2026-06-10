# CHRONOS v3.3 â€” Architecture

## Overview

ChronosMCP is a single-process MCP server backed by a single SQLite file. It exposes three interconnected pipelines â€” a free-text memory layer, a structured event graph, and a cognitive belief/maintenance subsystem â€” that share the same database, server instance, and singleton lifecycle. There are no background threads, no external services, and no inter-process communication.

The runtime model is a **single-threaded asyncio event loop** via FastMCP. All tool calls are serialized by the event loop; there is no concurrent mutation of shared in-memory state.

---

## Startup Sequence

```
chronos_mcp.py
  1. init_db()                        â€” Apply DDL + column migrations, enable WAL mode
  2. HyperbolicEmbedder(dim=32)       â€” Create node embedding index
  3. embedder.load_from_db()          â€” Restore node vectors from DB
  4. CausalAnalyzer()                 â€” Stateless analyzer instance
  5. ConstraintSolver()               â€” Stateless solver instance
  6. StructureAnalyzer()              â€” Stateless analyzer instance
  7. TFIDFIndex()                     â€” Create empty in-memory text index
  8. MemoryEmbedder(dim=32)           â€” Create memory vector index
  9. MemoryStore(tfidf, mem_embedder) â€” Bind memory layer
  10. mem_store.load()                â€” Rebuild TF-IDF from DB + load memory vectors
  11. BeliefEngine()                  â€” Stateless confidence/FSRS engine
  12. ConsolidationEngine(beliefs, tfidf) â€” Dream consolidation engine
  13. register(mcp, ...)              â€” Wire all tool handlers to FastMCP instance
  14. mcp.run()                       â€” Enter asyncio event loop
```

All singletons are constructed before `mcp.run()`. Tool handlers capture singletons via closure, not module globals. This makes the dependency graph explicit and avoids import-time side effects.

---

## Module Responsibilities

Each file has exactly one stated responsibility. No file may exceed 400 lines.

### Entry Point

**`chronos_mcp.py`** (62 lines) â€” Orchestration only. Constructs singletons in dependency order, calls `register()`, calls `mcp.run()`. No domain logic.

### Infrastructure

**`chronos/db.py`** (131 lines) â€” SQLite connection context manager (`get_db()`), schema DDL, and `init_db()`. WAL mode is set once at startup. Connections are opened per-operation and closed in `finally` blocks. No connection pooling â€” acceptable for single-threaded asyncio.

**`chronos/uuid7.py`** (26 lines) â€” RFC 9562 UUIDv7 generation. Uses `time.time_ns()` (no float precision loss) and `secrets.randbits()` (CSPRNG). All IDs across all tables are UUIDv7.

**`chronos/validation.py`** (57 lines) â€” Event schema validation. Validates `aggregate_id` format, `event_type` whitelist, and payload type. Strict mode â€” raises `ValueError` on any violation, no silent coercion.

### Memory Pipeline

**`chronos/memory.py`** (~353 lines) - `MemoryStore`: the primary interface between MCP tools and the storage/indexing layer. Owns `remember()`, `recall()`, `forget()`, `update()`, and delegates `query_at()`. Coordinates writes between the `memories` DB table, the `TFIDFIndex`, and the `MemoryEmbedder`. v3.3: recall now supports `token_budget` parameter with three-tier progressive compression.

**`chronos/time_travel.py`** (~132 lines) â€” Extracted from memory.py to keep modules focused and under the 400-line limit. Owns the `query_at()` time-travel query logic and batched version history fetching.

**`chronos/compression.py`** (~170 lines) â€” Token-budget compression for recall results. Three stateless tiers: Tier 1 (Trim) truncates oversized individual results >150 tokens; Tier 2 (Compact) drops lowest-scored results to fit budget; Tier 3 (Summarize) replaces long content with extract + metadata stub. Called from `recall()` when `token_budget` is set. All thresholds independently derived.

**`chronos/tfidf.py`** (217 lines) â€” `TFIDFIndex`: in-memory TF-IDF document index. Pure Python + numpy. Maintains term-frequency counters and cached IDF values. Rebuilt from DB on startup. Updated incrementally on remember/forget/update. No persistence of its own â€” relies on the `memories` table as the source of truth.

**`chronos/mem_embed.py`** (203 lines) â€” `MemoryEmbedder`: maps memory content metadata (word count, unique term ratio, tag density, project bucket, recency) into a 5-dimensional hyperbolic vector. Backed by the `memory_vectors` table. Enables `query_similar_memories()`. Structural similarity only â€” not semantic.

### Graph Pipeline

**`chronos/geometry.py`** (185 lines) â€” `PoincareBall` (MÃ¶bius addition, exponential map, hyperbolic distance) and `HyperbolicEmbedder` (embed, remove, nearest-neighbor, adaptive resize). The PoincarÃ© ball uses curvature `c=1.0`. Distance computation uses norm-based clipping (not per-element) to enforce the `||x|| < 1` constraint in high dimensions.

**`chronos/analyzers.py`** (265 lines) â€” Three stateless analytical engines:
- `CausalAnalyzer` â€” greedy propensity-score matching with pooled normalization
- `StructureAnalyzer` â€” iterative DFS connected components + degree-based bottleneck heuristic
- `ConstraintSolver` â€” greedy topological sort with cycle detection

### Cognitive Subsystem (v3.2)

**`chronos/beliefs.py`** (240 lines) â€” `BeliefEngine`: confidence scoring with Bayesian updates and FSRS-6 forgetting curve math. Provides `compute_retention()` (FSRS decay), `set_confidence()` / `boost_confidence()` / `weaken_confidence()` (Bayesian updates with audit logging), `record_review()` (FSRS stability/difficulty update), and `log_feedback()` / `get_feedback_stats()` (search meta-learning). All state lives in DB â€” the engine is stateless. Inspired by ai-iq's beliefs system, simplified to use columns on the existing `memories` table instead of a separate table constellation.

**`chronos/consolidation.py`** (~397 lines) - `ConsolidationEngine`: on-demand memory maintenance modeled on biological REM sleep. v3.3 four-phase model: (1) Orient - snapshot memory health metrics; (2) Gather - find near-duplicates via TF-IDF cosine similarity > 0.85; (3) Consolidate - merge duplicates, decay stale confidence, flag low retention; (4) Prune - auto-forget memories with both confidence < 0.10 AND retention < 0.15. Called explicitly via MCP tool, not as a background process - consistent with the single-threaded asyncio model.

### MCP Tool Registration

**`chronos/tools.py`** (~230 lines) â€” Registers `remember`, `recall` (with `token_budget`), `forget`, `query_at`, the `chronos://stats` resource, and calls `register_graph_tools()`, `register_analysis_tools()`, `register_memory_tools()`. The `register()` function is the single wiring point called from `chronos_mcp.py`.

**`chronos/graph_tools.py`** (169 lines) â€” Registers `add_event`, `query_similar`, `add_constraint`. Owns the embedding lifecycle for nodes: embed on create/update, tombstone on delete, restore on node_restored.

**`chronos/analysis_tools.py`** (245 lines) â€” Registers `analyze_causal`, `suggest_next_tasks`, `analyze_structure`. Fetches from DB, delegates to the relevant analyzer, persists results.

**`chronos/memory_tools.py`** (103 lines) â€” Registers `update_memory` and `query_similar_memories`. Separated from `tools.py` when that file reached the module size limit.

**`chronos/belief_tools.py`** (130 lines) â€” Registers `set_confidence`, `boost_confidence`, `weaken_confidence`, `review_memory`, `get_memory_health`, `log_search_feedback`, `search_effectiveness`. All delegate to `BeliefEngine`.

**`chronos/consolidation_tools.py`** (55 lines) â€” Registers `consolidate_memories`. Delegates to `ConsolidationEngine`.

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

-- Soft deletes â€” never removed
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

-- v3.2: Cognitive subsystem tables

-- Audit trail for confidence changes
belief_updates (
    id             TEXT PRIMARY KEY,   -- UUIDv7
    memory_id      TEXT NOT NULL,
    old_confidence REAL NOT NULL,
    new_confidence REAL NOT NULL,
    reason         TEXT NOT NULL,      -- e.g. 'confirmed', 'dream_decay', 'manual'
    updated_at     TEXT NOT NULL
)

-- Search feedback for meta-learning
search_feedback (
    id          TEXT PRIMARY KEY,      -- UUIDv7
    query       TEXT NOT NULL,
    memory_id   TEXT NOT NULL,
    used        INTEGER NOT NULL,      -- 1 = result was used, 0 = skipped
    recalled_at TEXT NOT NULL
)
```

**v3.2 column additions to `memories`:** `confidence REAL DEFAULT 0.5`, `stability REAL DEFAULT 1.0`, `difficulty REAL DEFAULT 0.5`, `last_reviewed TEXT`, `review_count INTEGER DEFAULT 0`. Applied via idempotent ALTER TABLE on startup.

Indexes: `idx_memories_project (project, forgotten)`, `idx_memories_created_at (created_at)`, `idx_memory_vectors_project (project)`, `idx_memory_versions_lookup (memory_id, valid_from, valid_to)`.

---

## Data Flow: remember() â†’ recall()

```
remember(content, project, tags)
  â”‚
  â”œâ”€â”€ INSERT INTO memories (DB commit)
  â”œâ”€â”€ TFIDFIndex.add_document()         â€” tokenize + update in-memory tf/df counts
  â””â”€â”€ MemoryEmbedder.embed_and_store()  â€” 5-feature vector â†’ PoincarÃ© ball â†’ memory_vectors

recall(query, project, k, recency_weight)
  â”‚
  â”œâ”€â”€ SELECT forgotten ids              â€” build exclusion set
  â”œâ”€â”€ SELECT memory metadata            â€” id, project, content, created_at (all active)
  â”œâ”€â”€ TFIDFIndex.query()                â€” cosine TF-IDF ranking, returns top k*3 candidates
  â”œâ”€â”€ Apply recency boost               â€” score *= (1 + weight * 1/(1+days_old))
  â”œâ”€â”€ Re-rank and slice to k
  â””â”€â”€ Return {results, total_tokens, count, query}
```

## Data Flow: add_event() â†’ query_similar()

```
add_event(aggregate_id, node_created, payload)
  â”‚
  â”œâ”€â”€ validate_event()                  â€” format + whitelist checks
  â”œâ”€â”€ INSERT INTO events
  â”œâ”€â”€ embedder.maybe_resize()           â€” grow dim if N crosses threshold
  â”œâ”€â”€ embedder.embed()                  â€” 4-feature vector â†’ PoincarÃ© ball
  â”œâ”€â”€ INSERT OR REPLACE INTO embeddings
  â””â”€â”€ db.commit()

query_similar(node_id, k)
  â”‚
  â”œâ”€â”€ get_tombstoned_ids()              â€” exclusion set from tombstones table
  â””â”€â”€ embedder.nearest()               â€” O(N) linear scan over in-memory node dict
```

---

## Data Flow: FSRS-Aware Recall with Compression (v3.3)

```
recall(query, project, k, recency_weight, token_budget)
  â”‚
  â”œâ”€â”€ SELECT forgotten ids + FSRS columns (confidence, stability, last_reviewed)
  â”œâ”€â”€ TFIDFIndex.query()               â€” cosine TF-IDF ranking, top k*3 candidates
  â”œâ”€â”€ For each candidate:
  â”‚     â”œâ”€â”€ If stability is set â†’ FSRS boost:
  â”‚     â”‚     retention = (1 + days/(9*stability))^(-1)
  â”‚     â”‚     conf_factor = 0.5 + 0.5 * confidence
  â”‚     â”‚     boost = retention * conf_factor
  â”‚     â””â”€â”€ Else â†’ legacy recency: 1/(1+days_old)
  â”‚     score *= (1 + recency_weight * boost)
  â”œâ”€â”€ Re-rank and slice to k
  â”œâ”€â”€ If token_budget set OR any result >150 tokens:
  â”‚     â”œâ”€â”€ Tier 1: Trim oversized individual results (>150 tok)
  â”‚     â”œâ”€â”€ Tier 2: Drop lowest-scored results to fit budget (if over)
  â”‚     â””â”€â”€ Tier 3: Summarize remaining long results to stubs (if still over)
  â””â”€â”€ Return {results, total_tokens, count, query, compression_applied}
```

## Data Flow: Dream Consolidation (v3.3 â€” Four-Phase)

```
consolidate_memories(project, auto_merge, auto_prune)
  â”‚
  â”œâ”€â”€ Phase 1: Orient
  â”‚     â”œâ”€â”€ Count active memories, avg confidence
  â”‚     â”œâ”€â”€ Compute retention distribution (high/medium/low/critical buckets)
  â”‚     â””â”€â”€ Count stale memories (>30 days without review)
  â”‚
  â”œâ”€â”€ Phase 2: Gather
  â”‚     â”œâ”€â”€ Build temporary TFIDFIndex from active memories
  â”‚     â””â”€â”€ Pairwise similarity check (query each doc, threshold=0.85)
  â”‚
  â”œâ”€â”€ Phase 3: Consolidate
  â”‚     â”œâ”€â”€ If auto_merge: keep higher-confidence, forget duplicate, boost survivor
  â”‚     â”œâ”€â”€ Decay stale: subtract 0.05 confidence for >30-day unreviewed (floor 0.01)
  â”‚     â””â”€â”€ Flag low retention: FSRS retention < 0.30 â†’ needs review
  â”‚
  â””â”€â”€ Phase 4: Prune
        â”œâ”€â”€ Find memories with confidence < 0.10 AND retention < 0.15
        â””â”€â”€ If auto_prune: soft-delete them (reason: dream_prune)
```

## Hyperbolic Geometry Details

The PoincarÃ© ball model with curvature `c=1` is used throughout. Key properties:

**Distance:** `d(x,y) = (1/âˆšc) Â· arccosh(1 + 2cÂ·||x-y||Â² / ((1-cÂ·||x||Â²)(1-cÂ·||y||Â²)))`

The implementation clips vectors by norm (not per-element) before distance computation to enforce `||x|| < 1`. The arccosh argument is clamped to `â‰¥ 1.0` for float safety.

**Embedding:** Features are min-max scaled to [0,1], zero-padded to the current dimension, then projected to the ball with `||x|| < 0.95`. This preserves relative magnitudes â€” nodes with more distinctive payloads end up further from the origin (closer to the ball boundary), acting as "hubs" in the hierarchy.

**Adaptive dimension:** `dim = max(32, min(128, ceil(4 * log2(N))))` where N is node count. The minimum of 32 is maintained for all N < 257, so adaptive resizing is effectively inactive until a project accumulates hundreds of nodes.

---

## Causal Analysis

The causal pipeline implements Â§5.2 greedy propensity-score matching:

1. Fetch all `node_created` events, split into treatment/control by `treatment_filter`
2. Resolve confounder field (caller-specified, or auto-detected from defaults: `size`, `complexity`, `priority`, `effort`, `weight`)
3. Normalize confounders using **pooled** mean/std across both groups (not independent â€” prevents spurious cross-group matches)
4. Greedy 1:1 matching with caliper of 0.5 standard deviations (Austin 2011)
5. ATE = mean(treatment outcomes) - mean(control outcomes) over matched pairs
6. Status: `hypothesis` (<10 pairs), `observational` (10-29), `counterfactual_validated` (â‰¥30)

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
