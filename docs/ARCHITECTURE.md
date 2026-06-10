# CHRONOS v4.0 — Architecture

## Overview

ChronosMCP is a single-process MCP server backed by a single SQLite file. The
core product is a **temporal memory layer**: BM25 full-text recall, version-aware
time-travel queries, and an FSRS-based cognitive subsystem (confidence,
retention decay, dream consolidation). An experimental event-sourced graph layer
ships behind the `CHRONOS_ENABLE_GRAPH` flag.

The runtime model is a **single-threaded asyncio event loop** via FastMCP. All
tool calls are serialized by the event loop; there is no concurrent mutation of
shared state. There are no background threads, no external services, and no
network access.

The defining v4.0 design decision: **all state lives in SQLite, including the
search index.** v3.x maintained an in-memory TF-IDF index that had to be rebuilt
on startup and manually kept in sync with the database on every write — with a
documented crash window where the two disagreed. v4.0 replaces it with an FTS5
virtual table maintained by database triggers, so index updates are part of the
same transaction as the content writes they mirror. Consistency is structural,
not procedural: application code cannot forget to update the index, because
application code doesn't update the index.

---

## Startup Sequence

```
chronos_mcp.py
  1. init_db()                      — DDL + column migrations + FTS table/
                                      triggers + one-time FTS backfill, WAL mode
  2. MemoryStore()                  — stateless; no load step
  3. BeliefEngine()                 — stateless FSRS/confidence math
  4. ConsolidationEngine(beliefs)   — dream consolidation
  5. [flag] HyperbolicEmbedder      — only when CHRONOS_ENABLE_GRAPH=1
  6. register(...)                  — wire tool handlers to FastMCP
  7. mcp.run()                      — enter asyncio event loop
```

There is no index rebuild at startup. A v3.x database upgrades in place: column
migrations are idempotent `ALTER TABLE`s, and the FTS backfill inserts only rows
not already indexed.

---

## Data Flow

### Write path (remember / update / forget / restore / purge)

```
tool handler → MemoryStore / lifecycle.py → INSERT/UPDATE/DELETE on memories
                                                  │
                                  (same transaction, via trigger)
                                                  ▼
                                            memories_fts
```

Triggers (`trg_memories_ai/au/ad` in db.py) are the **only** write path into
`memories_fts`. An UPDATE re-indexes the row from scratch (delete + conditional
re-insert when `forgotten = 0`), which makes forget/restore index maintenance
automatic and crash-safe.

`update_memory` additionally snapshots the old content into `memory_versions`
before overwriting — this is what funds time-travel.

`purge_memory` (lifecycle.py) hard-deletes content from `memories` (trigger
cleans FTS), `memory_versions`, `belief_updates`, and `search_feedback`, then
writes a content-free stub to `purge_log`.

### Read path (recall)

```
recall(query)
  → search.match_query()       free text → quoted OR-terms (injection-safe)
  → SELECT ... FROM memories_fts JOIN memories ... ORDER BY bm25(...)
  → FSRS re-rank               score × (1 + w · retention · confidence_factor)
  → optional compression       ONLY when token_budget is set
```

### Time-travel path (query_at)

```
query_at(query, timestamp)
  → select memories with created_at <= t, excluding forgotten-before-t
  → resolve each memory's content as of t via memory_versions
  → rank the snapshot in an ephemeral in-memory FTS5 table (same BM25 +
    porter pipeline as live recall, so scores are comparable)
```

---

## Module Responsibilities

Each file has one stated responsibility. No file may exceed 400 lines
(enforced by the code-warden lint gate during development).

### Entry Point

**`chronos_mcp.py`** — Orchestration only. Builds singletons, gates the graph
layer on `CHRONOS_ENABLE_GRAPH`, calls `register()`, runs the loop.

### Infrastructure

**`chronos/db.py`** — Connection management (`get_db()`), schema DDL, column
migrations, FTS5 table + sync triggers, `init_db()`. `db_path()` resolves
`CHRONOS_DB_PATH` at call time so tests can repoint databases. Fails loudly at
startup if the Python build lacks FTS5.

**`chronos/search.py`** — All retrieval: `match_query()` sanitization,
`search_memories()` (live BM25 over the join), `rank_snapshot()` (ephemeral
FTS5 for time-travel), `token_counter()`/`cosine_similarity()` (duplicate
detection), `estimate_tokens()`.

**`chronos/uuid7.py`** — RFC 9562 UUIDv7 (CSPRNG randomness, `time_ns()`).

**`chronos/validation.py`** — Graph event schema validation (whitelist, strict).

### Memory Pipeline

**`chronos/memory.py`** — `MemoryStore`: remember, recall (with FSRS re-rank
and opt-in compression), get, update. Stateless; delegates lifecycle and
time-travel.

**`chronos/lifecycle.py`** — forget (soft, reason persisted), restore, purge
(hard, all tables).

**`chronos/time_travel.py`** — `query_at()`: snapshot reconstruction + version
resolution + ephemeral ranking.

**`chronos/compression.py`** — Three stateless tiers (trim / compact /
summarize) applied progressively, only when the caller sets `token_budget`.
Tier-3 thresholds are derived from the budget.

### Cognitive Subsystem

**`chronos/beliefs.py`** — `BeliefEngine`: confidence operations with audit
logging (`belief_updates`), FSRS retention math
(`retention = (1 + days/(9·S))^(-1)`), review cycle, search-feedback stats.

**`chronos/consolidation.py`** — `ConsolidationEngine`: the mutating phases —
merge duplicates (keep higher confidence), decay stale confidence, prune.

**`chronos/consolidation_scan.py`** — The read-only phases: orient (health
snapshot), gather (O(N²) pairwise cosine duplicate detection), retention flags,
prune candidates.

**`chronos/consolidation_config.py`** — Tunable thresholds.

### Tool Registration

**`chronos/tools.py`** — Core memory tools (remember, recall, get_memory,
forget, restore_memory, purge_memory, query_at) + stats resource + `register()`.

**`chronos/memory_tools.py`** — update_memory, related_memories.

**`chronos/belief_tools.py`** — Confidence + FSRS review tools.

**`chronos/consolidation_tools.py`** — consolidate_memories.

### Graph Layer (flag-gated, experimental)

**`chronos/geometry.py`** — Poincaré ball ops + `HyperbolicEmbedder` node index.
Features must arrive pre-scaled to [0,1] with fixed scales (v4.0 fix — the v3.x
per-vector min-max made proportional payloads embed identically).

**`chronos/graph_tools.py`** — add_event (resize runs *before* the write
transaction — v4.0 deadlock fix), query_similar, add_constraint.

**`chronos/analyzers.py` / `chronos/analysis_tools.py`** — propensity-score
matching, topological task ordering, connectivity analysis.

---

## Schema (v4.0)

| Table | Purpose |
|---|---|
| `memories` | Content + project/tags/source/forget_reason + FSRS columns |
| `memories_fts` | FTS5 index (porter), trigger-synced — never written by app code |
| `memory_versions` | Old content snapshots; funds query_at |
| `belief_updates` | Audit trail of every confidence change |
| `search_feedback` | Which recall results were actually used |
| `purge_log` | Content-free record of hard deletions |
| `events`, `embeddings`, `constraints`, `tombstones`, `causal_results` | Graph layer |

All IDs are UUIDv7 (time-ordered). Migrations are additive-only; v3.x databases
upgrade in place.

---

## Concurrency Model

Single writer by design. WAL mode is enabled for read concurrency, but the
server assumes it is the only writer to its database file. Within the process,
the asyncio loop serializes tool calls; sqlite work is synchronous, which is
acceptable for the single-client stdio transport (a multi-client transport
would need `asyncio.to_thread()` offloading — see KNOWN_LIMITATIONS).

The one place two connections interact — the graph layer's embedding resize —
runs strictly before the event-insert transaction opens (v4.0 fix; the v3.x
ordering deadlocked at the first resize threshold).
