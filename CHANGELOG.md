# Changelog

All notable changes to ChronosMCP are documented here. Entries are ordered newest-first within each version.

---

## v3.1 — 2026-03-30

Schema version bump from 3.0 → 3.1. Comprehensive audit pass covering error handling, data integrity, input boundaries, resource lifecycle, and API contract compliance. 13 bugs confirmed via reproduction scripts; all critical and medium-severity issues fixed.

### Adversarial Audit Fixes (2026-03-30)

**memory.py — SQLite placeholder limit crash in `query_at()`**
The `WHERE memory_id IN (?, ?, ...)` clause in the version history query generated one placeholder per candidate memory. SQLite's `SQLITE_MAX_VARIABLE_NUMBER` limit (999 on older builds, 32,766 on newer) would cause an `OperationalError` crash if enough memories existed. Fix: batched into chunks of 900 IDs per query.

**memory.py — Timezone-aware timestamps broke time-travel comparison**
If a caller passed a tz-aware ISO timestamp to `query_at()` (e.g., `"2026-03-01T00:00:00+05:30"`), the `+05:30` suffix caused incorrect lexicographic comparison against tz-naive stored timestamps. Fix: `query_at()` now strips timezone info from the parsed timestamp before using it in SQL, normalizing to tz-naive for consistent comparison.

### Final-Pass Audit Fixes (2026-03-30)

**tfidf.py — Silent deletion on all-stop-word update** (CRITICAL)
Previously, calling `add_document(id, text)` where the new text tokenized to zero tokens (all stop words) would remove the old index entry and return early, silently deleting the document from the search index while the DB record survived. Fix: moved the token-empty check to before the `remove_document()` call. If new content produces no tokens, the old entry is preserved.

**graph_tools.py — `node_restored` dimension mismatch** (HIGH)
When restoring a tombstoned node, the vector was loaded from DB via raw `np.frombuffer` without pad/truncate. If `maybe_resize()` had run between the node's deletion and restoration, the restored vector had the wrong dimension, causing a numpy broadcast `ValueError` during distance computation. Fix: applied the same pad/truncate logic that `load_from_db()` uses.

**tools.py, graph_tools.py, memory_tools.py — Negative and zero k values** (MEDIUM)
`k=0` caused empty results where at least one result was expected. `k=-1` in TF-IDF caused `ranked[:-1]` behavior (returned all but the last result — wrong). `query_similar` had no bounds checking at all. Fix: added `k = max(1, min(k, N))` at every tool entry point — recall/query_at/query_similar_memories to [1, 20], query_similar to [1, 50].

**memory.py — Garbage timestamps accepted silently** (MEDIUM)
`query_at()` passed the `timestamp` parameter directly to a SQL `WHERE created_at <= ?` clause with no format validation. SQLite performed string comparison, producing unpredictable results for inputs like `"not_a_date"` with no error. Fix: added `datetime.fromisoformat()` validation with a structured error response.

**analysis_tools.py — Causal error detail silently swallowed** (MEDIUM)
When `simple_match()` returned an error response (missing confounder, no matches within caliper), `analyze_causal` accessed the result fields and built a success-shaped response without checking for or propagating the `error` key. The error message was lost, and the zero-ATE result was written to `causal_results` as if it were a valid analysis. Fix: check for `error` key before persisting; return error dict directly.

**analyzers.py — Circular dependencies silently dropped** (MEDIUM)
The topological sort in `ConstraintSolver.solve_next_actions()` would silently omit any tasks involved in dependency cycles. Callers had no way to detect that tasks had been dropped. Fix: detect unprocessed tasks after the sort loop, append them to the result with a `_cycle_warning` field identifying them as participants in a cycle.

### Schema Additions (2026-03-30)

- `memory_versions` table: added composite index `idx_memory_versions_lookup (memory_id, valid_from, valid_to)` for efficient time-travel queries.

---

## v3.0 — 2026-03-29

Memory pipeline introduced. Outside review verification and fix pass.

### Outside Review Fixes

**db.py — `memory_versions.id` missing PRIMARY KEY constraint**
The column was declared `TEXT NOT NULL` without `PRIMARY KEY`. Duplicate version IDs could be silently inserted. Fixed: `id TEXT PRIMARY KEY`.

**memory.py — `query_at()` version table load unscoped**
Version history was loaded from the full `memory_versions` table rather than scoped to the candidate memory IDs. On large stores this loaded the entire version history into memory on every time-travel call. Fixed: added `WHERE memory_id IN (...)` parameterized query scoped to candidate IDs.

**validation.py — Regex allowed whitespace in aggregate segments**
`[^:]+` matched any non-colon character including spaces, tabs, and newlines. Fixed: changed to `[^\s:]+` to reject whitespace. Changed line terminator from `$` to `\Z` to prevent trailing newline bypass.

**validation.py — Payload checked only for truthiness**
`if not payload` accepts `None` and any other falsy value but not `{"x": None}`. Fixed: added `isinstance(payload, dict)` check: `if not isinstance(payload, dict) or not payload`.

### Memory Pipeline (v3.0 initial)

- Added `remember`, `recall`, `forget`, `query_at`, `update_memory`, `query_similar_memories` tools
- Added `MemoryStore`, `TFIDFIndex`, `MemoryEmbedder` modules
- Added `memories`, `memory_vectors`, `memory_versions` tables
- Added `chronos://stats` resource
- Added `update_memory` content versioning via `memory_versions` table

---

## v2.3 — 2026-03-28

Core graph pipeline. Initial audit and bug fix pass.

### Audit Fixes (2026-03-28)

**analyzers.py — Propensity score normalization was per-group** (Bug #1)
Treatment and control groups were normalized independently, centering both at 0. This caused spurious matches when groups had wildly different confounder distributions (e.g., treatment complexity [1,2,3] matched control [7,8,9] as if they were equivalent). Fixed: pooled mean and std computed across both groups combined before normalization.

**analysis_tools.py — Constraint JOIN duplication** (Bug #2)
`suggest_next_tasks()` used a LEFT JOIN that multiplied rows when a node had multiple constraints. `data.update()` silently overwrote the first constraint's `depends_on` list. Fixed: separated event and constraint queries; aggregated constraints in Python with list extension.

**geometry.py — Embedding resize not persisted** (Bug #3)
`maybe_resize()` padded in-memory vectors to the new dimension but did not update the DB. After a server restart, the DB still had old-dimension vectors, causing dimension mismatches. Fixed: all resized vectors are written to the DB in a single transaction inside `maybe_resize()`.

**memory.py — `query_at()` returned current content** (Bug #4)
`query_at()` excluded memories created after the timestamp but returned their current content, not the content they had at that time. Historical queries were correct about which memories existed but wrong about what they said. Fixed: added `memory_versions` table; `update()` snapshots old content before overwriting; `query_at()` resolves historical content via version matching.

**analysis_tools.py — Relations fetched globally in `analyze_structure`** (Bug #5)
All `relation_added` events were fetched without project scoping, causing cross-project edges to bleed into structural analysis. Fixed: filtered edges to `project_node_ids` set before building the adjacency list.

**geometry.py — Per-element clipping violated Poincaré ball constraint** (Bug #6)
`dist()` used `np.clip` per-element, which does not enforce `||x|| < 1` in high dimensions. A vector with many small-but-nonzero elements can have all elements in (0,1) while its norm exceeds 1.0. Fixed: replaced with `_clip_norm()` which clips by total vector norm, preserving direction. Added `max(1.0, arg)` clamp on the arccosh argument for float safety.

**db.py — WAL mode not enabled** (Bug #7)
SQLite default journal mode causes writer-blocks-readers locking behavior. Fixed: added `PRAGMA journal_mode=WAL` in `init_db()`.

**analysis_tools.py — `relation_removed` not handled in `analyze_structure`**
Deleted edges persisted in graph analysis. Fixed: subtracted `relation_removed` events from `active_edges` set.

### Initial Architecture (v2.3)

- Core event sourcing model: `events`, `embeddings`, `causal_results`, `constraints`, `tombstones` tables
- Poincaré ball hyperbolic geometry with adaptive dimensionality (§4.1)
- CausalAnalyzer: greedy propensity-score matching (§5.2)
- StructureAnalyzer: iterative DFS connected components + degree bottleneck
- ConstraintSolver: greedy topological sort
- UUIDv7 for all IDs (RFC 9562)
- Module split: tools.py → graph_tools.py, analysis_tools.py, memory_tools.py (all under 400-line limit)
