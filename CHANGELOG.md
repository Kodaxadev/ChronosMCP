# Changelog

All notable changes to ChronosMCP are documented here. Entries are ordered newest-first within each version.

---

## v4.2.0 — 2026-06-10

### At-rest encryption (opt-in)

`CHRONOS_DB_KEY` + `pip install "chronosmcp[encryption]"` encrypts the entire
database — content, version history, FTS index, and WAL — with SQLCipher
(4.12 community). Plaintext mode is untouched: without a key the stdlib
`sqlite3` driver is used and nothing changes.

- **Driver:** `sqlcipher3-wheels` (prebuilt wheels including Windows;
  `sqlcipher3-binary` is manylinux-only — verified empirically).
- **Performance:** SQLCipher's PBKDF2 costs ~300ms per connection (measured
  320ms vs 0.9ms plaintext) and Chronos connects per operation. The KDF now
  runs once at startup: the key is re-derived locally from the header salt,
  verified, and cached in SQLCipher's raw `x'<key><salt>'` form — measured
  ~1ms per connection after that, with a passphrase-per-connection fallback
  for nonstandard cipher settings.
- **Fail-loud UX:** wrong key, plaintext-file-with-key, and
  encrypted-file-without-key all fail at startup with specific guidance.
- **Migration:** `scripts/encrypt_db.py` (encrypt / decrypt / rekey) with
  row-count verification, a `.bak` of the original, and WAL checkpointing.
  Keys come from the env or a getpass prompt — never argv.
- **Tests:** 8 new (round-trip under encryption incl. FTS5+porter, raw-key
  fast path, all three mismatch cases, quoted passphrases, migration
  round-trip, rekey). CI gains a dedicated encryption job on Linux + Windows.
- FTS5, triggers, time-travel, FSRS, and semantic search are unaffected —
  encryption is below the page layer.
- Residual exposure documented honestly in KNOWN_LIMITATIONS (env-var key,
  RAM, pre-encryption artifacts, no key recovery).

---

## v4.1.0 — 2026-06-10

### Hybrid semantic search (opt-in)

`CHRONOS_SEMANTIC=1` + `pip install "chronosmcp[semantic]"` upgrades recall
from lexical BM25 to **hybrid retrieval**: BM25 candidates and local-embedding
nearest neighbors (fastembed / ONNX bge-small-en-v1.5, ~70MB downloaded once,
no API keys, no torch) merged with Reciprocal Rank Fusion. "car" now finds a
memory about an *automobile* with zero lexical overlap. Off by default —
flag-off behavior is byte-identical to v4.0 and asserted by test.

Design notes:
- Hybrid retrieval, not re-rank-only: re-ranking can't fix the synonym
  problem when BM25 returns zero candidates to re-rank.
- RRF fusion (k=60) avoids score-normalization between BM25 and cosine.
- Vectors persist in the new `memory_embeddings` table with a content hash;
  startup backfill (stale-checked) covers memories written or edited while
  the flag was off. Writes are write-through; purge removes vectors.
- Brute-force numpy cosine — single-digit ms at personal scale, documented
  ceiling in KNOWN_LIMITATIONS.
- Recall pipeline extracted to `chronos/ranking.py` (memory.py stays a thin
  facade under the 400-line limit).
- Tests inject a deterministic fake embedder (CI downloads nothing); a
  real-model smoke test runs where the extra is installed. Verified locally
  end-to-end with the real ONNX model.

Honesty update: the README "no network" claim is now "no network by default" —
the semantic flag's one-time model download is the sole, opt-in exception.

---

## v4.0.0 — 2026-06-10

Major release driven by a full adversarial audit of v3.3 (code quality, security,
competitive landscape). Three confirmed bugs fixed with regression tests, the
search engine replaced, the graph layer quarantined, and the project repositioned
around its two genuinely differentiated capabilities: time-travel and FSRS-based
forgetting. **Breaking changes** are marked.

### Search engine: TF-IDF → SQLite FTS5/BM25

The in-memory pure-Python TF-IDF index is gone. Search now uses an FTS5 virtual
table (`memories_fts`, porter stemming) maintained by SQLite **triggers**, so index
updates commit atomically with the content writes they mirror. This removes, by
construction, the v3.3 documented "DB and index can disagree until restart"
window, the O(N) startup rebuild, and the manual index-update calls scattered
through every write path. Ranking is BM25; recall scores are now relative ranking
values rather than [0,1] cosines (**breaking** if you compared scores across
versions). Existing databases are backfilled automatically on first start.

### Confirmed v3.3 bugs fixed (each with a regression test)

- **`add_event` resize deadlock.** `maybe_resize()` opened a second connection and
  committed while `add_event` held the write lock — the first resize past the
  257-node threshold failed with `database is locked` (verified by repro). The
  resize now runs before the write transaction opens.
- **Silent recall truncation.** v3.3 trimmed any result over ~150 tokens even when
  no budget was requested, and offered no tool to fetch full content — long
  memories were unreadable through the MCP interface. Compression now runs only
  when `token_budget` is set, and the new `get_memory` tool returns full content
  by id. Tier-3 summarize thresholds are now budget-derived.
- **Forgotten memories leaking from the vector index.** Consolidation merge/prune
  removed TF-IDF entries but not structural vectors, so `query_similar_memories`
  kept returning forgotten memories forever. The structural vector index is
  removed entirely (see below); forgotten rows leave search via trigger in the
  same transaction.
- **Degenerate embedding normalization.** Per-vector min-max scaling mapped any
  two proportional node payloads to the identical embedding. Replaced with fixed
  per-feature scales.

### New tools

- `get_memory` — full single-memory retrieval, never truncated.
- `restore_memory` — un-forget (v3.3's error messages referenced a restore that
  didn't exist).
- `purge_memory` — irreversible hard-delete of content from every table
  (memories, versions, FTS, audit rows), leaving a content-free `purge_log` stub.
  `forget` vs `purge` is now an honest distinction.
- `related_memories` — BM25 "more like this" using a memory's own distinctive
  vocabulary. **Replaces `query_similar_memories`** (breaking), whose structural
  5-feature similarity surfaced same-*length* memories, not related ones.

### Security

- `source` provenance column on memories (`user`/`claude`/`web`/`document`),
  echoed in recall results; tool descriptions direct the model to treat
  externally-sourced content as data, not instructions (memory-poisoning tripwire).
- `forget(reason=...)` is now persisted (`forget_reason` column) instead of
  being silently discarded.
- FTS MATCH input is sanitized to quoted terms — query-syntax injection is
  structurally impossible; all SQL remains parameterized.
- README gained an explicit threat-model section (plaintext at rest, forget vs
  purge, no-network guarantee).

### Graph layer quarantined (**breaking**)

`add_event`, `query_similar`, `add_constraint`, `analyze_causal`,
`suggest_next_tasks`, and `analyze_structure` are now registered only when
`CHRONOS_ENABLE_GRAPH=1`. The audit found the structural-similarity premise weak
and the causal tooling rarely usable on real payloads; the layer stays available
for users who want it, but the default server is the focused memory product.

### Architecture

- `MemoryStore` is stateless; `chronos/search.py` owns retrieval,
  `chronos/lifecycle.py` owns forget/restore/purge, `chronos/consolidation_scan.py`
  owns read-only consolidation phases. `tfidf.py` and `mem_embed.py` deleted.
- `db_path()` resolves `CHRONOS_DB_PATH` at call time (testability).
- Schema migrations are additive and idempotent; pre-v4 databases upgrade in
  place on first start.

### Tooling

- Test suite rewritten: 4 → 48 tests across 8 files, including regression tests
  for every audit finding and FTS injection attempts.
- GitHub Actions CI: Linux + Windows × Python 3.10/3.12/3.14, ruff + pytest.

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
