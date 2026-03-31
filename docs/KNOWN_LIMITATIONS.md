# CHRONOS v3.1 — Known Limitations

This document gives an honest accounting of where ChronosMCP diverges from the CHRONOS v2.3 specification, what is not implemented, and what trade-offs have been made deliberately. It is not a bug list — it describes boundaries of the current design.

---

## Specification Gaps

### §4 — Hyperbolic Embeddings

**What is implemented:** Poincaré ball geometry (Möbius addition, exponential map, distance), HyperbolicEmbedder with adaptive dimensionality, nearest-neighbor linear scan.

**What is not implemented:** Riemannian gradient descent (§4.3). Embeddings are placed by a deterministic feature-to-ball mapping (min-max scale → zero-pad → project_to_ball), not by iterative optimization against a loss function. Nodes with similar features will be close; nodes that are "hierarchically related" in the domain sense will only be close if that relationship is captured in the 4 payload features used (priority, tag count, author bucket, complexity). The embedding does not learn from the event graph topology.

**Impact:** `query_similar` finds nodes with structurally similar payloads, not nodes that are semantically or topologically related in the knowledge graph.

### §5 — Causal Analysis

**What is implemented:** §5.2 greedy propensity-score matching with pooled normalization, caliper of 0.5 SD, and three status tiers (hypothesis / observational / counterfactual_validated).

**What is not implemented:** §5.1 full causal graph (DoWhy, NetworkX-based structural causal models). The greedy matching is a fallback path, not the primary path described in the spec. Instrumental variable estimation, regression discontinuity, and difference-in-differences are not implemented.

**Impact:** The `analyze_causal` tool is best understood as a propensity-score matching tool, not a full causal inference engine. Results labeled `counterfactual_validated` should be read as "sufficient matched pairs for observational study" — not as formal counterfactual identification.

### §6 — Constraint Solving

**What is implemented:** `dependency` constraint type via greedy topological sort. Circular dependencies detected and flagged. Priority-based tie-breaking.

**What is not implemented:** `uniqueness`, `temporal`, and `capacity` constraint types. These require the `python-constraint` backtracking solver described in §6.2. These types are accepted and stored by `add_constraint()` but have no effect on `suggest_next_tasks()` output. The response includes an `enforced: false` warning when non-dependency constraints are added.

**Impact:** Skill-match constraints, assignee uniqueness, and time-window constraints cannot be enforced with the current solver.

### §7 — Topological Data Analysis

**What is implemented:** Iterative DFS connected components + degree-based bottleneck heuristic. Fast, dependency-free, adequate for small graphs.

**What is not implemented:** The full Mapper nerve complex: gudhi `RipsComplex`, DBSCAN-based cover, Betti number computation, persistence diagrams, and persistent homology. These require the `gudhi` package which is not in the dependency list. The `analyze_structure` response includes `"method": "degree_heuristic (not full TDA/Mapper)"` to make this explicit.

**Impact:** `analyze_structure` detects graph disconnection and degree bottlenecks but cannot detect holes, voids, or higher-dimensional topological features in the data.

---

## Memory Pipeline Limitations

### TF-IDF is not semantic

`recall()` ranks by TF-IDF cosine similarity over tokenized content. It will not find a memory about "JWT token expiry" when queried for "authentication session timeout" unless those specific tokens overlap. No synonym resolution, no word vectors, no transformer embeddings.

For semantic similarity you would need to replace or augment the TF-IDF pipeline with an embedding model (e.g., via sentence-transformers). That would require an external dependency and model hosting.

### `query_similar_memories` is structural, not semantic

The 5-feature vector (word count, unique term ratio, tag count, project bucket, recency) captures content *structure*, not content *meaning*. Two memories about entirely different topics can be close if they have similar length, tag density, and project assignment. The docstring makes this explicit; callers should use `recall()` for semantic similarity.

### Token estimates are approximate

`token_estimate` uses a fixed ratio of 0.75 tokens per word. Actual token counts depend on the tokenizer of the consuming model, punctuation density, code content, and other factors. The estimate is a budget hint, not a precise count.

### All-stop-word content is not indexed

If a memory's content tokenizes to zero terms (all stop words — e.g., "the is and or"), it cannot appear in `recall()` results. The content is stored in the DB and retrievable via direct ID lookup, but TF-IDF search will not surface it. This is an inherent limitation of the stop-word filtering design, not a bug.

---

## Data Integrity Trade-offs

### DB commit and TF-IDF update are not atomic

In `remember()`, `update()`, and `forget()`, the DB write is committed first, then the TF-IDF index is updated. If the process crashes or the TF-IDF call throws after commit but before index update, the DB and index will be temporarily inconsistent. This resolves itself on the next server restart: `mem_store.load()` rebuilds the TF-IDF index from the DB, which is the source of truth.

During the session, a memory in this inconsistent state will be present in the DB but absent from `recall()` results. The `chronos://stats` resource will show a discrepancy between `Memories (active)` and `TF-IDF indexed`. A restart corrects it.

**Why not fix this with a transaction?** The TF-IDF index is a pure-Python in-memory structure with no rollback mechanism. Making DB and TF-IDF updates truly atomic would require either (a) a WAL-style redo log for the TF-IDF index, or (b) always building TF-IDF from DB on every recall. Option (b) would be correct but O(N) per call. The current design accepts the narrow inconsistency window in exchange for O(1) per-operation index updates and O(N) rebuild only at startup.

### maybe_resize() commits independently

When the node count crosses an embedding dimension threshold, `maybe_resize()` opens its own DB connection and commits the resized vectors. This commit is independent from the caller's (add_event's) transaction. If add_event subsequently fails and does not commit its event insert, the DB will have resized embeddings but no record of the event that triggered the resize. In practice this is harmless — the resize is idempotent, the embeddings remain valid, and the next add_event will succeed. But it means the resize and the triggering event are not in the same atomic unit.

### recall() loads all memory content per call

`recall()` fetches full memory content for all non-forgotten memories in the selected project (or all projects if unfiltered) into a Python dict before TF-IDF re-ranking. For personal use with hundreds of memories this is fast. At tens of thousands of long memories, this becomes the bottleneck. An optimized implementation would fetch only metadata (IDs, created_at) for ranking, then fetch full content only for the top-k results.

---

## Embedding Version Column

The `version` column in the `embeddings` table is always written as `1`. It was intended to track the embedding generation for detecting stale vectors, but was never wired to an incrementing counter. It currently carries no information.

---

## Synchronous Blocking in Async Handlers

All tool handlers are `async def` but execute `sqlite3` calls and CPU-bound numpy math synchronously. In the current single-client stdio MCP transport, this is a non-issue — the client sends one request at a time and waits for the response. If ChronosMCP were adapted to a multi-client transport (SSE, WebSocket), DB operations and heavy analytics (causal matching, structure analysis) would need to be offloaded to a thread pool via `asyncio.to_thread()` to avoid blocking the event loop for other clients.

## Algorithm Complexity at Scale

The constraint solver's `ready.sort()` inside its while loop makes the topological sort O(V² log V) instead of the optimal O(V + E). Replacing it with `heapq` would recover linear-time performance. At typical personal task management scale (tens to hundreds of tasks), this is sub-millisecond. At thousands of tasks it would become noticeable.

The causal matching algorithm is O(N × M) (treatment × control). No indexing or early-exit optimization is applied. For typical usage with tens to hundreds of nodes per group, this is fast. For thousands of nodes per group, a sorted-array + binary-search approach would be needed.

## Scale Considerations

ChronosMCP is designed for personal project memory and small team knowledge graphs. It is not designed for:

- High write throughput (>100 events/second) — SQLite WAL mode handles modest concurrency but is not a distributed write log
- Large-scale graph analytics (>100K nodes) — the `embedder.nearest()` linear scan is O(N); at 100K nodes with dim=128 this is ~12MB of vector data scanned per query
- Multi-user concurrent write loads — single-threaded asyncio serializes all writes, but multiple simultaneous users would contend on the single SQLite file
- Semantic search at scale — TF-IDF has no vocabulary generalization; for large corpora with diverse terminology, a vector embedding model is more appropriate

For the intended use case (one Claude instance, one project, hundreds to low thousands of memories and nodes), all current design choices are appropriate.
