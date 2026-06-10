# CHRONOS v4.0 — Known Limitations

An honest accounting of what ChronosMCP does not do, and which trade-offs are
deliberate. This is not a bug list — it describes boundaries of the current
design. (Several v3.x entries are gone because v4.0 removed the limitation
itself: the DB/index consistency window, the silent recall truncation, and the
unreachable-full-content problem no longer exist.)

---

## Search is lexical, not semantic

`recall()` ranks with BM25 over porter-stemmed tokens. Morphological variants
match ("deploying" ↔ "deploy"), but synonyms do not: a memory about "JWT token
expiry" will not surface for "authentication session timeout" unless tokens
overlap. This is the central trade-off of the zero-dependency design.

The intended escape hatch is an optional embedding back-end (e.g. local ONNX
sentence embeddings behind a flag) that re-ranks BM25 candidates. That work is
scoped but not built; until then, write memories using the vocabulary you
expect to search with.

## Token estimates are approximate

`token_estimate` uses a fixed 0.75 tokens/word ratio. Real token counts depend
on the consuming model's tokenizer, punctuation, and code density. Treat it as
a budget hint, not a count.

## BM25 scores are relative

Recall scores are `-bm25()` values re-scaled by the FSRS boost. They order
results correctly but are not normalized probabilities, and are not comparable
across different queries (only within one response). v3.x cosine scores were
[0,1]; if you stored thresholds against those, they no longer apply.

## Consolidation duplicate detection is O(N²)

`gather_duplicates` compares every active memory pair (precomputed token
counters, cheap dot products). At 1,000 memories that's ~500k comparisons —
fine on demand; at 50,000 memories it would need blocking/LSH. Personal-scale
assumption, explicitly.

## Synchronous blocking in async handlers

Tool handlers are `async def` but run sqlite and numpy work synchronously. For
the single-client stdio transport this is a non-issue (one request at a time).
A multi-client transport (SSE/WebSocket) would need `asyncio.to_thread()`
offloading for DB calls and the O(N²) consolidation pass.

## Single-writer assumption

WAL mode allows concurrent readers, but two server processes pointed at the
same `chronos.db` (e.g. Claude Desktop and Claude Code simultaneously) can
contend on writes. Writes are short transactions, so SQLite's busy timeout
usually absorbs the conflict — but the design assumes one writer, and nothing
enforces it.

## The graph layer is experimental for a reason

Behind `CHRONOS_ENABLE_GRAPH=1`:

- **`query_similar` measures payload-structure similarity, not meaning.**
  Nodes embed from 4 fixed-scale features (priority, tag count, author bucket,
  complexity). v4.0 fixed the normalization bug that made proportional payloads
  identical, but the deeper limitation stands: the Poincaré-ball machinery
  operates on near-origin vectors where its distances are ~Euclidean, and no
  Riemannian optimization is performed. It is a feature-vector index with
  hyperbolic distance, not a learned hierarchy.
- **`analyze_causal` is propensity-score matching, not causal identification.**
  `counterfactual_validated` means "≥30 matched pairs", nothing stronger. It
  also requires payloads to carry numeric confounder/outcome fields, which
  real-world task payloads rarely do.
- **Constraint types other than `dependency`** (`temporal`, `uniqueness`,
  `capacity`) are stored but not enforced; `add_constraint` warns accordingly.
- **`analyze_structure`** is connected-components + degree heuristics, not TDA.

These are kept because they're useful at the margin and honest about scope —
not because they compete with dedicated project tooling.

## Plaintext at rest

The database is unencrypted SQLite. `purge_memory` genuinely removes content
from every table Chronos controls, but cannot scrub OS-level artifacts (WAL
segments already checkpointed, filesystem snapshots, backups you've made).
Full at-rest encryption (SQLCipher or application-level) is possible but not
implemented; it would complicate the zero-dependency story.

## Embedding `version` column is vestigial

`embeddings.version` is always written as 1. Reserved for stale-vector
detection; never wired up. Graph layer only.

## Scale envelope

Designed for one Claude instance and personal/small-team scale: hundreds to
low thousands of memories, modest write rates. Known ceilings:

- recall is one indexed FTS query — fine to ~100k memories
- consolidation O(N²) — fine to a few thousand memories
- graph `nearest()` linear scan — fine to ~10k nodes
- no sharding, no replication, no multi-tenant isolation

Within that envelope, the current design choices are appropriate.
