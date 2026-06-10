# Changelog

## v3.3.0 — 2026-03-31

### Four-Phase Dream Consolidation

Restructured `consolidation.py` from three unnamed phases to an explicit four-phase model: Orient, Gather, Consolidate, Prune. Each phase has a clear responsibility and returns its own sub-report.

- **Orient** (new): Snapshots memory health before changes — total count, average confidence, retention distribution (high/medium/low/critical), stale count. Provides a baseline to measure consolidation effectiveness.
- **Gather** (renamed from Phase 1): Finds near-duplicate pairs via TF-IDF cosine similarity > 0.85. No action taken — findings feed into Consolidate.
- **Consolidate** (renamed, expanded): Merges duplicates (if `auto_merge=True`), decays stale confidence (-0.05 for >30-day unreviewed), flags low FSRS retention (<0.30).
- **Prune** (new): Auto-forgets memories with BOTH confidence < 0.10 AND retention < 0.15. Controlled by new `auto_prune` parameter (default False). Only the most degraded memories are affected — both untrusted AND nearly forgotten.

### Token-Budget Compression

New `chronos/compression.py` module with three-tier progressive compression, wired into the recall pipeline:

- **Tier 1 — Trim**: Truncates individual results exceeding 150 tokens. Always runs when any result is oversized.
- **Tier 2 — Compact**: Drops lowest-scored results to fit within the token budget. Only runs when `token_budget` is set and total exceeds it.
- **Tier 3 — Summarize**: Replaces remaining long content with truncated extract + metadata stub. Most aggressive tier — last resort.

New `token_budget` parameter on `recall()` tool. When set, compression is applied progressively. Response includes `compression_applied` field listing which tiers were used. Thresholds independently derived from Chronos's own token estimation data.

### Tool Docstring Audit

Hardened all 19+ MCP tool docstrings across 6 registration files. Docstrings are load-bearing — they're Claude's sole API documentation. Changes:

- `add_event`: Documented all 7 event types with payload requirements, return type.
- `query_similar`: Added distance range interpretation, parameter docs, return format.
- `add_constraint`: Already solid — minor clarification.
- `analyze_causal`: Added explicit return format for both success and error cases.
- `suggest_next_tasks`: Added return format, explained task matching requirements.
- `analyze_structure`: Added return format, documented edge scoping rules.
- `consolidate_memories`: Complete rewrite for four-phase model with workflow recommendations.
- `recall`: Added `token_budget` parameter docs with recommended range (2000–6000).

### Module Extraction

- Extracted `query_at()` from `memory.py` into `chronos/time_travel.py` to keep modules under the project line limit.

---

## v3.2.0

### Cognitive Belief Subsystem

- `chronos/beliefs.py` — BeliefEngine with FSRS-6 forgetting curve, Bayesian confidence scoring (asymmetric: +0.10 boost, -0.15 weaken), search feedback meta-learning.
- `chronos/belief_tools.py` — 7 new MCP tools: `set_confidence`, `boost_confidence`, `weaken_confidence`, `review_memory`, `get_memory_health`, `log_search_feedback`, `search_effectiveness`.
- FSRS-aware recall ranking: retention * confidence replaces simple recency. Backward-compatible — legacy fallback when FSRS columns are NULL.
- Schema: 5 new columns on `memories` table + `belief_updates` and `search_feedback` tables.
- `chronos/consolidation.py` — Three-phase dream consolidation (duplicate detection, confidence decay, retention flagging).

---

## v3.1.0

- Memory pipeline: `remember()`, `recall()`, `forget()`, `update()`, `query_at()`.
- Time-travel queries with version history.
- TF-IDF content retrieval with recency-weighted ranking.
- Memory vector embeddings via MemoryEmbedder for structural similarity.

---

## v2.3.0

- Knowledge graph: `add_event()`, `query_similar()`, `add_constraint()`.
- Hyperbolic embeddings in Poincaré ball (c=1.0, 32-dim, adaptive resize).
- Causal analysis via propensity-score matching.
- Constraint solver with topological sort.
- Graph structure analysis with connected components and bottleneck detection.
