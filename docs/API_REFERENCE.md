# CHRONOS v4.0 — API Reference

All tools are exposed via the MCP protocol. Parameters map directly to MCP tool
call arguments. Return values are JSON-serializable dicts unless noted.
Graph-layer tools exist only when the server runs with `CHRONOS_ENABLE_GRAPH=1`.

---

## Memory Tools

### `remember`

Store a free-text memory. Indexed for search in the same transaction.

| Param | Type | Default | Notes |
|---|---|---|---|
| `content` | str | required | Non-empty free text |
| `project` | str | `"default"` | Logical grouping for scoped recall |
| `tags` | list[str] | `[]` | Keyword labels |
| `source` | str | `"user"` | Provenance: `user` / `claude` / `web` / `document` |

Returns `{id, project, source, token_estimate}`.

### `recall`

BM25 full-text search (porter-stemmed) with FSRS-aware re-ranking.

| Param | Type | Default | Notes |
|---|---|---|---|
| `query` | str | required | Natural language; sanitized before FTS |
| `project` | str | None | Restrict to one project |
| `k` | int | 5 | Clamped to [1, 20] |
| `recency_weight` | float | 0.3 | 0.0 = pure relevance; boost = retention × confidence |
| `token_budget` | int | None | When set, progressive compression (trim → drop → summarize). When omitted, content is **never** truncated |

Returns `{results: [{id, project, content, score, token_estimate, source,
confidence}], total_tokens, count, query}` plus `compression_applied` when a
budget was set. Scores are relative ranking values (not [0,1]).

### `get_memory`

Fetch one memory in full by id — never truncated. Works on forgotten memories.

Returns `{id, project, content, tags, source, created_at, updated_at,
confidence, forgotten, forget_reason, version_count, token_estimate}` or
`{id, status: "not_found"}`.

### `update_memory`

Replace content; old content is snapshotted to `memory_versions` (time-travel
keeps working). Rejected for forgotten memories.

Returns `{id, status, token_estimate}`; on failure `{id, status: "error", detail}`.

### `related_memories`

"More like this": uses the memory's own distinctive vocabulary as a BM25 query.

| Param | Type | Default |
|---|---|---|
| `memory_id` | str | required |
| `k` | int | 5 (max 20) |
| `project` | str | None |

Returns `{results: [{memory_id, project, score, content_preview}], count, source_id}`.

### `query_at`

Time-travel recall: ranks the memories that existed at `timestamp`, with each
memory's content resolved to what it said at that moment.

| Param | Type | Default | Notes |
|---|---|---|---|
| `query` | str | required | |
| `timestamp` | str | required | ISO 8601; invalid input returns an `error` field |
| `project` | str | None | |
| `k` | int | 5 | Clamped to [1, 20] |

Returns the recall shape plus `as_of`.

### `forget`

Soft-delete: removed from search (transactionally), content retained for audit
and time-travel. The reason is persisted.

Returns `{id, status, reason}` — status `forgotten` / `not_found` /
`already_forgotten`.

### `restore_memory`

Reverse a forget; the memory reappears in recall immediately.

Returns `{id, status}` — `restored` / `not_forgotten` / `not_found`.

### `purge_memory`

**Irreversible** hard-delete: removes content from `memories`, `memory_versions`,
`belief_updates`, `search_feedback`, and the FTS index; writes a content-free
stub to `purge_log`. Use for "actually delete this," not routine cleanup.

Returns `{id, status, versions_removed, purged_at}`.

---

## Cognitive Tools

### `set_confidence` / `boost_confidence` / `weaken_confidence`

Direct or evidence-driven trust adjustment. Confidence lives in [0.01, 0.99]
(clamped — Bayesian: never fully certain). Boost adds +0.10, weaken subtracts
0.15 (asymmetric caution). Every change writes an audit row to `belief_updates`.

Each returns `{memory_id, old_confidence, new_confidence, reason}`.

### `review_memory`

FSRS review: multiplies stability (`easy` ×3.25 / `good` ×2.5 / `hard` ×1.5)
and adjusts difficulty. Reviewed memories decay slower in ranking and
consolidation.

Returns `{memory_id, quality, old_stability, new_stability, old_difficulty,
new_difficulty, review_count}`.

### `get_memory_health`

Full cognitive state of one memory.

Returns `{memory_id, confidence, stability, difficulty, retention,
days_since_review, review_count, forgotten}`. `retention` is the FSRS
forgetting-curve probability `(1 + days/(9·S))^(-1)`.

### `log_search_feedback`

Record whether a recall result was actually used. `used=True` also counts as
an FSRS review (stability boost).

Returns `{feedback_id, memory_id, used}`.

### `search_effectiveness`

Hit-rate stats over a window (`days`, default 30, clamped to [1, 365]).
Returns `{total_searches, results_used, hit_rate, window_days, recommendation}`
(recommendation appears after 20+ samples).

### `consolidate_memories`

Four-phase dream pass. Dry-run by default.

| Param | Type | Default | Notes |
|---|---|---|---|
| `project` | str | None | None = all projects |
| `auto_merge` | bool | False | True merges duplicate pairs (keeps higher confidence) |
| `auto_prune` | bool | False | True soft-deletes confidence<0.10 AND retention<0.15 |

Returns `{timestamp, project, phases_run, orient, gather, consolidate, prune}`.

---

## Graph Tools (require `CHRONOS_ENABLE_GRAPH=1`)

### `add_event`

Append an event to the project graph. `aggregate_id` format:
`node:{project}:{id}` (also `sprint:`/`team:`). Event types:
`node_created`, `node_updated`, `node_deleted`, `node_restored`,
`relation_added`, `relation_removed`, `relation_updated`. Node events embed
payload features (priority, tag count, author, complexity) at fixed scales.

Returns the event id (uuid7 string).

### `query_similar`

Nearest nodes by payload-feature distance (NOT content similarity — use
`related_memories` for that). `k` clamped to [1, 50]; tombstoned nodes excluded.

Returns `[{node_id, distance}]`, ascending distance.

### `add_constraint`

Store an ordering constraint. Only `constraint_type="dependency"` is enforced
by `suggest_next_tasks`; other types are stored with an `enforced: false`
warning.

Returns `{constraint_id, enforced}`.

### `suggest_next_tasks`

Topological sort of a project's tasks by dependency constraints, tie-broken by
priority. Returns `{suggested_order, rationale, total_tasks, ready_now}`.
Cyclic tasks are flagged rather than silently dropped.

### `analyze_structure`

Connected components + degree-based bottleneck heuristic (not full TDA).
Returns `{total_nodes, connected_components, bottlenecks, isolated_nodes,
method, recommendation}`.

### `analyze_causal`

Greedy propensity-score matching over `node_created` payloads.
`status` tiers by matched pairs: `hypothesis` (<10), `observational` (10–29),
`counterfactual_validated` (30+ — read as "enough pairs", not formal
identification). Returns `{result_id, ate, n, status, confounder_used,
interpretation}` or `{error, ...}`.

---

## Resources

### `chronos://stats`

Plain-text system snapshot: active/forgotten/purged memory counts, FTS index
size, version snapshots, average confidence, belief/feedback counts, graph
layer status, schema version.
