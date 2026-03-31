# CHRONOS v3.1 — API Reference

All tools are exposed via the MCP protocol. Parameters map directly to MCP tool call arguments. Return values are JSON-serializable dicts or lists unless noted.

---

## Memory Tools

### `remember`

Store a free-text memory for later retrieval.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `content` | `str` | required | Free-text string to store. No structure required. |
| `project` | `str` | `"default"` | Logical project grouping. Use consistent names across calls to enable project-scoped recall. |
| `tags` | `list[str]` | `[]` | Optional keyword labels for filtering. |

**Returns**

```json
{
  "id": "019526b7-...",
  "project": "auth-service",
  "token_estimate": 42,
  "indexed_terms": 187,
  "embedded": true
}
```

`token_estimate` — approximate tokens this memory will consume when recalled (~0.75 tokens/word).
`indexed_terms` — total documents now in the TF-IDF index.
`embedded` — `true` if the memory was also added to the vector index (always `true` in standard mode).

**Notes**
- `content` must be a non-empty string after stripping. Empty content raises a `ValueError` returned as MCP error.
- `id` is a UUIDv7 — time-ordered, safe to use as a chronological sort key.

---

### `recall`

Retrieve the most relevant memories for a natural language query.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `query` | `str` | required | Natural language question or topic. |
| `project` | `str` | `null` | If provided, restrict results to this project only. |
| `k` | `int` | `5` | Number of results to return. Clamped to [1, 20]. |
| `recency_weight` | `float` | `0.3` | Recency boost coefficient. Clamped to [0.0, 1.0]. `0.0` = pure TF-IDF ranking; `1.0` = strong recency preference. |

**Returns**

```json
{
  "results": [
    {
      "id": "019526b7-...",
      "project": "auth-service",
      "content": "JWT tokens should expire after 15 minutes...",
      "score": 0.04821,
      "token_estimate": 38
    }
  ],
  "total_tokens": 198,
  "count": 3,
  "query": "authentication token expiry"
}
```

`score` — TF-IDF cosine similarity with optional recency boost applied.
`total_tokens` — sum of all result token estimates plus ~40 overhead tokens.

**Notes**
- Ranking: `final_score = tfidf_score * (1 + recency_weight * (1 / (1 + days_old)))`
- If `recency_weight > 0`, the internal candidate set is `k * 3` before re-ranking, ensuring recency boost doesn't cause the top TF-IDF result to be missed.
- Forgotten memories are excluded automatically.
- Empty query returns `{"results": [], "total_tokens": 0, "count": 0}`.

---

### `forget`

Soft-delete a memory so it no longer appears in recall results.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `memory_id` | `str` | required | The `id` returned by `remember()` or `recall()`. |
| `reason` | `str` | `"manual"` | Optional explanation for the deletion. |

**Returns**

```json
{ "id": "019526b7-...", "status": "forgotten", "reason": "outdated" }
```

`status` is one of:
- `"forgotten"` — successfully soft-deleted
- `"already_forgotten"` — was already deleted; no change
- `"not_found"` — no memory with this ID exists

**Notes**
- The DB record is retained permanently for audit and `query_at()` time-travel.
- The memory is removed from the TF-IDF index and the memory vector index.
- Forgotten memories can be found via `query_at()` for timestamps before the deletion.

---

### `query_at`

Time-travel recall: retrieve memories as they existed at a past timestamp.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `query` | `str` | required | Natural language question, same as `recall()`. |
| `timestamp` | `str` | required | ISO 8601 datetime string, e.g. `"2026-03-01T00:00:00"`. Memories created after this time are excluded. |
| `project` | `str` | `null` | Optional project filter. |
| `k` | `int` | `5` | Number of results. Clamped to [1, 20]. |

**Returns**

Same shape as `recall()` plus an `as_of` field:

```json
{
  "results": [...],
  "total_tokens": 145,
  "count": 2,
  "query": "authentication",
  "as_of": "2026-03-01T00:00:00"
}
```

If the timestamp is invalid:
```json
{
  "results": [], "total_tokens": 0, "count": 0,
  "error": "Invalid ISO 8601 timestamp: 'not_a_date'",
  "as_of": "not_a_date"
}
```

**Notes**
- Memories forgotten before `timestamp` are excluded from the snapshot.
- Content returned reflects what the memory contained at `timestamp`, not its current content (uses `memory_versions` history).
- A fresh TF-IDF index is built over the snapshot for each call — this is O(snapshot_size) work.

---

### `update_memory`

Replace the content of an existing memory and re-index it.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `memory_id` | `str` | required | The `id` returned by `remember()` or `recall()`. |
| `content` | `str` | required | New free-text content to replace the existing entry. |

**Returns**

```json
{ "id": "019526b7-...", "status": "updated", "token_estimate": 55 }
```

On error:
```json
{ "id": "019526b7-...", "status": "error", "detail": "Memory '...' is forgotten — restore first" }
```

**Notes**
- The old content is snapshotted into `memory_versions` before overwriting, enabling `query_at()` to show historical content.
- Forgotten memories cannot be updated — call `remember()` with corrected content instead. This preserves audit integrity.
- The original `created_at` timestamp is preserved. Only `updated_at` and `content` change.
- The memory is re-indexed in TF-IDF and re-embedded in the vector index.

---

### `query_similar_memories`

Find memories structurally similar to a given memory using hyperbolic distance.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `memory_id` | `str` | required | ID of a stored memory. |
| `k` | `int` | `5` | Number of similar memories to return. Clamped to [1, 20]. |
| `project` | `str` | `null` | Optional project filter. |

**Returns**

```json
{
  "results": [
    {
      "memory_id": "019526b8-...",
      "distance": 0.0312,
      "content_preview": "Auth service should use RS256 not HS256 for..."
    }
  ],
  "count": 3,
  "source_id": "019526b7-..."
}
```

**Notes**
- Similarity is **structural**, not semantic. Features: word count, unique term ratio, tag count, project bucket, recency. Two memories with identical content length and tag count will be close regardless of topic.
- For semantic/keyword similarity, use `recall()` with a descriptive query.
- Returns empty results if `memory_id` is not in the vector index (e.g., server was restarted before the memory was embedded, or the memory was forgotten).

---

## Graph Tools

### `add_event`

Add a node or relationship event to the knowledge graph.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `aggregate_id` | `str` | required | Format: `{type}:{project}:{id}` — e.g. `"node:myproject:task_001"`. Type must be `node`, `sprint`, or `team`. No whitespace in any segment. |
| `event_type` | `str` | required | One of the values below. |
| `payload` | `dict` | required | Non-empty dict with event-specific fields. |

**Valid event types**

| Event type | Effect |
|------------|--------|
| `node_created` | Inserts event, embeds node in hyperbolic space. Payload fields used: `priority` (int), `tags` (list), `author` (str), `complexity` (int). |
| `node_updated` | Same as `node_created` — re-embeds with updated features. |
| `node_deleted` | Inserts tombstone, removes node from similarity search. Vector kept in DB for causal validity. |
| `node_restored` | Removes tombstone, restores node to similarity search with dimension-aligned vector. |
| `relation_added` | Inserts event. Payload should include `source` and `target` aggregate IDs. Used by `analyze_structure`. |
| `relation_removed` | Inserts event. Payload should include `source` and `target`. Removes edge from active graph in `analyze_structure`. |
| `relation_updated` | Inserts event. No structural side effects beyond the event record. |
| `snapshot_created` | Inserts event. No side effects — documentation event. |
| `embedding_recomputed` | Inserts event. No side effects — documentation event. |
| `dimension_changed` | Inserts event. No side effects — documentation event. |

**Returns**

```
"019526b7-f4a2-7000-a1b2-c3d4e5f60001"
```

The `event_id` (UUIDv7 string) for the created event.

**Notes**
- `aggregate_id` is validated against the pattern `^(node|sprint|team):[^\s:]+:[^\s:]+\Z`. Whitespace and extra colons are rejected.
- `payload` must be a non-empty dict — `{}` raises a validation error.
- Embedding dimension may grow after `node_created` if the total node count crosses an adaptive threshold. See `ARCHITECTURE.md` for details.

---

### `query_similar`

Find the k most structurally similar nodes via hyperbolic distance.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `node_id` | `str` | required | The `aggregate_id` of the query node. Must have been embedded (i.e., appeared in a `node_created` or `node_updated` event). |
| `k` | `int` | `5` | Number of neighbors to return. Clamped to [1, 50]. |

**Returns**

```json
[
  { "node_id": "node:myproject:task_002", "distance": 0.0218 },
  { "node_id": "node:myproject:task_005", "distance": 0.0441 }
]
```

Sorted by distance ascending (closest first). Returns empty list if `node_id` has no embedding.

**Notes**
- Tombstoned nodes are automatically excluded.
- Similarity is based on payload features (priority, tag count, author hash, complexity) — not content.
- Linear scan over in-memory node dict: O(N) per call.

---

### `add_constraint`

Register a constraint for the dependency solver.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `node_id` | `str` | required | Aggregate ID of the constrained task node. |
| `constraint_type` | `str` | required | One of: `dependency`, `uniqueness`, `temporal`, `capacity`. |
| `depends_on` | `list[str]` | `[]` | List of aggregate IDs this node depends on. Used only when `constraint_type = "dependency"`. |
| `priority` | `int` | `1` | Ordering priority. Lower value = higher priority (1 is highest). |

**Returns**

```json
{ "constraint_id": "019526b9-...", "enforced": true }
```

If `constraint_type` is not `"dependency"`:
```json
{
  "constraint_id": "019526b9-...",
  "enforced": false,
  "warning": "constraint_type='temporal' is stored but NOT enforced. Only 'dependency' constraints affect suggest_next_tasks() output."
}
```

**Notes**
- Only `dependency` constraints affect `suggest_next_tasks()` output. Other types are stored for future implementation but have no current effect.
- Multiple constraints can be added to the same `node_id`. Their `depends_on` lists are merged and the minimum (highest) priority is used.

---

## Analysis Tools

### `analyze_causal`

Estimate average treatment effect (ATE) via greedy propensity-score matching.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `treatment_filter` | `dict` | required | Dict of `{field: value}` to select treatment nodes. Nodes matching all fields are treatment; all others are control. |
| `outcome_metric` | `str` | required | Payload field used as the outcome variable (numeric). |
| `confounder_keys` | `list[str]` | `null` | Ordered list of payload fields to use as the propensity score variable. If `null`, auto-detected from: `size`, `complexity`, `priority`, `effort`, `weight` (first found). |

**Returns (success)**

```json
{
  "result_id": "019526ba-...",
  "ate": -2.341,
  "n": 18,
  "status": "observational",
  "confounder_used": "complexity",
  "interpretation": "Treatment effect: -2.341 on velocity (observational, 18 matched pairs, confounder: complexity)"
}
```

`status` values:
- `"hypothesis"` — fewer than 10 matched pairs; unreliable
- `"observational"` — 10–29 matched pairs
- `"counterfactual_validated"` — 30+ matched pairs

**Returns (error)**

```json
{
  "error": "No usable confounder field found in node payloads. Tried: ['size', 'complexity', ...]",
  "status": "hypothesis",
  "treatment_n": 5,
  "control_n": 12
}
```

Also returned as error if fewer than 3 nodes on either side, or no matches found within the 0.5 SD caliper.

**Notes**
- Error results are **not** stored in `causal_results`. Only successful analyses are persisted.
- Always check `confounder_used` in the response to confirm which field was actually used.
- Tombstoned nodes are excluded from both treatment and control groups.

---

### `suggest_next_tasks`

Return optimal task execution order for a project.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `project_id` | `str` | `"default"` | Filter tasks by this project ID. Matches against `payload.project_id` on `node_created` events. |

**Returns**

```json
{
  "suggested_order": ["Define API contract", "node:myproject:task_003", "node:myproject:task_001"],
  "rationale": "Ordered by dependency constraints then priority (lower = sooner)",
  "total_tasks": 5,
  "ready_now": 2
}
```

Tasks with `_cycle_warning` in their data are appended at the end of `suggested_order` with a warning note — they cannot be automatically ordered due to circular dependencies.

**Notes**
- Tasks must have `payload.project_id` set when added via `add_event`. Tasks without this field will not appear.
- Uses node `title` field if present, otherwise falls back to `aggregate_id`.
- Only the first 5 entries are returned in `suggested_order`. Use `total_tasks` to understand the full scope.
- Circular dependencies are detected and reported, not silently dropped.

---

### `analyze_structure`

Graph connectivity analysis: find disconnected components and bottleneck nodes.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `project_id` | `str` | `"default"` | Project to analyze. Scoped to nodes with matching `payload.project_id`. |

**Returns**

```json
{
  "total_nodes": 12,
  "connected_components": 3,
  "bottlenecks": ["node:myproject:task_004"],
  "isolated_nodes": ["node:myproject:task_011"],
  "method": "degree_heuristic (not full TDA/Mapper)",
  "recommendation": "Found 3 disconnected groups. Consider connecting via: ['node:myproject:task_004']"
}
```

**Notes**
- Relations are built from `relation_added` events scoped to the project's node set. `relation_removed` events are subtracted from the active edge set.
- Cross-project edges (where one endpoint is outside the project) are excluded.
- Bottleneck heuristic: nodes with degree `> 0` and `< 30%` of the average degree.
- This is iterative DFS, not the full TDA/Mapper (gudhi Rips complex + persistence diagrams). The `method` field confirms this.
- Tombstoned nodes are excluded.

---

## Resource

### `chronos://stats`

Live system statistics as a plaintext string.

**Returns** (plaintext)

```
Memories (active):    47
Memories (forgotten): 3
Memory vectors:       47
TF-IDF indexed:       47
Events:               312
Active nodes:         28
Tombstoned nodes:     2
Causal analyses:      5
Constraints:          14
Embedding dim:        32 (target: 32)
Schema version:       3.1
```

---

## Error Handling

All tools follow consistent error conventions:

- **Validation errors** (`aggregate_id` format, empty content, unknown `event_type`) — returned as MCP error responses with descriptive messages.
- **Not found / already deleted** — returned as structured dicts with a `status` field (`"not_found"`, `"already_forgotten"`, `"error"`).
- **Insufficient data** — structured dict with `"error"` key containing a human-readable explanation (e.g., `analyze_causal` with too few samples).
- **Invalid timestamp** — `query_at` returns `{"error": "Invalid ISO 8601 timestamp: ..."}` with empty results rather than silently misranking.
- **Unknown `memory_id`** for vector ops — returns empty results, not an error.

---

## Aggregate ID Format

```
{type}:{project}:{id}

Examples:
  node:auth-service:task_001
  node:sprint-12:auth_refactor
  sprint:q1-2026:sprint_003
  team:backend:alice
```

Rules:
- Type must be `node`, `sprint`, or `team`
- No whitespace in any segment (spaces, tabs, newlines are rejected)
- No colons within a segment
- Validated via regex `^(node|sprint|team):[^\s:]+:[^\s:]+\Z`
