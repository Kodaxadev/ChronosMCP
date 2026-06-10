# CHRONOS v2.3 Technical Specification - Part 2

Source split from CHRONOS_v2.3_Technical_Specification.md to keep files under 400 lines.

**Collision behavior**: Same payload â†’ return cached response (`DUPLICATE`); different payload â†’ reject `409` (`CONFLICT`)

```python
class IdempotencyChecker:
    async def check_and_store(
        self, key: str, tenant_id: str, payload_hash: str
    ) -> IdempotencyResult:
        full_key = f"idempotency:{tenant_id}:{key}"
        async with self.redis.pipeline() as pipe:
            try:
                await pipe.watch(full_key)
                existing = await pipe.get(full_key)
                if existing:
                    stored_hash, cached_response = json.loads(existing)
                    if stored_hash == payload_hash:
                        return IdempotencyResult.DUPLICATE(cached_response)
                    return IdempotencyResult.CONFLICT()
                pipe.multi()
                pipe.setex(full_key, self.ttl, json.dumps([payload_hash, None]))
                await pipe.execute()
                return IdempotencyResult.PROCEED()
            except redis.WatchError:
                return await self.check_and_store(key, tenant_id, payload_hash)
```

### 3.3 Enrichment Rules

| Field | Source | Logic |
|-------|--------|-------|
| `metadata.ingestion_timestamp` | Gateway | Server receive time |
| `metadata.source_ip` | Gateway | X-Forwarded-For or remote_addr |
| `metadata.geo_region` | GeoIP | MaxMind DB lookup |
| `payload.normalized_title` | NLP | Lowercase, ASCII fold, trim |
| `payload.mentioned_entities` | NER | spaCy entity extraction |
| `payload.sentiment_score` | NLP | VADER sentiment (âˆ’1.0 to 1.0) |

---

## 4. Hyperbolic Embedding Engine

### 4.1 Adaptive Dimensionality

```python
import math

def calculate_dimension(node_count: int) -> int:
    """
    Formula: 4 * log2(N), rounded up.
    Min: 16 (small graphs). Max: 128 (large graphs).
    Override: 32 for N < 50 (high variance protection).
    """
    if node_count < 50:
        return 32
    return min(128, max(16, math.ceil(4 * math.log2(node_count))))
```

### 4.2 Validation Edge Sampling

```python
def sample_validation_edges(
    edges: List[Edge], ratio: float = 0.1
) -> Tuple[List[Edge], List[Edge]]:
    """
    Hold out 10% of edges (20% if |E| < 50).
    Stratified sampling by degree quartile.
    """
    if len(edges) < 50:
        ratio = 0.2
    degrees   = compute_degree_distribution(edges)
    strata    = stratify_by_degree(edges, degrees, n_strata=4)
    validation, training = [], []
    for stratum in strata:
        n_val = max(1, int(len(stratum) * ratio))
        val_samples = random.sample(stratum, n_val)
        validation.extend(val_samples)
        training.extend([e for e in stratum if e not in val_samples])
    return training, validation
```

### 4.3 Embedding Update Cadence

**Incremental (5-min cycle)**: Process new events, update 2-hop neighborhood, check reconstruction loss.  
**Full re-embedding triggers**:
1. Three consecutive reconstruction losses > 0.15 (with 6h cooldown)
2. Single loss > 0.25 (emergency)
3. Node count changes by > 5%
4. Manual admin trigger or nightly maintenance

```python
class EmbeddingManager:
    def __init__(self):
        self.loss_history = []
        self.reembed_cooldown_hours = 6
        self.last_full_reembed = datetime.min

    def check_reembed_needed(self, current_loss: float) -> bool:
        """3-strike rule with emergency override and cooldown."""
        self.loss_history = (self.loss_history + [current_loss])[-5:]

        if current_loss > 0.25:
            return True  # Emergency

        if len(self.loss_history) >= 3 and all(l > 0.15 for l in self.loss_history[-3:]):
            cooldown_elapsed = datetime.now() - self.last_full_reembed
            return cooldown_elapsed > timedelta(hours=self.reembed_cooldown_hours)

        return False

    async def trigger_full_reembed(self):
        self.last_full_reembed = datetime.now()
        # ... proceed with dimension change protocol
```

### 4.4 Dimension Change Protocol

1. Set `dim_change_in_progress = true` in `embedding_versions`
2. Gateway begins serving stale embeddings + `is_stale: true` flag
3. Compute new embeddings in background
4. Atomic swap: update `embedding_versions`, set `is_active = true`
5. Gateway serves fresh embeddings at new version

### 4.5 PoincarÃ© Ball Operations

```python
import numpy as np

class PoincareBall:
    def __init__(self, dim: int, c: float = 1.0):
        self.dim = dim
        self.c = c

    def mobius_add(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x2  = np.sum(x**2)
        y2  = np.sum(y**2)
        xy  = np.sum(x * y)
        num = (1 + 2*self.c*xy + self.c*y2)*x + (1 - self.c*x2)*y
        den = 1 + 2*self.c*xy + self.c**2*x2*y2
        return num / den

    def exponential_map(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-6:
            return x
        sqrt_c = np.sqrt(self.c)
        return self.mobius_add(x, (np.tanh(sqrt_c * v_norm) / (sqrt_c * v_norm)) * v)

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        x = np.clip(x, -0.99, 0.99)
        y = np.clip(y, -0.99, 0.99)
        x2  = np.sum(x**2)
        y2  = np.sum(y**2)
        num = 2 * self.c * np.sum((x - y)**2)
        den = (1 - self.c*x2) * (1 - self.c*y2)
        return np.arccosh(1 + num/den) / np.sqrt(self.c)
```

---

## 5. Causal Analysis Engine

### 5.1 Status State Machine

```
hypothesis
    â†“ (â‰¥10 samples collected)
observational
    â†“ (â‰¥30 matched pairs + balance + overlap)
counterfactual_validated
```

**Validation Criteria**: â‰¥30 matched pairs; all covariate SMDs < 0.1; propensity scores 0.1â€“0.9; effect size stable across 5 bootstrap samples.

### 5.2 Propensity Score Matching

**Step 1 â€” Estimate propensity scores**:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

def estimate_propensity_scores(treatment: list, control: list):
    features = []
    for node in treatment + control:
        feat = [
            np.log(node.author_tenure_days + 1),
            np.sqrt(node.files_touched),
            np.log(node.pr_lines_changed + 1),
            np.sin(2 * np.pi * node.created_at.hour / 24),
            np.cos(2 * np.pi * node.created_at.hour / 24),
            *[1 if node.created_at.weekday() == i else 0 for i in range(7)]
        ]
        features.append(feat)
    X = StandardScaler().fit_transform(np.array(features))
    y = np.array([1]*len(treatment) + [0]*len(control))
    model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X, y)
    return model.predict_proba(X)[:, 1], model
```

**Step 2 â€” KD-tree matching (default for N > 500, greedy fallback below)**:

```python
from sklearn.neighbors import KDTree

def match_samples_kdtree(
    treatment_indices, control_indices, propensity_scores, caliper=0.2
):
    if len(treatment_indices) + len(control_indices) < 500:
        return match_samples_greedy(
            treatment_indices, control_indices, propensity_scores, caliper
        )
    logit   = np.log(propensity_scores / (1 - propensity_scores))
    thresh  = caliper * np.std(logit)
    tree    = KDTree(logit[control_indices].reshape(-1, 1))
    dists, idxs = tree.query(logit[treatment_indices].reshape(-1, 1), k=1)

    matches, used = [], set()
    for t_idx, dist, local_c in zip(treatment_indices, dists, idxs):
        c_idx = control_indices[local_c[0]]
        if c_idx not in used and dist[0] <= thresh:
            matches.append((t_idx, c_idx))
            used.add(c_idx)
    return matches
```

**Step 3 â€” Balance assessment (SMD < 0.1 target)**:

```python
def check_balance(matched_pairs, covariates):
    smds = {}
    for i in range(covariates.shape[1]):
        t_vals = covariates[[p[0] for p in matched_pairs], i]
        c_vals = covariates[[p[1] for p in matched_pairs], i]
        pooled = np.sqrt((np.var(t_vals) + np.var(c_vals)) / 2)
        smds[f"covariate_{i}"] = abs(np.mean(t_vals) - np.mean(c_vals)) / pooled
    return smds
```

---

## 6. Constraint Solver

### 6.1 Constraint Definition Schema

```json
{
  "constraint_id": "unique_task_assignee",
  "constraint_type": { "enum": ["uniqueness", "dependency", "temporal", "capacity"] },
  "scope": "project|team|global",
  "priority": "hard|soft",
  "condition": {
    "field": "attributes.assignee",
    "operator": "unique_per",
    "context": ["project_id", "sprint_id"]
  },
  "violation_weight": 1.0
}
```

### 6.2 CSP Solver

```python
from constraint import Problem, AllDifferentConstraint

class ConstraintSolver:
    def solve_project_constraints(self, project_id: str):
        nodes = self.get_project_nodes(project_id)
        problem = Problem()

        # Add variables: task_id â†’ domain of eligible assignee IDs
        for task in nodes["tasks"]:
            problem.addVariable(task["id"], self.get_eligible_assignees(task))

        # Hard constraint: no overlapping high-priority assignments
        problem.addConstraint(
            AllDifferentConstraint(),
            [t["id"] for t in nodes["tasks"] if t["priority"] == "high"]
        )

        # Soft constraint: skill matching (unary closure â€” receives domain value)
        task_skills_cache     = {t["id"]: self.get_task_skills(t["id"]) for t in nodes["tasks"]}
        assignee_skills_cache = {a: self.get_assignee_skills(a) for a in self.get_all_assignees()}

        for task in nodes["tasks"]:
            problem.addConstraint(
                self._make_skill_match(task["id"], task_skills_cache, assignee_skills_cache),
                [task["id"]]  # Single variable; constraint receives its domain value
            )

        solutions = problem.getSolutions()
        if not solutions:
            return ConstraintResult.UNSATISFIABLE(self.hard_violations)

        best = max(solutions, key=self.score_solution)
        return ConstraintResult.SATISFIED(assignment=best, score=self.score_solution(best))

    @staticmethod
    def _make_skill_match(task_id, task_skills_cache, assignee_skills_cache):
        def skill_match(assignee_id):  # Receives domain value, not variable name
            task_skills     = task_skills_cache[task_id]
            assignee_skills = assignee_skills_cache.get(assignee_id, [])
            if not task_skills:
                return True
            return (
                len(set(task_skills) & set(assignee_skills)) / len(task_skills) >= 0.5
            )
        return skill_match

    def calculate_resolution_confidence(
        self, constraint_satisfaction: float, previous_scores: list
    ) -> float:
        """0.5 * satisfaction + 0.3 * score_gain + 0.2 * stability"""
        score_gain = max(0, constraint_satisfaction - previous_scores[-1]) if previous_scores else 0
        stability  = 1.0
        if len(previous_scores) >= 2:
            v1 = np.array(previous_scores[-2:])
            v2 = np.array(previous_scores[-1:] + [constraint_satisfaction])
            stability = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return 0.5 * constraint_satisfaction + 0.3 * score_gain + 0.2 * stability
```

---

## 7. TDA/Mapper Engine

### 7.1 Mapper Graph Construction

```python
import gudhi
import numpy as np
from sklearn.cluster import DBSCAN

class TDAEngine:
    def construct_mapper_graph(
        self, data: np.ndarray, filter_func, cover,
        clusterer=DBSCAN(eps=0.5, min_samples=5)
    ):
        """
        Mapper nerve complex (Singh et al., 2007).
        filter_func: lens f: R^D â†’ R (e.g., eccentricity, PCA1, density)
        cover: list of overlapping (left, right) intervals
        """
        filter_values = filter_func(data)
        clusters, cluster_id = [], 0

        for i, (left, right) in enumerate(cover):
            mask    = (filter_values >= left) & (filter_values <= right)
            indices = np.where(mask)[0]
            if len(indices) < 2:
                continue
