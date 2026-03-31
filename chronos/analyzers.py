# chronos/analyzers.py
# Responsibility: Three analytical engines — causal, structural, constraint.
#
# CausalAnalyzer   — greedy propensity-score matching (§5.2)
# StructureAnalyzer — connected components + bottleneck detection (§7 local approx.)
# ConstraintSolver  — greedy topological sort for task ordering (§6)

from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# Causal thresholds (§5.1)
# ---------------------------------------------------------------------------

CAUSAL_MIN_OBSERVATIONAL = 10   # minimum matched pairs for observational status
CAUSAL_MIN_VALIDATED     = 30   # minimum matched pairs for counterfactual_validated

# Greedy caliper: maximum normalized propensity distance allowed for a match.
# 0.5 standard deviations is a common rule-of-thumb (Austin 2011).
# Tighter values reduce bias; looser values increase matched pairs.
CAUSAL_CALIPER           = 0.5


class CausalAnalyzer:
    """
    Greedy propensity-score matching (§5.2 greedy fallback path).

    Status returned:
      'hypothesis'               < 10 matched pairs  (unreliable)
      'observational'            >= 10 matched pairs
      'counterfactual_validated' >= 30 matched pairs
    """

    # Default confounder fields tried in order when caller provides none.
    # First field present across any node in the dataset is used.
    _DEFAULT_CONFOUNDERS = ["size", "complexity", "priority", "effort", "weight"]

    def _resolve_confounder_key(
        self,
        all_nodes: list,
        confounder_keys: list,
    ) -> str:
        """
        Pick the first confounder key that exists on at least one node.
        Falls back to the first default if none are found, with a warning
        embedded in the returned key name so callers can detect it.
        """
        candidates = confounder_keys or self._DEFAULT_CONFOUNDERS
        for key in candidates:
            if any(key in n for n in all_nodes):
                return key
        return "__missing__"

    def simple_match(self, treatment_nodes: list, control_nodes: list,
                     outcome_key: str,
                     confounder_keys: list = None) -> dict:
        """
        Match treatment and control nodes on a propensity score derived from
        confounder_keys (caller-specified) or auto-detected from the default
        candidate list. Uses the first matching key found in the data.

        confounder_keys: ordered list of payload field names to use as the
                         propensity variable, e.g. ['complexity', 'priority'].
                         If None, auto-detected from defaults.
        """
        all_nodes     = treatment_nodes + control_nodes
        confounder_key = self._resolve_confounder_key(all_nodes, confounder_keys)

        if confounder_key == "__missing__":
            return {
                "ate":    0,
                "n":      0,
                "status": "hypothesis",
                "error":  (
                    "No usable confounder field found in node payloads. "
                    f"Tried: {confounder_keys or self._DEFAULT_CONFOUNDERS}. "
                    "Add one of these fields to your node payloads, or pass "
                    "confounder_keys=['your_field'] explicitly."
                ),
            }

        # FIX #1: Normalize using POOLED statistics across both groups.
        # Previous code normalized each group independently, which centered
        # both at 0 and allowed spurious matches between groups with wildly
        # different confounder values.
        all_vals = [n.get(confounder_key, 0) for n in all_nodes]
        pooled_mean = np.mean(all_vals)
        pooled_std  = np.std(all_vals)

        def normalize(nodes):
            # When std≈0, all nodes have identical confounder values.
            # Scores all become 0.0 — every pair is within caliper,
            # which is correct: identical confounders means no selection bias.
            if pooled_std < 1e-9:
                return [(n, 0.0) for n in nodes]
            return [(n, (n.get(confounder_key, 0) - pooled_mean) / pooled_std)
                    for n in nodes]

        t_norm = normalize(treatment_nodes)
        c_norm = normalize(control_nodes)

        matches = []
        used    = set()
        for t_node, t_score in t_norm:
            best, best_dist = None, float("inf")
            for i, (c_node, c_score) in enumerate(c_norm):
                if i in used:
                    continue
                dist = abs(t_score - c_score)
                if dist < best_dist and dist < CAUSAL_CALIPER:
                    best, best_dist = i, dist
            if best is not None:
                matches.append((t_node, c_norm[best][0]))
                used.add(best)

        n = len(matches)
        if n == 0:
            return {"ate": 0, "n": 0, "status": "hypothesis",
                    "error": "No valid matches within caliper"}

        t_outcomes = [m[0].get(outcome_key, 0) for m in matches]
        c_outcomes = [m[1].get(outcome_key, 0) for m in matches]
        ate        = np.mean(t_outcomes) - np.mean(c_outcomes)

        if n >= CAUSAL_MIN_VALIDATED:
            status = "counterfactual_validated"
        elif n >= CAUSAL_MIN_OBSERVATIONAL:
            status = "observational"
        else:
            status = "hypothesis"

        return {
            "ate":             round(ate, 3),
            "n":               n,
            "status":          status,
            "confounder_used": confounder_key,
            "treatment_mean":  round(np.mean(t_outcomes), 3),
            "control_mean":    round(np.mean(c_outcomes), 3),
        }


class StructureAnalyzer:
    """
    Graph connectivity analysis (local approximation of §7 TDA/Mapper Engine).

    What this IS:  Connected components (iterative DFS) + degree-based
                   bottleneck heuristic.
    What this is NOT: The full §7 Mapper nerve complex (gudhi.RipsComplex,
                   DBSCAN cover, Betti numbers, persistence diagrams).
                   Those require the `gudhi` package.
    """

    def analyze(self, nodes: List[dict], relations: List[tuple]) -> dict:
        """
        Find disconnected components and low-degree bottleneck nodes.

        Returns: {total_nodes, connected_components, bottlenecks,
                  isolated_nodes, method, recommendation}
        """
        adj = {n["id"]: set() for n in nodes}
        for src, dst in relations:
            if src in adj and dst in adj:
                adj[src].add(dst)
                adj[dst].add(src)

        visited    = set()
        components = []

        # Iterative DFS — avoids Python recursion limit on large graphs
        for node in nodes:
            if node["id"] in visited:
                continue
            component = set()
            stack     = [node["id"]]
            while stack:
                nid = stack.pop()
                if nid in visited:
                    continue
                visited.add(nid)
                component.add(nid)
                for neighbor in adj[nid]:
                    if neighbor not in visited:
                        stack.append(neighbor)
            components.append(component)

        all_degrees = {nid: len(neighbors) for nid, neighbors in adj.items()}
        avg_degree  = np.mean(list(all_degrees.values())) if all_degrees else 0

        bottlenecks = [
            nid for nid, deg in all_degrees.items()
            if 0 < deg < avg_degree * 0.3
        ]
        isolated = [nid for nid, deg in all_degrees.items() if deg == 0]

        return {
            "total_nodes":          len(nodes),
            "connected_components": len(components),
            "bottlenecks":          bottlenecks[:10],
            "isolated_nodes":       isolated,
            "method":               "degree_heuristic (not full TDA/Mapper)",
            "recommendation": (
                f"Found {len(components)} disconnected groups. "
                f"Consider connecting via: {bottlenecks[:3]}"
                if len(components) > 1 else "Well-connected structure."
            ),
        }


class ConstraintSolver:
    """
    Localized constraint solver supporting DEPENDENCY ordering only.

    Scope note: The full §6 spec also handles 'uniqueness', 'temporal', and
    'capacity' constraints via python-constraint backtracking. This local
    implementation covers 'dependency' (topological sort + priority weighting)
    which is the primary use case for personal task management. If you need
    skill-match or assignee uniqueness constraints, see §6.2 in the spec.
    """

    def solve_next_actions(self, tasks: List[dict]) -> List[dict]:
        """
        Return tasks in optimal execution order using greedy topological sort.
        Ties broken by (priority ASC, due_date ASC).
        """
        in_degree  = {t["id"]: 0 for t in tasks}
        dependents = {t["id"]: [] for t in tasks}

        for task in tasks:
            for dep in task.get("depends_on", []):
                if dep in in_degree:
                    in_degree[task["id"]] += 1
                    dependents[dep].append(task["id"])

        ready  = [t for t in tasks if in_degree[t["id"]] == 0]
        result = []

        while ready:
            ready.sort(key=lambda t: (t.get("priority", 99), t.get("due_date", "9999")))
            next_task = ready.pop(0)
            result.append(next_task)

            for dep_id in dependents[next_task["id"]]:
                in_degree[dep_id] -= 1
                if in_degree[dep_id] == 0:
                    dep_task = next((t for t in tasks if t["id"] == dep_id), None)
                    if dep_task:
                        ready.append(dep_task)

        # FIX: Detect tasks trapped in dependency cycles. Without this,
        # cyclic tasks silently vanish from the output with no warning.
        if len(result) < len(tasks):
            cycle_ids = [t["id"] for t in tasks if t["id"] not in
                         {r["id"] for r in result}]
            for cid in cycle_ids:
                result.append({
                    "id": cid,
                    "priority": 99,
                    "_cycle_warning": (
                        f"Task '{cid}' is part of a circular dependency "
                        "and cannot be automatically ordered"
                    ),
                })

        return result
