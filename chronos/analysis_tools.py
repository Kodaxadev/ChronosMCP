# chronos/analysis_tools.py
# Responsibility: MCP tool registrations for the analytics layer.
# Owns: analyze_causal, suggest_next_tasks, analyze_structure
#
# Separated from tools.py as part of the module split (tools.py exceeded 400 lines).
# Registration: called from tools.register() via register_analysis_tools().

import json
from typing import List

from mcp.server.fastmcp import FastMCP

from chronos.analyzers import CAUSAL_MIN_VALIDATED
from chronos.db import get_db, get_tombstoned_ids
from chronos.uuid7 import uuid7


def register_analysis_tools(mcp: FastMCP, causal, solver, structure) -> None:
    """
    Register analytics MCP tools on the given FastMCP instance.

    causal:    CausalAnalyzer singleton
    solver:    ConstraintSolver singleton
    structure: StructureAnalyzer singleton
    """

    @mcp.tool()
    async def analyze_causal(
        treatment_filter: dict,
        outcome_metric: str,
        confounder_keys: List[str] = None,
    ) -> dict:
        """
        Estimate average treatment effect (ATE) via greedy propensity-score matching.

        treatment_filter: dict of {field: value} to select treatment nodes.
        outcome_metric:   payload field used as the outcome variable.
        confounder_keys:  ordered list of payload fields to use as the propensity
                          score variable, e.g. ['complexity', 'priority'].
                          If omitted, auto-detected from: size, complexity,
                          priority, effort, weight (first found wins).
                          Always check 'confounder_used' in the response to confirm
                          which field was actually used for matching.

        Requires ≥3 nodes on each side of the treatment split.

        Status in response:
          'hypothesis'               — fewer than 10 matched pairs (unreliable)
          'observational'            — 10–29 matched pairs
          'counterfactual_validated' — 30+ matched pairs (spec threshold)
        """
        # Fetch data, close connection, then compute — avoids holding DB lock
        # during the O(n²) greedy matching pass.
        with get_db() as db:
            tombstoned = get_tombstoned_ids(db)
            events = db.execute(
                "SELECT aggregate_id, payload FROM events WHERE event_type = 'node_created'"
            ).fetchall()

        treatment, control = [], []
        for row in events:
            agg_id = row[0]
            if agg_id in tombstoned:
                continue
            data       = json.loads(row[1])
            data["id"] = agg_id
            if all(data.get(k) == v for k, v in treatment_filter.items()):
                treatment.append(data)
            else:
                control.append(data)

        if len(treatment) < 3 or len(control) < 3:
            return {
                "error":       "Insufficient samples for matching",
                "treatment_n": len(treatment),
                "control_n":   len(control),
                "note": (
                    f"Need ≥3 on each side to attempt; "
                    f"≥{CAUSAL_MIN_VALIDATED} for validated status"
                ),
            }

        result = causal.simple_match(treatment, control, outcome_metric, confounder_keys)

        # FIX: Propagate error detail from simple_match instead of silently
        # dropping it. Errors (missing confounder, no matches within caliper)
        # must be surfaced — not stored as real analysis results.
        if "error" in result:
            return {
                "error":       result["error"],
                "status":      result["status"],
                "treatment_n": len(treatment),
                "control_n":   len(control),
            }

        result_id = uuid7()

        with get_db() as db:
            db.execute(
                "INSERT INTO causal_results VALUES (?, ?, ?, ?, ?, ?)",
                (result_id, json.dumps(treatment_filter), outcome_metric,
                 result["ate"], result["n"], result["status"]),
            )
            db.commit()

        return {
            "result_id":       result_id,
            "ate":             result["ate"],
            "n":               result["n"],
            "status":          result["status"],
            "confounder_used": result.get("confounder_used", "unknown"),
            "interpretation": (
                f"Treatment effect: {result['ate']:+.3f} on {outcome_metric} "
                f"({result['status']}, {result['n']} matched pairs, "
                f"confounder: {result.get('confounder_used', 'unknown')})"
            ),
        }

    @mcp.tool()
    async def suggest_next_tasks(project_id: str = "default") -> dict:
        """
        Constraint solver: return the optimal order to work on tasks in a project.

        Tasks are ordered by dependency constraints first, then by priority (lower = higher).
        Only 'dependency' constraint_type is enforced — see add_constraint() for details.

        aggregate_id of tasks must match 'node:{project_id}:*' pattern.
        payload must include 'project_id' field matching the project_id argument.
        """
        # FIX #2: Fetch events and constraints separately to avoid duplicate
        # rows when a node has multiple constraints (LEFT JOIN multiplies rows).
        # Merge all depends_on lists and use the highest (lowest-number) priority.
        with get_db() as db:
            tombstoned = get_tombstoned_ids(db)
            event_rows = db.execute(
                """SELECT aggregate_id, payload FROM events
                   WHERE event_type = 'node_created'
                   AND json_extract(payload, '$.project_id') = ?""",
                (project_id,),
            ).fetchall()

            constraint_rows = db.execute(
                """SELECT node_id, data FROM constraints
                   WHERE constraint_type = 'dependency'"""
            ).fetchall()

        # Aggregate all constraints per node: merge depends_on, keep best priority
        constraint_map = {}
        for c_row in constraint_rows:
            nid = c_row[0]
            cdata = json.loads(c_row[1])
            if nid not in constraint_map:
                constraint_map[nid] = {"depends_on": [], "priority": 99}
            constraint_map[nid]["depends_on"].extend(cdata.get("depends_on", []))
            constraint_map[nid]["priority"] = min(
                constraint_map[nid]["priority"], cdata.get("priority", 99)
            )

        tasks = []
        for row in event_rows:
            agg_id = row[0]
            if agg_id in tombstoned:
                continue
            data       = json.loads(row[1])
            data["id"] = agg_id
            if agg_id in constraint_map:
                data["depends_on"] = constraint_map[agg_id]["depends_on"]
                data.setdefault("priority", constraint_map[agg_id]["priority"])
            tasks.append(data)

        if not tasks:
            return {
                "suggestion": "No tasks found for this project_id",
                "count":      0,
                "hint": (
                    "Set payload.project_id when calling add_event "
                    "with event_type='node_created'"
                ),
            }

        ordered = solver.solve_next_actions(tasks)
        return {
            "suggested_order": [t.get("title", t["id"]) for t in ordered[:5]],
            "rationale":       "Ordered by dependency constraints then priority (lower = sooner)",
            "total_tasks":     len(tasks),
            "ready_now":       len([t for t in tasks if not t.get("depends_on")]),
        }

    @mcp.tool()
    async def analyze_structure(project_id: str = "default") -> dict:
        """
        Graph connectivity analysis: find disconnected components and bottleneck nodes.

        Method: iterative DFS connected-components + degree-based bottleneck heuristic.
        Note: This is NOT the full TDA/Mapper (gudhi Rips complex + persistence diagrams).
        It provides graph-connectivity insight without external dependencies.

        To build edges: emit add_event with event_type='relation_added' and
        payload={'source': '<aggregate_id>', 'target': '<aggregate_id>'}.
        """
        with get_db() as db:
            tombstoned = get_tombstoned_ids(db)

            nodes = []
            for row in db.execute(
                """SELECT aggregate_id, payload FROM events
                   WHERE event_type = 'node_created'
                   AND json_extract(payload, '$.project_id') = ?""",
                (project_id,),
            ).fetchall():
                if row[0] in tombstoned:
                    continue
                data       = json.loads(row[1])
                data["id"] = row[0]
                nodes.append(data)

            # FIX #5: Scope relations to this project's nodes only.
            # Previously ALL relation_added events were fetched globally,
            # causing cross-project edges to bleed into the analysis.
            project_node_ids = {n["id"] for n in nodes}

            active_edges = set()
            for row in db.execute(
                "SELECT payload FROM events WHERE event_type = 'relation_added'"
            ).fetchall():
                data = json.loads(row[0])
                src, tgt = data.get("source"), data.get("target")
                if src and tgt:
                    if (src in project_node_ids and tgt in project_node_ids
                            and src not in tombstoned
                            and tgt not in tombstoned):
                        active_edges.add((src, tgt))

            # FIX #7: Subtract relation_removed edges so deleted relations
            # don't persist in the graph analysis.
            for row in db.execute(
                "SELECT payload FROM events WHERE event_type = 'relation_removed'"
            ).fetchall():
                data = json.loads(row[0])
                src, tgt = data.get("source"), data.get("target")
                if src and tgt:
                    active_edges.discard((src, tgt))
                    active_edges.discard((tgt, src))

        return structure.analyze(nodes, list(active_edges))
