# chronos/consolidation.py
# Responsibility: Dream consolidation — autonomous memory maintenance.
#
# Four-phase model inspired by biological memory consolidation during REM sleep,
# adapted for Chronos's on-demand, single-threaded architecture. Called explicitly
# via the consolidate_memories MCP tool — not a background process.
#
# Phases (always executed in order):
#   1. Orient      — assess current memory health (counts, confidence, retention)
#   2. Gather      — find near-duplicates via pairwise cosine similarity
#   3. Consolidate — merge duplicates, decay stale confidence, flag low retention
#   4. Prune       — auto-forget memories with critically low confidence + retention
#
# Phase design rationale:
#   Orient first gives the caller a baseline to compare against post-consolidation.
#   Gather is separated from Consolidate so dry-run mode (auto_merge=False) can
#   report findings without acting. Prune is the most aggressive action and runs
#   last — only after all softer interventions have been applied.
#
# Module split: the read-only scans live in consolidation_scan.py; this
# module owns the engine and every database mutation.
#
# v4.0: soft-deletes no longer require manual index cleanup — the FTS
# triggers remove forgotten rows in the same transaction (see db.py).

from datetime import datetime
from typing import List, Optional

from chronos.beliefs import BeliefEngine, CONFIDENCE_DEFAULT, CONFIDENCE_MIN
from chronos.consolidation_config import (
    STALE_DAYS_THRESHOLD,
    STALE_DECAY_DELTA,
)
from chronos.consolidation_scan import (
    find_prune_candidates,
    flag_low_retention,
    gather_duplicates,
    orient,
)
from chronos.db import get_db


class ConsolidationEngine:
    """
    Runs four-phase memory consolidation: Orient, Gather, Consolidate, Prune.

    Requires a BeliefEngine (for FSRS math). Constructed once at startup,
    invoked on-demand via the consolidate_memories tool.
    """

    def __init__(self, belief_engine: BeliefEngine) -> None:
        self.beliefs = belief_engine

    # ------------------------------------------------------------------
    # Phase 3: Consolidate — merge, decay, flag
    # ------------------------------------------------------------------

    def _merge_pair(self, keep_id: str, discard_id: str, reason: str) -> dict:
        """
        Merge two memories: keep the higher-confidence one, forget the other.
        The survivor gets a small confidence boost from absorbing corroboration.
        The FTS trigger removes the discarded memory from search transactionally.
        """
        with get_db() as db:
            keep_row = db.execute(
                "SELECT confidence FROM memories WHERE id = ?", (keep_id,)
            ).fetchone()
            discard_row = db.execute(
                "SELECT confidence FROM memories WHERE id = ?", (discard_id,)
            ).fetchone()

        if not keep_row or not discard_row:
            return {"error": "One or both memories not found"}

        keep_conf = keep_row[0] if keep_row[0] is not None else CONFIDENCE_DEFAULT
        disc_conf = discard_row[0] if discard_row[0] is not None else CONFIDENCE_DEFAULT

        # Boost the survivor (absorbed corroborating memory)
        boost = min(0.05, disc_conf * 0.1)
        new_conf = min(0.99, keep_conf + boost)
        self.beliefs.set_confidence(keep_id, new_conf, reason)

        # Soft-delete the duplicate, recording why
        now = datetime.now().isoformat()
        with get_db() as db:
            db.execute(
                """UPDATE memories SET forgotten = 1, forget_reason = ?,
                   updated_at = ? WHERE id = ?""",
                (reason, now, discard_id),
            )
            db.commit()

        return {
            "kept": keep_id,
            "discarded": discard_id,
            "new_confidence": round(new_conf, 4),
        }

    def _consolidate(
        self,
        duplicates: List[dict],
        auto_merge: bool,
        project: Optional[str] = None,
    ) -> dict:
        """
        Act on gathered data: merge duplicates, decay stale, flag low retention.
        Returns sub-report for the Consolidate phase.
        """
        merged = []
        if auto_merge and duplicates:
            for pair in duplicates:
                state_a = self.beliefs.get_confidence(pair["id_a"])
                state_b = self.beliefs.get_confidence(pair["id_b"])
                conf_a = state_a["confidence"] if state_a else 0
                conf_b = state_b["confidence"] if state_b else 0
                if conf_a >= conf_b:
                    result = self._merge_pair(
                        pair["id_a"], pair["id_b"],
                        f"dream_merge: sim={pair['similarity']}"
                    )
                else:
                    result = self._merge_pair(
                        pair["id_b"], pair["id_a"],
                        f"dream_merge: sim={pair['similarity']}"
                    )
                merged.append(result)

        decayed = self._decay_stale(project)
        flagged = flag_low_retention(self.beliefs, project)

        return {
            "duplicates_merged": len(merged),
            "merge_details": merged,
            "memories_decayed": len(decayed),
            "decay_details": decayed[:10],
            "low_retention_count": len(flagged),
            "needs_review": flagged[:10],
        }

    def _decay_stale(self, project: Optional[str] = None) -> List[dict]:
        """
        Apply confidence decay to memories not reviewed in > STALE_DAYS_THRESHOLD.
        """
        decayed = []
        with get_db() as db:
            if project:
                rows = db.execute(
                    """SELECT id, confidence, last_reviewed, created_at
                       FROM memories WHERE forgotten = 0 AND project = ?""",
                    (project,),
                ).fetchall()
            else:
                rows = db.execute(
                    """SELECT id, confidence, last_reviewed, created_at
                       FROM memories WHERE forgotten = 0"""
                ).fetchall()

        for row in rows:
            mem_id, confidence = row[0], row[1]
            confidence = confidence if confidence is not None else CONFIDENCE_DEFAULT
            review_ts = row[2] or row[3]
            days = self.beliefs.days_since(review_ts)

            if days > STALE_DAYS_THRESHOLD and confidence > CONFIDENCE_MIN:
                new_conf = max(CONFIDENCE_MIN, confidence - STALE_DECAY_DELTA)
                self.beliefs.set_confidence(
                    mem_id, new_conf,
                    f"dream_decay: {days:.0f} days without review"
                )
                decayed.append({
                    "memory_id": mem_id,
                    "old_confidence": round(confidence, 4),
                    "new_confidence": round(new_conf, 4),
                    "days_stale": round(days, 1),
                })

        return decayed

    # ------------------------------------------------------------------
    # Phase 4: Prune — auto-forget critically degraded memories
    # ------------------------------------------------------------------

    def _prune(
        self, project: Optional[str] = None, auto_prune: bool = False,
    ) -> dict:
        """
        Identify and optionally auto-forget memories that are both
        low-confidence AND low-retention (see consolidation_scan.
        find_prune_candidates for the threshold logic).

        auto_prune=False: report candidates only.
        auto_prune=True: soft-delete them with reason 'dream_prune'.
        """
        candidates = find_prune_candidates(self.beliefs, project)

        pruned = 0
        if auto_prune and candidates:
            now = datetime.now().isoformat()
            with get_db() as db:
                for c in candidates:
                    db.execute(
                        """UPDATE memories SET forgotten = 1,
                           forget_reason = 'dream_prune', updated_at = ?
                           WHERE id = ?""",
                        (now, c["memory_id"]),
                    )
                    pruned += 1
                db.commit()

        return {
            "prune_candidates": len(candidates),
            "pruned": pruned,
            "prune_details": candidates[:10],
        }

    # ------------------------------------------------------------------
    # Main entry point — runs all four phases
    # ------------------------------------------------------------------

    def consolidate(
        self,
        project: Optional[str] = None,
        auto_merge: bool = False,
        auto_prune: bool = False,
    ) -> dict:
        """
        Run a full four-phase consolidation pass.

        Phase 1 — Orient:      snapshot memory health metrics
        Phase 2 — Gather:      find near-duplicate pairs
        Phase 3 — Consolidate: merge, decay, flag
        Phase 4 — Prune:       auto-forget critically degraded memories

        project:    optional — scope to a single project. None = all.
        auto_merge: if True, merge detected duplicates. False = dry run.
        auto_prune: if True, auto-forget low-confidence + low-retention memories.
                    False = report candidates only.

        Returns full report with sub-reports from each phase.
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "project": project or "all",
            "phases_run": ["orient", "gather", "consolidate", "prune"],
        }

        report["orient"] = orient(self.beliefs, project)

        duplicates = gather_duplicates(project)
        report["gather"] = {
            "duplicates_found": len(duplicates),
            "duplicate_pairs": duplicates,
        }

        report["consolidate"] = self._consolidate(duplicates, auto_merge, project)
        report["prune"] = self._prune(project, auto_prune)

        return report
