# chronos/consolidation.py
# Responsibility: Dream consolidation â€” autonomous memory maintenance.
#
# Four-phase model inspired by biological memory consolidation during REM sleep,
# adapted for Chronos's on-demand, single-threaded architecture. Called explicitly
# via the consolidate_memories MCP tool â€” not a background process.
#
# Phases (always executed in order):
#   1. Orient     â€” assess current memory health (counts, confidence, retention)
#   2. Gather     â€” find near-duplicates via TF-IDF similarity
#   3. Consolidate â€” merge duplicates, decay stale confidence, flag low retention
#   4. Prune      â€” auto-forget memories with critically low confidence + retention
#
# Phase design rationale:
#   Orient first gives the caller a baseline to compare against post-consolidation.
#   Gather is separated from Consolidate so dry-run mode (auto_merge=False) can
#   report findings without acting. Prune is the most aggressive action and runs
#   last â€” only after all softer interventions have been applied.

import json
from datetime import datetime
from typing import List, Optional

from chronos.beliefs import (
    BeliefEngine,
    CONFIDENCE_MIN,
    CONFIDENCE_DEFAULT,
    STABILITY_DEFAULT,
)
from chronos.consolidation_config import (
    DUPLICATE_THRESHOLD,
    PRUNE_CONFIDENCE_THRESHOLD,
    PRUNE_RETENTION_THRESHOLD,
    RETENTION_WARNING_THRESHOLD,
    STALE_DAYS_THRESHOLD,
    STALE_DECAY_DELTA,
)
from chronos.db import get_db
from chronos.tfidf import TFIDFIndex
from chronos.uuid7 import uuid7




class ConsolidationEngine:
    """
    Runs four-phase memory consolidation: Orient, Gather, Consolidate, Prune.

    Requires a BeliefEngine (for FSRS math) and a TFIDFIndex (for similarity).
    Constructed once at startup, invoked on-demand via the consolidate_memories tool.
    """

    def __init__(self, belief_engine: BeliefEngine, tfidf: TFIDFIndex) -> None:
        self.beliefs = belief_engine
        self.tfidf = tfidf

    # ------------------------------------------------------------------
    # Phase 1: Orient â€” assess current memory health
    # ------------------------------------------------------------------

    def _orient(self, project: Optional[str] = None) -> dict:
        """
        Snapshot current memory health metrics before consolidation begins.
        Returns counts, average confidence, retention distribution, and staleness.
        """
        with get_db() as db:
            if project:
                rows = db.execute(
                    """SELECT id, confidence, stability, last_reviewed, created_at
                       FROM memories WHERE forgotten = 0 AND project = ?""",
                    (project,),
                ).fetchall()
            else:
                rows = db.execute(
                    """SELECT id, confidence, stability, last_reviewed, created_at
                       FROM memories WHERE forgotten = 0"""
                ).fetchall()

        if not rows:
            return {
                "total_active": 0,
                "avg_confidence": 0.0,
                "retention_buckets": {"high": 0, "medium": 0, "low": 0, "critical": 0},
                "stale_count": 0,
            }

        confidences = []
        retention_buckets = {"high": 0, "medium": 0, "low": 0, "critical": 0}
        stale_count = 0

        for row in rows:
            conf = row[1] if row[1] is not None else CONFIDENCE_DEFAULT
            confidences.append(conf)

            stability = row[2] if row[2] is not None else STABILITY_DEFAULT
            review_ts = row[3] or row[4]
            days = self.beliefs.days_since(review_ts)
            retention = self.beliefs.compute_retention(stability, days)

            if retention >= 0.7:
                retention_buckets["high"] += 1
            elif retention >= 0.4:
                retention_buckets["medium"] += 1
            elif retention >= PRUNE_RETENTION_THRESHOLD:
                retention_buckets["low"] += 1
            else:
                retention_buckets["critical"] += 1

            if days > STALE_DAYS_THRESHOLD:
                stale_count += 1

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        return {
            "total_active": len(rows),
            "avg_confidence": round(avg_conf, 4),
            "retention_buckets": retention_buckets,
            "stale_count": stale_count,
        }

    # ------------------------------------------------------------------
    # Phase 2: Gather â€” find near-duplicates
    # ------------------------------------------------------------------

    def _gather_duplicates(
        self, project: Optional[str] = None
    ) -> List[dict]:
        """
        Find pairs of memories with TF-IDF cosine similarity > DUPLICATE_THRESHOLD.
        Returns list of {id_a, id_b, similarity, preview_a, preview_b}.
        """
        with get_db() as db:
            if project:
                rows = db.execute(
                    "SELECT id, content FROM memories "
                    "WHERE forgotten = 0 AND project = ?",
                    (project,),
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT id, content FROM memories WHERE forgotten = 0"
                ).fetchall()

        if len(rows) < 2:
            return []

        # Build a temporary index for pairwise comparison
        temp_index = TFIDFIndex()
        temp_index.load_documents([(r[0], r[1]) for r in rows])

        pairs = []
        seen = set()

        for row in rows:
            mem_id, content = row[0], row[1]
            results = temp_index.query(content, k=5, exclude={mem_id})
            for other_id, score in results:
                pair_key = tuple(sorted([mem_id, other_id]))
                if pair_key in seen:
                    continue
                if score >= DUPLICATE_THRESHOLD:
                    seen.add(pair_key)
                    other_content = temp_index.get_text(other_id)
                    pairs.append({
                        "id_a": mem_id,
                        "id_b": other_id,
                        "similarity": round(score, 4),
                        "preview_a": content[:80],
                        "preview_b": other_content[:80],
                    })

        return pairs

    # ------------------------------------------------------------------
    # Phase 3: Consolidate â€” merge, decay, flag
    # ------------------------------------------------------------------

    def _merge_pair(self, keep_id: str, discard_id: str, reason: str) -> dict:
        """
        Merge two memories: keep the higher-confidence one, forget the other.
        The survivor gets a small confidence boost from absorbing corroboration.
        """
        with get_db() as db:
            keep_row = db.execute(
                "SELECT confidence, content FROM memories WHERE id = ?",
                (keep_id,),
            ).fetchone()
            discard_row = db.execute(
                "SELECT confidence, content FROM memories WHERE id = ?",
                (discard_id,),
            ).fetchone()

        if not keep_row or not discard_row:
            return {"error": "One or both memories not found"}

        keep_conf = keep_row[0] if keep_row[0] is not None else CONFIDENCE_DEFAULT
        disc_conf = discard_row[0] if discard_row[0] is not None else CONFIDENCE_DEFAULT

        # Boost the survivor (absorbed corroborating memory)
        boost = min(0.05, disc_conf * 0.1)
        new_conf = min(0.99, keep_conf + boost)
        self.beliefs.set_confidence(keep_id, new_conf, reason)

        # Soft-delete the duplicate
        now = datetime.now().isoformat()
        with get_db() as db:
            db.execute(
                "UPDATE memories SET forgotten = 1, updated_at = ? WHERE id = ?",
                (now, discard_id),
            )
            db.commit()

        self.tfidf.remove_document(discard_id)

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
        # --- Merge duplicates ---
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

        # --- Decay stale confidence ---
        decayed = self._decay_stale(project)

        # --- Flag low retention ---
        flagged = self._flag_low_retention(project)

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

    def _flag_low_retention(self, project: Optional[str] = None) -> List[dict]:
        """
        Find memories whose FSRS retention has dropped below the warning threshold.
        """
        flagged = []
        with get_db() as db:
            if project:
                rows = db.execute(
                    """SELECT id, stability, last_reviewed, created_at, content
                       FROM memories WHERE forgotten = 0 AND project = ?""",
                    (project,),
                ).fetchall()
            else:
                rows = db.execute(
                    """SELECT id, stability, last_reviewed, created_at, content
                       FROM memories WHERE forgotten = 0"""
                ).fetchall()

        for row in rows:
            mem_id = row[0]
            stability = row[1] if row[1] is not None else STABILITY_DEFAULT
            review_ts = row[2] or row[3]
            days = self.beliefs.days_since(review_ts)
            retention = self.beliefs.compute_retention(stability, days)

            if retention < RETENTION_WARNING_THRESHOLD:
                flagged.append({
                    "memory_id": mem_id,
                    "retention": round(retention, 4),
                    "stability": round(stability, 4),
                    "days_since_review": round(days, 1),
                    "preview": row[4][:80] if row[4] else "",
                })

        return flagged

    # ------------------------------------------------------------------
    # Phase 4: Prune â€” auto-forget critically degraded memories
    # ------------------------------------------------------------------

    def _prune(
        self, project: Optional[str] = None, auto_prune: bool = False,
    ) -> dict:
        """
        Identify and optionally auto-forget memories that are both low-confidence
        AND low-retention. These are memories the system has effectively lost
        faith in â€” both untrusted and nearly forgotten.

        Thresholds: confidence < 0.10 AND retention < 0.15. Both must be true.
        This prevents pruning high-value memories that simply haven't been reviewed.

        auto_prune=False: report candidates only.
        auto_prune=True: soft-delete them with reason 'dream_prune'.
        """
        candidates = []
        with get_db() as db:
            if project:
                rows = db.execute(
                    """SELECT id, confidence, stability, last_reviewed,
                              created_at, content
                       FROM memories WHERE forgotten = 0 AND project = ?""",
                    (project,),
                ).fetchall()
            else:
                rows = db.execute(
                    """SELECT id, confidence, stability, last_reviewed,
                              created_at, content
                       FROM memories WHERE forgotten = 0"""
                ).fetchall()

        for row in rows:
            mem_id = row[0]
            conf = row[1] if row[1] is not None else CONFIDENCE_DEFAULT
            stability = row[2] if row[2] is not None else STABILITY_DEFAULT
            review_ts = row[3] or row[4]
            days = self.beliefs.days_since(review_ts)
            retention = self.beliefs.compute_retention(stability, days)

            if conf < PRUNE_CONFIDENCE_THRESHOLD and retention < PRUNE_RETENTION_THRESHOLD:
                candidates.append({
                    "memory_id": mem_id,
                    "confidence": round(conf, 4),
                    "retention": round(retention, 4),
                    "days_since_review": round(days, 1),
                    "preview": row[5][:80] if row[5] else "",
                })

        pruned = 0
        if auto_prune and candidates:
            now = datetime.now().isoformat()
            with get_db() as db:
                for c in candidates:
                    db.execute(
                        "UPDATE memories SET forgotten = 1, updated_at = ? "
                        "WHERE id = ?",
                        (now, c["memory_id"]),
                    )
                    self.tfidf.remove_document(c["memory_id"])
                    pruned += 1
                db.commit()

        return {
            "prune_candidates": len(candidates),
            "pruned": pruned,
            "prune_details": candidates[:10],
        }

    # ------------------------------------------------------------------
    # Main entry point â€” runs all four phases
    # ------------------------------------------------------------------

    def consolidate(
        self,
        project: Optional[str] = None,
        auto_merge: bool = False,
        auto_prune: bool = False,
    ) -> dict:
        """
        Run a full four-phase consolidation pass.

        Phase 1 â€” Orient:      snapshot memory health metrics
        Phase 2 â€” Gather:      find near-duplicate pairs
        Phase 3 â€” Consolidate: merge, decay, flag
        Phase 4 â€” Prune:       auto-forget critically degraded memories

        project:    optional â€” scope to a single project. None = all.
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

        # Phase 1: Orient
        report["orient"] = self._orient(project)

        # Phase 2: Gather
        duplicates = self._gather_duplicates(project)
        report["gather"] = {
            "duplicates_found": len(duplicates),
            "duplicate_pairs": duplicates if not auto_merge else duplicates,
        }

        # Phase 3: Consolidate
        consolidate_result = self._consolidate(duplicates, auto_merge, project)
        report["consolidate"] = consolidate_result

        # Phase 4: Prune
        report["prune"] = self._prune(project, auto_prune)

        return report
