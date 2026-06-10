# chronos/beliefs.py
# Responsibility: Confidence scoring, FSRS-inspired decay math, Bayesian belief updates.
#
# Inspired by ai-iq's beliefs + FSRS systems, adapted for Chronos's zero-dependency,
# single-threaded architecture. All math is pure Python + numpy (existing dep).
#
# Key concepts:
#   confidence  — How strongly we trust this memory (0.01–0.99). Updated by evidence.
#   stability   — How long until this memory fades (FSRS). Higher = longer retention.
#   difficulty  — How hard this memory is to retain (FSRS). Higher = faster decay.
#   retention   — Predicted probability of still being useful: (1 + days/(9*S))^(-1)
#
# References:
#   FSRS-6: https://github.com/open-spaced-repetition/fsrs4anki/wiki/Algorithm
#   Austin 2011 caliper: 0.5 SD (used by CausalAnalyzer, consistency)

import math
from datetime import datetime
from typing import Optional

from chronos.db import get_db
from chronos.uuid7 import uuid7

# --- Constants ---

# Confidence bounds — never allow 0.0 or 1.0 (Bayesian: leave room for updating)
CONFIDENCE_MIN = 0.01
CONFIDENCE_MAX = 0.99
CONFIDENCE_DEFAULT = 0.5

# FSRS parameters
STABILITY_DEFAULT = 1.0     # days — new memory fades in ~1 day without reinforcement
DIFFICULTY_DEFAULT = 0.5     # 0.0=trivial, 1.0=very hard
DIFFICULTY_MIN = 0.01
DIFFICULTY_MAX = 0.99

# Bayesian update magnitudes (symmetric confirmation/refutation)
CONFIRM_DELTA = 0.10         # Confirmed evidence boosts confidence by this much
REFUTE_DELTA = 0.15          # Refuted evidence weakens by slightly more (asymmetric caution)

# FSRS stability multiplier on successful review
STABILITY_GROWTH = 2.5       # Reviewed memory stability multiplies by this
# Difficulty adjustment rate on review
DIFFICULTY_ADJ_RATE = 0.1


def _clamp_confidence(val: float) -> float:
    """Clamp to valid confidence range."""
    return max(CONFIDENCE_MIN, min(CONFIDENCE_MAX, val))


def _clamp_difficulty(val: float) -> float:
    """Clamp to valid difficulty range."""
    return max(DIFFICULTY_MIN, min(DIFFICULTY_MAX, val))


class BeliefEngine:
    """
    Stateless engine for confidence scoring and FSRS decay calculations.

    All state lives in the database — this class provides the math and
    DB operations. Constructed once at startup, injected into tools via closure.
    """

    # ------------------------------------------------------------------
    # FSRS Retention
    # ------------------------------------------------------------------

    @staticmethod
    def compute_retention(stability: float, days_elapsed: float) -> float:
        """
        FSRS forgetting curve: retention = (1 + days / (9 * stability))^(-1)

        Returns a value in (0, 1] representing the probability that this
        memory is still useful/accurate. Drops faster for low-stability
        memories and slow for high-stability ones.

        When stability <= 0 (shouldn't happen), returns 0.01 as a safety floor.
        """
        if stability <= 0:
            return CONFIDENCE_MIN
        return math.pow(1.0 + days_elapsed / (9.0 * stability), -1.0)

    @staticmethod
    def days_since(iso_timestamp: Optional[str]) -> float:
        """Days elapsed since the given ISO timestamp. Returns 0.0 on parse failure."""
        if not iso_timestamp:
            return 0.0
        try:
            then = datetime.fromisoformat(iso_timestamp)
            now = datetime.now()
            return max(0.0, (now - then.replace(tzinfo=None)).total_seconds() / 86400)
        except (ValueError, TypeError):
            return 0.0

    # ------------------------------------------------------------------
    # Confidence operations
    # ------------------------------------------------------------------

    def get_confidence(self, memory_id: str) -> Optional[dict]:
        """
        Fetch current confidence + FSRS state for a memory.
        Returns None if memory not found.
        """
        with get_db() as db:
            row = db.execute(
                """SELECT confidence, stability, difficulty,
                          last_reviewed, review_count, content, forgotten
                   FROM memories WHERE id = ?""",
                (memory_id,),
            ).fetchone()
        if not row:
            return None

        confidence = row[0] if row[0] is not None else CONFIDENCE_DEFAULT
        stability = row[1] if row[1] is not None else STABILITY_DEFAULT
        difficulty = row[2] if row[2] is not None else DIFFICULTY_DEFAULT
        last_reviewed = row[3]
        review_count = row[4] if row[4] is not None else 0

        days = self.days_since(last_reviewed or None)
        retention = self.compute_retention(stability, days)

        return {
            "memory_id": memory_id,
            "confidence": round(confidence, 4),
            "stability": round(stability, 4),
            "difficulty": round(difficulty, 4),
            "retention": round(retention, 4),
            "days_since_review": round(days, 2),
            "review_count": review_count,
            "forgotten": bool(row[6]),
        }

    def set_confidence(
        self, memory_id: str, confidence: float, reason: str
    ) -> dict:
        """
        Set confidence directly with audit logging.
        Returns the new state or error dict.
        """
        confidence = _clamp_confidence(confidence)

        with get_db() as db:
            row = db.execute(
                "SELECT confidence, forgotten FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
            if not row:
                return {"error": f"Memory '{memory_id}' not found"}
            if row[1]:
                return {"error": f"Memory '{memory_id}' is forgotten"}

            old_conf = row[0] if row[0] is not None else CONFIDENCE_DEFAULT

            # Update memory
            db.execute(
                "UPDATE memories SET confidence = ?, updated_at = ? WHERE id = ?",
                (confidence, datetime.now().isoformat(), memory_id),
            )
            # Audit log
            db.execute(
                "INSERT INTO belief_updates VALUES (?, ?, ?, ?, ?, ?)",
                (uuid7(), memory_id, old_conf, confidence,
                 reason, datetime.now().isoformat()),
            )
            db.commit()

        return {
            "memory_id": memory_id,
            "old_confidence": round(old_conf, 4),
            "new_confidence": round(confidence, 4),
            "reason": reason,
        }

    def boost_confidence(self, memory_id: str, reason: str = "confirmed") -> dict:
        """Increase confidence by CONFIRM_DELTA. Used when evidence supports a memory."""
        state = self.get_confidence(memory_id)
        if state is None:
            return {"error": f"Memory '{memory_id}' not found"}
        new_conf = _clamp_confidence(state["confidence"] + CONFIRM_DELTA)
        return self.set_confidence(memory_id, new_conf, reason)

    def weaken_confidence(self, memory_id: str, reason: str = "refuted") -> dict:
        """Decrease confidence by REFUTE_DELTA. Used when evidence contradicts a memory."""
        state = self.get_confidence(memory_id)
        if state is None:
            return {"error": f"Memory '{memory_id}' not found"}
        new_conf = _clamp_confidence(state["confidence"] - REFUTE_DELTA)
        return self.set_confidence(memory_id, new_conf, reason)

    # ------------------------------------------------------------------
    # FSRS review cycle
    # ------------------------------------------------------------------

    def record_review(self, memory_id: str, quality: str = "good") -> dict:
        """
        Record that a memory was reviewed (accessed and found useful).
        Updates FSRS stability + difficulty based on review quality.

        quality: 'good' (default), 'easy', or 'hard'
          - 'easy': stability grows faster, difficulty decreases
          - 'good': standard stability growth
          - 'hard': stability grows less, difficulty increases
        """
        if quality not in ("easy", "good", "hard"):
            return {"error": f"Invalid quality '{quality}' — use easy/good/hard"}

        with get_db() as db:
            row = db.execute(
                """SELECT stability, difficulty, review_count, forgotten
                   FROM memories WHERE id = ?""",
                (memory_id,),
            ).fetchone()
            if not row:
                return {"error": f"Memory '{memory_id}' not found"}
            if row[3]:
                return {"error": f"Memory '{memory_id}' is forgotten"}

            stability = row[0] if row[0] is not None else STABILITY_DEFAULT
            difficulty = row[1] if row[1] is not None else DIFFICULTY_DEFAULT
            review_count = row[2] if row[2] is not None else 0

            # FSRS stability update — grows on each successful review
            growth = {
                "easy": STABILITY_GROWTH * 1.3,
                "good": STABILITY_GROWTH,
                "hard": STABILITY_GROWTH * 0.6,
            }[quality]
            new_stability = stability * growth

            # Difficulty adjustment — easy makes it easier, hard makes it harder
            diff_adj = {
                "easy": -DIFFICULTY_ADJ_RATE,
                "good": 0.0,
                "hard": DIFFICULTY_ADJ_RATE,
            }[quality]
            new_difficulty = _clamp_difficulty(difficulty + diff_adj)

            now = datetime.now().isoformat()
            db.execute(
                """UPDATE memories SET stability = ?, difficulty = ?,
                   last_reviewed = ?, review_count = ?, updated_at = ?
                   WHERE id = ?""",
                (new_stability, new_difficulty, now, review_count + 1, now, memory_id),
            )
            db.commit()

        return {
            "memory_id": memory_id,
            "quality": quality,
            "old_stability": round(stability, 4),
            "new_stability": round(new_stability, 4),
            "old_difficulty": round(difficulty, 4),
            "new_difficulty": round(new_difficulty, 4),
            "review_count": review_count + 1,
        }

    # ------------------------------------------------------------------
    # Search feedback logging
    # ------------------------------------------------------------------

    def log_feedback(self, query: str, memory_id: str, used: bool) -> dict:
        """
        Log whether a recall result was actually used.
        Feeds the meta-learning loop for search weight tuning.
        """
        fb_id = uuid7()
        now = datetime.now().isoformat()
        with get_db() as db:
            db.execute(
                "INSERT INTO search_feedback VALUES (?, ?, ?, ?, ?)",
                (fb_id, query, memory_id, 1 if used else 0, now),
            )
            db.commit()

        # If the memory was used, also record it as a review (FSRS integration)
        if used:
            self.record_review(memory_id, quality="good")

        return {"feedback_id": fb_id, "memory_id": memory_id, "used": used}

    def get_feedback_stats(self, days: int = 30) -> dict:
        """
        Compute search effectiveness stats over a time window.
        Returns hit rate (used/total) and recommendation for recency_weight.
        """
        with get_db() as db:
            # Simple approach: count total and used in recent window
            rows = db.execute(
                """SELECT COUNT(*) as total,
                          SUM(CASE WHEN used = 1 THEN 1 ELSE 0 END) as used_count
                   FROM search_feedback
                   WHERE recalled_at >= datetime('now', ?)""",
                (f"-{days} days",),
            ).fetchone()

        total = rows[0] if rows[0] else 0
        used_count = rows[1] if rows[1] else 0
        hit_rate = used_count / total if total > 0 else 0.0

        # Meta-learning recommendation: if hit rate is low, suggest
        # lowering recency_weight (content relevance matters more than freshness)
        recommendation = None
        if total >= 20:
            if hit_rate < 0.3:
                recommendation = "Low hit rate — consider lowering recency_weight to 0.1"
            elif hit_rate > 0.7:
                recommendation = "High hit rate — current search settings are effective"

        return {
            "total_searches": total,
            "results_used": used_count,
            "hit_rate": round(hit_rate, 4),
            "window_days": days,
            "recommendation": recommendation,
        }
