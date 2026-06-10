# tests/test_beliefs.py
# FSRS math properties, confidence clamping, review cycle, feedback stats.

from chronos.beliefs import (
    CONFIDENCE_MAX,
    CONFIDENCE_MIN,
    STABILITY_GROWTH,
    BeliefEngine,
)
from chronos.memory import MemoryStore


def test_retention_curve_properties():
    eng = BeliefEngine()
    # Fresh memory: full retention
    assert eng.compute_retention(1.0, 0.0) == 1.0
    # Monotonically decreasing in elapsed days
    r = [eng.compute_retention(1.0, d) for d in (0, 1, 5, 30, 365)]
    assert all(a > b for a, b in zip(r, r[1:], strict=False))
    # Higher stability retains longer
    assert eng.compute_retention(10.0, 30) > eng.compute_retention(1.0, 30)
    # Degenerate stability floors out instead of dividing by zero
    assert eng.compute_retention(0.0, 10) == CONFIDENCE_MIN


def test_days_since_handles_garbage():
    assert BeliefEngine.days_since(None) == 0.0
    assert BeliefEngine.days_since("not a date") == 0.0
    assert BeliefEngine.days_since("") == 0.0


def test_confidence_set_boost_weaken_and_clamps():
    store, eng = MemoryStore(), BeliefEngine()
    m = store.remember("fact to score")

    out = eng.set_confidence(m["id"], 5.0, "way too high")
    assert out["new_confidence"] == CONFIDENCE_MAX  # clamped

    out = eng.set_confidence(m["id"], -3.0, "way too low")
    assert out["new_confidence"] == CONFIDENCE_MIN  # clamped

    eng.set_confidence(m["id"], 0.5, "reset")
    boosted = eng.boost_confidence(m["id"], "confirmed by test")
    assert abs(boosted["new_confidence"] - 0.6) < 1e-9
    weakened = eng.weaken_confidence(m["id"], "refuted by test")
    assert abs(weakened["new_confidence"] - 0.45) < 1e-9  # asymmetric delta

    assert "error" in eng.set_confidence("missing-id", 0.5, "x")


def test_confidence_audit_trail_written():
    store, eng = MemoryStore(), BeliefEngine()
    m = store.remember("audited fact")
    eng.set_confidence(m["id"], 0.9, "integration evidence")
    from chronos.db import get_db
    with get_db() as db:
        rows = db.execute(
            "SELECT old_confidence, new_confidence, reason FROM belief_updates "
            "WHERE memory_id = ?", (m["id"],),
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][2] == "integration evidence"


def test_review_cycle_updates_fsrs_state():
    store, eng = MemoryStore(), BeliefEngine()
    m = store.remember("reviewable fact")

    good = eng.record_review(m["id"], "good")
    assert abs(good["new_stability"] - 1.0 * STABILITY_GROWTH) < 1e-9
    assert good["review_count"] == 1

    easy = eng.record_review(m["id"], "easy")
    assert easy["new_stability"] > good["new_stability"]
    assert easy["new_difficulty"] < good["new_difficulty"]

    assert "error" in eng.record_review(m["id"], "impossible")
    assert "error" in eng.record_review("missing-id", "good")


def test_feedback_logging_drives_review_and_stats():
    store, eng = MemoryStore(), BeliefEngine()
    m = store.remember("fact used in answer")

    eng.log_feedback("some query", m["id"], used=True)
    state = eng.get_confidence(m["id"])
    assert state["review_count"] == 1  # used=True triggers an FSRS review

    eng.log_feedback("other query", m["id"], used=False)
    state = eng.get_confidence(m["id"])
    assert state["review_count"] == 1  # used=False must NOT

    stats = eng.get_feedback_stats(days=30)
    assert stats["total_searches"] == 2
    assert stats["results_used"] == 1
    assert stats["hit_rate"] == 0.5
    assert stats["recommendation"] is None  # < 20 samples
