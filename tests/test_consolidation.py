# tests/test_consolidation.py
# Dream consolidation: duplicate gathering, merge direction, stale decay,
# prune thresholds (both must be met), and the v3.3 audit regression that
# merged/pruned memories must vanish from search.

from datetime import datetime, timedelta

from chronos.beliefs import BeliefEngine
from chronos.consolidation import ConsolidationEngine
from chronos.db import get_db
from chronos.memory import MemoryStore


def _engine():
    return ConsolidationEngine(BeliefEngine())


def _backdate(memory_id: str, days: int):
    """Shift created_at into the past (last_reviewed left NULL)."""
    past = (datetime.now() - timedelta(days=days)).isoformat()
    with get_db() as db:
        db.execute(
            "UPDATE memories SET created_at = ? WHERE id = ?", (past, memory_id)
        )
        db.commit()


def test_gather_finds_duplicates_and_ignores_distinct():
    store, eng = MemoryStore(), _engine()
    store.remember("JWT tokens expire after 24 hours and live in Redis")
    store.remember("JWT tokens expire after 24 hours and live in Redis")
    store.remember("the deployment pipeline uses GitHub Actions runners")

    report = eng.consolidate(auto_merge=False)
    assert report["gather"]["duplicates_found"] == 1
    pair = report["gather"]["duplicate_pairs"][0]
    assert pair["similarity"] >= 0.85


def test_auto_merge_keeps_higher_confidence_and_removes_from_search():
    store, eng = MemoryStore(), _engine()
    beliefs = BeliefEngine()
    keep = store.remember("postgres connection pool is capped at 20 clients")
    drop = store.remember("postgres connection pool is capped at 20 clients")
    beliefs.set_confidence(keep["id"], 0.9, "verified")
    beliefs.set_confidence(drop["id"], 0.3, "unverified")

    report = eng.consolidate(auto_merge=True)
    merges = report["consolidate"]["merge_details"]
    assert len(merges) == 1
    assert merges[0]["kept"] == keep["id"]
    assert merges[0]["discarded"] == drop["id"]
    assert merges[0]["new_confidence"] > 0.9  # survivor absorbed corroboration

    # v3.3 REGRESSION: discarded duplicates must vanish from search
    out = store.recall("postgres connection pool")
    assert {r["id"] for r in out["results"]} == {keep["id"]}
    # and the forget reason is recorded
    rec = store.get(drop["id"])
    assert rec["forgotten"] is True
    assert "dream_merge" in rec["forget_reason"]


def test_decay_reduces_confidence_of_stale_memories():
    store, eng = MemoryStore(), _engine()
    m = store.remember("stale memory that nobody reviewed")
    _backdate(m["id"], days=45)  # > STALE_DAYS_THRESHOLD (30)

    report = eng.consolidate()
    decayed = report["consolidate"]["decay_details"]
    assert any(d["memory_id"] == m["id"] for d in decayed)
    entry = next(d for d in decayed if d["memory_id"] == m["id"])
    assert entry["new_confidence"] < entry["old_confidence"]


def test_prune_requires_both_thresholds():
    store, eng = MemoryStore(), _engine()
    beliefs = BeliefEngine()

    # Low confidence but FRESH (high retention) — must NOT be a candidate
    fresh = store.remember("low confidence but recent")
    beliefs.set_confidence(fresh["id"], 0.05, "doubtful")

    # Low confidence AND stale (low retention) — must be pruned
    doomed = store.remember("low confidence and abandoned")
    beliefs.set_confidence(doomed["id"], 0.05, "doubtful")
    _backdate(doomed["id"], days=90)  # retention (1+90/9)^-1 ≈ 0.09 < 0.15

    report = eng.consolidate(auto_prune=True)
    pruned_ids = {c["memory_id"] for c in report["prune"]["prune_details"]}
    assert doomed["id"] in pruned_ids
    assert fresh["id"] not in pruned_ids

    # pruned memory is gone from search but restorable (soft delete)
    assert store.get(doomed["id"])["forgotten"] is True
    assert store.get(doomed["id"])["forget_reason"] == "dream_prune"
    assert store.recall("abandoned")["count"] == 0


def test_orient_reports_health_buckets():
    store, eng = MemoryStore(), _engine()
    store.remember("healthy fresh memory")
    report = eng.consolidate()
    orient = report["orient"]
    assert orient["total_active"] == 1
    assert orient["retention_buckets"]["high"] == 1
    assert orient["avg_confidence"] == 0.5
