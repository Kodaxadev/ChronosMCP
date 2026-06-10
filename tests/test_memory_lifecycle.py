# tests/test_memory_lifecycle.py
# Full lifecycle round-trips: remember → recall → get → update → forget →
# restore → purge. Includes the v3.3 audit regressions: no silent truncation,
# opt-in compression, persisted forget reason, full-content get_memory.

from chronos.db import get_db
from chronos.memory import MemoryStore


def test_remember_recall_roundtrip_with_source():
    store = MemoryStore()
    m = store.remember(
        "JWT tokens expire after 24 hours.", project="auth",
        tags=["jwt"], source="claude",
    )
    out = store.recall("JWT expiry", project="auth")
    assert out["count"] == 1
    top = out["results"][0]
    assert top["id"] == m["id"]
    assert top["source"] == "claude"
    assert top["confidence"] == 0.5  # default


def test_recall_never_truncates_without_budget():
    # v3.3 REGRESSION: anything over ~150 tokens was silently trimmed.
    store = MemoryStore()
    long_content = "checksum " + " ".join(f"word{i}" for i in range(600))
    store.remember(long_content)
    out = store.recall("checksum")
    assert out["results"][0]["content"] == long_content
    assert "trimmed" not in out["results"][0]
    assert "compression_applied" not in out


def test_recall_compresses_only_when_budget_set():
    store = MemoryStore()
    long_content = "checksum " + " ".join(f"word{i}" for i in range(600))
    store.remember(long_content)
    out = store.recall("checksum", token_budget=100)
    assert "compression_applied" in out
    assert out["total_tokens"] <= 100 + 40  # budget + overhead tolerance
    assert out["results"][0]["content"] != long_content


def test_get_memory_returns_full_content_and_metadata():
    store = MemoryStore()
    long_content = "needle " + " ".join(f"word{i}" for i in range(600))
    m = store.remember(long_content, project="p1", tags=["t1", "t2"])
    rec = store.get(m["id"])
    assert rec["content"] == long_content  # full, never truncated
    assert rec["tags"] == ["t1", "t2"]
    assert rec["project"] == "p1"
    assert rec["source"] == "user"
    assert rec["forgotten"] is False
    assert rec["version_count"] == 0

    store.update(m["id"], "needle replaced content")
    rec = store.get(m["id"])
    assert rec["version_count"] == 1
    assert store.get("nonexistent")["status"] == "not_found"


def test_forget_persists_reason_and_restore_reverses():
    store = MemoryStore()
    m = store.remember("the old staging URL is staging.example.com")
    res = store.forget(m["id"], reason="url decommissioned")
    assert res["status"] == "forgotten"

    rec = store.get(m["id"])
    assert rec["forgotten"] is True
    assert rec["forget_reason"] == "url decommissioned"  # v3.3: was discarded
    assert store.recall("staging URL")["count"] == 0

    res = store.restore(m["id"])
    assert res["status"] == "restored"
    rec = store.get(m["id"])
    assert rec["forgotten"] is False and rec["forget_reason"] is None
    assert store.recall("staging URL")["count"] == 1

    assert store.restore(m["id"])["status"] == "not_forgotten"
    assert store.restore("nope")["status"] == "not_found"


def test_purge_removes_all_content_copies():
    store = MemoryStore()
    m = store.remember("SECRET-VALUE-A original")
    store.update(m["id"], "SECRET-VALUE-B revised")   # creates a version row
    res = store.purge(m["id"])
    assert res["status"] == "purged"
    assert res["versions_removed"] == 1

    assert store.get(m["id"])["status"] == "not_found"
    assert store.recall("SECRET-VALUE-B")["count"] == 0
    with get_db() as db:
        for table, col in [
            ("memories", "id"), ("memory_versions", "memory_id"),
            ("memories_fts", "memory_id"), ("belief_updates", "memory_id"),
        ]:
            n = db.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {col} = ?", (m["id"],)
            ).fetchone()[0]
            assert n == 0, f"content survived purge in {table}"
        # content-free audit record exists
        assert db.execute(
            "SELECT COUNT(*) FROM purge_log WHERE memory_id = ?", (m["id"],)
        ).fetchone()[0] == 1

    assert store.purge(m["id"])["status"] == "not_found"


def test_update_forgotten_memory_rejected():
    store = MemoryStore()
    m = store.remember("something")
    store.forget(m["id"])
    try:
        store.update(m["id"], "new content")
        raise AssertionError("update of forgotten memory should raise")
    except ValueError as exc:
        assert "restore_memory" in str(exc)


def test_empty_content_rejected():
    store = MemoryStore()
    for bad in ["", "   "]:
        try:
            store.remember(bad)
            raise AssertionError("empty content should raise")
        except ValueError:
            pass
