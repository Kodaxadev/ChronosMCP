# tests/test_time_travel.py
# Time-travel correctness: version resolution at a timestamp, exclusion of
# memories forgotten before the snapshot, timestamp validation.
#
# Timestamps are injected directly via SQL rather than read from the wall
# clock: datetime.now() ticks at ~15.6ms on Windows Python <= 3.12, so
# sleep-based "before/after" tests are flaky there (caught by the CI matrix —
# Windows 3.10/3.12 failed while 3.14 and Linux passed). Synthetic timelines
# are deterministic on every platform.

from datetime import datetime, timedelta

from chronos.db import get_db
from chronos.memory import MemoryStore


def _day(n: int) -> str:
    """A stable ISO timestamp n days in the past."""
    return (datetime.now() - timedelta(days=30 - n)).isoformat()


def _set_memory_times(memory_id: str, created_at: str, updated_at: str):
    with get_db() as db:
        db.execute(
            "UPDATE memories SET created_at = ?, updated_at = ? WHERE id = ?",
            (created_at, updated_at, memory_id),
        )
        db.commit()


def _set_version_window(memory_id: str, valid_from: str, valid_to: str):
    with get_db() as db:
        db.execute(
            "UPDATE memory_versions SET valid_from = ?, valid_to = ? "
            "WHERE memory_id = ?",
            (valid_from, valid_to, memory_id),
        )
        db.commit()


def test_query_at_resolves_historical_content():
    store = MemoryStore()
    m = store.remember("the API rate limit is 100 requests per minute")
    store.update(m["id"], "the API rate limit is 500 requests per minute")

    # Synthetic timeline: created day 0, edited day 10
    _set_memory_times(m["id"], created_at=_day(0), updated_at=_day(10))
    _set_version_window(m["id"], valid_from=_day(0), valid_to=_day(10))

    # Day 5: the ORIGINAL content was current
    past = store.query_at("API rate limit", timestamp=_day(5))
    assert past["count"] == 1
    assert "100 requests" in past["results"][0]["content"]

    # Day 15: the new content is current
    present = store.query_at("API rate limit", timestamp=_day(15))
    assert "500 requests" in present["results"][0]["content"]


def test_query_at_excludes_not_yet_created_and_already_forgotten():
    store = MemoryStore()
    a = store.remember("alpha fact about quasars")
    b = store.remember("beta fact about quasars")
    _set_memory_times(a["id"], created_at=_day(0), updated_at=_day(0))
    _set_memory_times(b["id"], created_at=_day(10), updated_at=_day(10))

    # Before anything existed
    assert store.query_at("quasars", timestamp=_day(-1))["count"] == 0

    # Day 5: only alpha existed
    only_a = store.query_at("quasars", timestamp=_day(5))
    assert {r["id"] for r in only_a["results"]} == {a["id"]}

    # Forget alpha on day 20: visible at day 15, gone at day 25
    store.forget(a["id"])
    _set_memory_times(a["id"], created_at=_day(0), updated_at=_day(20))
    assert a["id"] in {
        r["id"] for r in store.query_at("quasars", timestamp=_day(15))["results"]
    }
    assert a["id"] not in {
        r["id"] for r in store.query_at("quasars", timestamp=_day(25))["results"]
    }


def test_query_at_invalid_timestamp_is_an_error_not_a_crash():
    store = MemoryStore()
    store.remember("anything")
    out = store.query_at("anything", timestamp="not-a-date")
    assert "error" in out
    assert out["count"] == 0


def test_query_at_project_filter():
    store = MemoryStore()
    store.remember("gamma rays note", project="physics")
    store.remember("gamma function note", project="math")
    # Far-future snapshot avoids any clock-boundary ambiguity
    out = store.query_at("gamma", timestamp="2099-01-01T00:00:00", project="physics")
    assert out["count"] == 1
    assert out["results"][0]["project"] == "physics"
