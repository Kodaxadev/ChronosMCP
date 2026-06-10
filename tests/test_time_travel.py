# tests/test_time_travel.py
# Time-travel correctness: version resolution at a timestamp, exclusion of
# memories forgotten before the snapshot, timestamp validation.

import time
from datetime import datetime

from chronos.memory import MemoryStore


def _now():
    # Small sleep so consecutive ISO timestamps are strictly ordered
    time.sleep(0.01)
    return datetime.now().isoformat()


def test_query_at_resolves_historical_content():
    store = MemoryStore()
    m = store.remember("the API rate limit is 100 requests per minute")
    t_after_create = _now()
    store.update(m["id"], "the API rate limit is 500 requests per minute")
    t_after_update = _now()

    # At t_after_create the ORIGINAL content was current
    past = store.query_at("API rate limit", timestamp=t_after_create)
    assert past["count"] == 1
    assert "100 requests" in past["results"][0]["content"]

    # At t_after_update the new content is current
    present = store.query_at("API rate limit", timestamp=t_after_update)
    assert "500 requests" in present["results"][0]["content"]


def test_query_at_excludes_not_yet_created_and_already_forgotten():
    store = MemoryStore()
    t_before_everything = _now()
    a = store.remember("alpha fact about quasars")
    t_after_a = _now()
    store.remember("beta fact about quasars")

    nothing = store.query_at("quasars", timestamp=t_before_everything)
    assert nothing["count"] == 0

    only_a = store.query_at("quasars", timestamp=t_after_a)
    assert {r["id"] for r in only_a["results"]} == {a["id"]}

    # Forget alpha; before the forget it must still be visible,
    # after the forget it must not.
    store.forget(a["id"])
    t_after_forget = _now()
    assert a["id"] in {
        r["id"] for r in store.query_at("quasars", timestamp=t_after_a)["results"]
    }
    assert a["id"] not in {
        r["id"]
        for r in store.query_at("quasars", timestamp=t_after_forget)["results"]
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
    t = _now()
    out = store.query_at("gamma", timestamp=t, project="physics")
    assert out["count"] == 1
    assert out["results"][0]["project"] == "physics"
