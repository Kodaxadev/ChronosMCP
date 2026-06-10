# tests/test_search.py
# Unit tests for chronos/search.py: MATCH query sanitization, BM25 ranking,
# porter stemming, trigger-maintained index consistency, pairwise similarity.

from chronos.db import get_db
from chronos.memory import MemoryStore
from chronos.search import (
    cosine_similarity,
    match_query,
    rank_snapshot,
    search_memories,
    token_counter,
)

# ---------------------------------------------------------------------------
# match_query sanitization
# ---------------------------------------------------------------------------

def test_match_query_quotes_terms():
    assert match_query("jwt token") == '"jwt" OR "token"'


def test_match_query_neutralises_fts_syntax():
    # FTS5 operators and syntax must come out as quoted plain terms
    q = match_query('NEAR(a, 1) AND "phrase" OR col:filter *')
    assert "NEAR" not in q.replace('"near"', "")
    assert ":" not in q
    assert "*" not in q


def test_match_query_empty_and_punctuation():
    assert match_query("") == ""
    assert match_query("!!! ... ???") == ""
    assert match_query("a") == ""  # single chars dropped


def test_match_query_caps_terms():
    text = " ".join(f"word{i}" for i in range(100))
    q = match_query(text, max_terms=10)
    assert q.count(" OR ") == 9


# ---------------------------------------------------------------------------
# Live search + trigger sync
# ---------------------------------------------------------------------------

def _remember(store, content, **kw):
    return store.remember(content, **kw)


def test_search_finds_stemmed_terms():
    store = MemoryStore()
    m = _remember(store, "We are deploying the staging environment nightly.")
    with get_db() as db:
        rows = search_memories(db, "deploy")
    assert [r["id"] for r in rows] == [m["id"]]


def test_search_injection_strings_do_not_crash():
    store = MemoryStore()
    _remember(store, "harmless content about databases")
    hostile = ['"); DROP TABLE memories; --', 'NEAR(', '"unclosed', "a* b:c"]
    with get_db() as db:
        for q in hostile:
            search_memories(db, q)  # must not raise
        # table still intact
        assert db.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1


def test_search_project_filter_and_forgotten_exclusion():
    store = MemoryStore()
    a = _remember(store, "redis cache eviction policy", project="alpha")
    b = _remember(store, "redis cluster sharding notes", project="beta")
    with get_db() as db:
        ids = {r["id"] for r in search_memories(db, "redis", project="alpha")}
    assert ids == {a["id"]}

    store.forget(b["id"], reason="test")
    with get_db() as db:
        ids = {r["id"] for r in search_memories(db, "redis")}
    assert b["id"] not in ids and a["id"] in ids


def test_update_reindexes_through_trigger():
    store = MemoryStore()
    m = _remember(store, "the password rotation schedule is quarterly")
    store.update(m["id"], "the certificate renewal schedule is monthly")
    with get_db() as db:
        old_hit = search_memories(db, "password rotation")
        new_hit = search_memories(db, "certificate renewal")
    assert old_hit == []
    assert [r["id"] for r in new_hit] == [m["id"]]


def test_fts_rows_match_active_memories_invariant():
    store = MemoryStore()
    a = _remember(store, "first note about kubernetes")
    _remember(store, "second note about terraform")
    store.forget(a["id"])
    with get_db() as db:
        n_active = db.execute(
            "SELECT COUNT(*) FROM memories WHERE forgotten = 0"
        ).fetchone()[0]
        n_fts = db.execute("SELECT COUNT(*) FROM memories_fts").fetchone()[0]
    assert n_active == n_fts == 1


# ---------------------------------------------------------------------------
# Snapshot ranking
# ---------------------------------------------------------------------------

def test_rank_snapshot_orders_by_relevance():
    docs = [
        ("d1", "jwt jwt jwt authentication tokens"),
        ("d2", "kubernetes deployment manifests"),
        ("d3", "jwt mentioned once among many other unrelated words here"),
    ]
    ranked = rank_snapshot(docs, "jwt authentication", k=3)
    assert ranked[0][0] == "d1"
    assert {d for d, _ in ranked} <= {"d1", "d3"}


def test_rank_snapshot_empty_inputs():
    assert rank_snapshot([], "query", k=5) == []
    assert rank_snapshot([("d1", "text")], "", k=5) == []


# ---------------------------------------------------------------------------
# Pairwise similarity
# ---------------------------------------------------------------------------

def test_cosine_similarity_bounds():
    a = token_counter("redis cache eviction policy tuning")
    assert abs(cosine_similarity(a, a) - 1.0) < 1e-9
    b = token_counter("completely unrelated gardening tips")
    assert cosine_similarity(a, b) == 0.0


def test_cosine_similarity_ignores_stopwords():
    a = token_counter("the and or but in on at")
    b = token_counter("the and or but in on at")
    assert cosine_similarity(a, b) == 0.0  # nothing left after stopwords
