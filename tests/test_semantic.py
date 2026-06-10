# tests/test_semantic.py
# Hybrid semantic search (CHRONOS_SEMANTIC). CI never downloads a model:
# these tests inject a deterministic fake embedder. One opt-in smoke test
# exercises the real fastembed model and skips when the extra isn't installed.

import importlib.util
import re
import zlib

import numpy as np
import pytest

from chronos.db import get_db
from chronos.memory import MemoryStore
from chronos.semantic import SemanticSearch, content_hash

_FASTEMBED = importlib.util.find_spec("fastembed") is not None

# Topic lexicon: words in the same topic embed near each other, so the fake
# behaves like a tiny synonym-aware model ("car" ≈ "automobile").
_TOPICS = {
    "car": 0, "automobile": 0, "vehicle": 0, "engine": 0,
    "repair": 0, "repairs": 0, "maintenance": 0,
    "jwt": 1, "token": 1, "tokens": 1, "auth": 1, "authentication": 1,
    "deploy": 2, "pipeline": 2, "github": 2, "actions": 2,
}


def fake_embed(texts):
    out = []
    for text in texts:
        v = np.zeros(8, dtype=np.float32)
        for w in re.findall(r"[a-z]+", text.lower()):
            if w in _TOPICS:
                v[_TOPICS[w]] += 1.0
            else:
                # stable low-weight noise (zlib.crc32, not hash(): PYTHONHASHSEED)
                v[3 + (zlib.crc32(w.encode()) % 5)] += 0.05
        out.append(v)
    return out


def _store():
    return MemoryStore(semantic=SemanticSearch(embed_fn=fake_embed))


def test_lexical_mode_response_shape_unchanged():
    """Flag off (semantic=None): v4.0 response shape, no hybrid fields."""
    store = MemoryStore()
    store.remember("plain lexical memory about kubernetes")
    out = store.recall("kubernetes")
    assert "retrieval" not in out
    assert "semantic_similarity" not in out["results"][0]


def test_hybrid_finds_synonyms_with_zero_lexical_overlap():
    store = _store()
    target = store.remember("The automobile engine needs maintenance")
    store.remember("JWT tokens expire after 24 hours")

    out = store.recall("car repairs")
    assert out["retrieval"] == "hybrid_rrf"
    ids = [r["id"] for r in out["results"]]
    assert target["id"] in ids  # BM25 alone finds nothing for 'car repairs'
    hit = next(r for r in out["results"] if r["id"] == target["id"])
    assert hit["semantic_similarity"] > 0.9


def test_rrf_ranks_lexical_and_semantic_agreement_first():
    store = _store()
    both = store.remember("car engine repair checklist")        # lexical + semantic
    sem_only = store.remember("automobile maintenance notes")   # semantic only
    store.remember("JWT authentication flow")                   # neither

    out = store.recall("car repair")
    ids = [r["id"] for r in out["results"]]
    assert ids.index(both["id"]) < ids.index(sem_only["id"])


def test_write_through_update_and_stale_backfill():
    store = _store()
    m = store.remember("vehicle engine diagnostics")
    with get_db() as db:
        row = db.execute(
            "SELECT content_hash FROM memory_embeddings WHERE memory_id = ?",
            (m["id"],),
        ).fetchone()
    assert row["content_hash"] == content_hash("vehicle engine diagnostics")

    # update() refreshes the vector write-through
    store.update(m["id"], "vehicle engine overhaul")
    with get_db() as db:
        row = db.execute(
            "SELECT content_hash FROM memory_embeddings WHERE memory_id = ?",
            (m["id"],),
        ).fetchone()
    assert row["content_hash"] == content_hash("vehicle engine overhaul")

    # Simulate an edit made while the flag was off → stored hash goes stale
    with get_db() as db:
        db.execute(
            "UPDATE memories SET content = 'vehicle gearbox teardown' WHERE id = ?",
            (m["id"],),
        )
        db.commit()
    assert store.semantic.backfill(check_stale=True) == 1
    with get_db() as db:
        row = db.execute(
            "SELECT content_hash FROM memory_embeddings WHERE memory_id = ?",
            (m["id"],),
        ).fetchone()
    assert row["content_hash"] == content_hash("vehicle gearbox teardown")


def test_backfill_embeds_rows_written_while_disabled():
    plain = MemoryStore()  # flag off — no vectors written
    a = plain.remember("automobile inspection report")
    sem = SemanticSearch(embed_fn=fake_embed)
    assert sem.backfill() == 1

    hybrid = MemoryStore(semantic=sem)
    out = hybrid.recall("car")
    assert a["id"] in {r["id"] for r in out["results"]}


def test_forgotten_and_purged_leave_semantic_results():
    store = _store()
    m = store.remember("automobile engine notes")
    store.forget(m["id"])
    assert m["id"] not in {r["id"] for r in store.recall("car")["results"]}

    store.restore(m["id"])
    store.purge(m["id"])
    with get_db() as db:
        n = db.execute(
            "SELECT COUNT(*) FROM memory_embeddings WHERE memory_id = ?",
            (m["id"],),
        ).fetchone()[0]
    assert n == 0


def test_project_filter_applies_to_semantic_hits():
    store = _store()
    store.remember("automobile fleet records", project="garage")
    store.remember("automobile insurance papers", project="office")
    out = store.recall("car", project="garage")
    assert out["count"] == 1
    assert out["results"][0]["project"] == "garage"


@pytest.mark.skipif(_FASTEMBED, reason="fastembed installed — error path n/a")
def test_missing_extra_fails_loud_with_install_hint():
    with pytest.raises(RuntimeError, match=r"chronosmcp\[semantic\]"):
        SemanticSearch()  # no embed_fn and no fastembed


@pytest.mark.skipif(not _FASTEMBED, reason="semantic extra not installed")
def test_real_model_smoke():
    """End-to-end with the real ONNX model. Downloads ~70MB on first run;
    runs locally when the extra is installed, always skipped in CI."""
    store = MemoryStore(semantic=SemanticSearch())
    target = store.remember("The automobile engine needs maintenance soon")
    store.remember("JWT tokens expire after twenty-four hours")
    out = store.recall("my car needs repairs")
    assert out["retrieval"] == "hybrid_rrf"
    assert out["results"][0]["id"] == target["id"]
