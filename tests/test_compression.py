# tests/test_compression.py
# Token-budget compression tiers: trim, compact, summarize, and the
# progressive pipeline.

from chronos.compression import (
    compress_results,
    tier1_trim,
    tier2_compact,
    tier3_summarize,
)


def _result(content: str, score: float = 1.0) -> dict:
    return {
        "id": "x",
        "content": content,
        "score": score,
        "token_estimate": max(1, round(len(content.split()) * 0.75)),
    }


def test_tier1_trims_only_oversized():
    short = _result("short content here")
    long = _result(" ".join(f"w{i}" for i in range(400)))  # ~300 tokens
    out = tier1_trim([short, long])
    assert "trimmed" not in out[0]
    assert out[1]["trimmed"] is True
    assert out[1]["token_estimate"] <= 155  # ~TRIM_TOKENS with rounding slack
    assert out[1]["content"].endswith("[...]")


def test_tier2_drops_tail_but_never_returns_empty():
    results = [_result(" ".join(f"w{i}" for i in range(200)), score=s)
               for s in (3.0, 2.0, 1.0)]
    kept = tier2_compact(results, budget=200, overhead=40)
    assert 1 <= len(kept) < 3
    assert kept[0]["score"] == 3.0  # highest score survives

    # Even when the first result alone exceeds the budget, one is returned
    kept = tier2_compact(results, budget=10, overhead=40)
    assert len(kept) == 1


def test_tier3_stubs_record_original_size():
    long = _result(" ".join(f"w{i}" for i in range(600)))
    out = tier3_summarize([long])
    assert out[0]["summarized"] is True
    assert out[0]["original_tokens"] > out[0]["token_estimate"]


def test_pipeline_applies_tiers_progressively():
    results = [_result(" ".join(f"w{i}" for i in range(400)), score=s)
               for s in (3.0, 2.0, 1.0)]
    out = compress_results(results, budget=120, overhead=40)
    assert out["compression_applied"][0] == "trim"
    assert out["total_tokens"] <= 200  # respects budget within stub tolerance

    no_budget = compress_results([_result("tiny")], budget=None)
    assert no_budget["compression_applied"] == ["trim"]
    assert no_budget["results"][0]["content"] == "tiny"
