# CHRONOS v4.0 Development Guide

## Prerequisites

- Python 3.10 or later, from a standard build (SQLite with FTS5 — python.org,
  conda, and OS packages all qualify; the server fails loudly at startup if
  FTS5 is missing)
- pip or uv

## Setup

```bash
git clone https://github.com/Kodaxadev/ChronosMCP
cd ChronosMCP
pip install -e ".[dev]"
```

Chronos has no Docker containers, external databases, API keys, or cloud
services. `CHRONOS_DB_PATH` is optional and controls where the SQLite file is
stored.

## Running

```bash
python chronos_mcp.py
```

Use an isolated database while developing:

```bash
CHRONOS_DB_PATH=/path/to/dev.db python chronos_mcp.py
```

On PowerShell:

```powershell
$env:CHRONOS_DB_PATH = "C:\path\to\dev.db"
python chronos_mcp.py
```

Enable the experimental graph layer with `CHRONOS_ENABLE_GRAPH=1`.

## Testing & Linting

```bash
python -m pytest -q     # 48 tests; each gets a fresh temp database (conftest.py)
ruff check .            # must be clean — CI enforces it
```

CI (`.github/workflows/ci.yml`) runs both on Linux + Windows across
Python 3.10 / 3.12 / 3.14. Tests never touch a real Chronos database:
`db.db_path()` resolves `CHRONOS_DB_PATH` at call time and the autouse
fixture points it at `tmp_path`.

When you fix a bug, add a regression test in the matching `tests/test_*.py`
file and name it after the failure mode (see
`test_add_event_during_resize_does_not_deadlock` for the pattern).

## File Limits

Every source or markdown file must stay under 400 lines.

| Threshold | Action |
|---|---|
| 150-300 lines | Target range |
| 300 lines | Review for split opportunities |
| 400 lines | Hard limit: create a new module first |

## Module Ownership

- `chronos_mcp.py`: startup orchestration + graph-layer gating only
- `chronos/db.py`: SQLite path, schema, migrations, FTS table + triggers
- `chronos/search.py`: BM25 retrieval, MATCH sanitization, snapshot ranking,
  pairwise similarity
- `chronos/ranking.py`: recall pipeline — hybrid candidates, RRF fusion,
  FSRS boost, response assembly
- `chronos/semantic.py`: optional local embeddings (CHRONOS_SEMANTIC=1)
- `chronos/memory.py`: remember / get / update + MemoryStore facade
- `chronos/lifecycle.py`: forget / restore / purge
- `chronos/time_travel.py`: historical memory reconstruction
- `chronos/compression.py`: opt-in token-budget compression
- `chronos/beliefs.py`: confidence and FSRS state logic
- `chronos/consolidation.py`: consolidation mutations (merge/decay/prune)
- `chronos/consolidation_scan.py`: consolidation read-only scans
- `chronos/consolidation_config.py`: consolidation thresholds
- `chronos/uuid7.py`: ID generation
- `chronos/validation.py`: graph event validation
- `chronos/geometry.py`, `chronos/analyzers.py`: graph layer (flag-gated)
- `chronos/*_tools.py`: MCP tool registration for each subsystem

When adding behavior, choose the owning module first. If no existing module owns
the behavior cleanly, create a focused new module.

## Dependencies

Runtime dependencies live in `pyproject.toml`:

```toml
mcp[cli]~=1.6
numpy~=2.2
```

numpy is used by the flag-gated graph layer and semantic search. Keep runtime
dependencies sparse; prefer stdlib unless a new package removes meaningful
complexity. Development-only tooling (pytest, ruff) is declared in the `dev`
extra; the optional `semantic` extra (fastembed) is required only when
`CHRONOS_SEMANTIC=1`, and the `encryption` extra (sqlcipher3-wheels) only
when `CHRONOS_DB_KEY` is set. Semantic tests inject a fake `embed_fn` — CI
never downloads a model; the real-model smoke test runs only where the extra
is installed. Encryption tests skip without the driver and run in CI's
dedicated encryption job (Linux + Windows).

## Database Rules

- Call `init_db()` once at startup before registering tools.
- Use `get_db()` for per-operation connections.
- **Never write to `memories_fts` directly** — the triggers in db.py are the
  only write path (the one-time backfill in `init_db()` is the sole exception).
- Never open a second write connection while a write transaction is in
  flight (this is the deadlock class fixed in v4.0 — see graph_tools.add_event).
- Keep schema migrations additive and idempotent.
- Never write real secrets or tokens into tests, docs, or fixtures.

## Before Finishing Work

1. `python -m pytest -q` — all green.
2. `ruff check .` — clean.
3. Check file line counts (400 hard limit).
4. Review `git status --short`.
