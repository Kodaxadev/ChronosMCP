# CHRONOS v3.3 Development Guide

## Prerequisites

- Python 3.10 or later
- pip or uv

## Setup

```bash
git clone <repo>
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

## Testing

Run the normal test suite through pytest:

```bash
python -m pytest -q
```

Compile-check importable modules:

```bash
python -m compileall -q chronos chronos_mcp.py tests
```

The tests set `CHRONOS_DB_PATH` to a temporary SQLite file before importing the
application modules, so they do not mutate a real Chronos database.

## File Limits

Every source or markdown file must stay under 400 lines.

| Threshold | Action |
|---|---|
| 150-300 lines | Target range |
| 300 lines | Review for split opportunities |
| 400 lines | Hard limit: create a new module first |

Check line counts with:

```powershell
Get-ChildItem -Recurse -File -Include *.py,*.md,*.toml |
  Where-Object { $_.FullName -notmatch '\\.venv\\|__pycache__|\\.git\\' } |
  ForEach-Object {
    $lines = (Get-Content -LiteralPath $_.FullName | Measure-Object -Line).Lines
    if ($lines -gt 400) { "$lines $($_.FullName)" }
  }
```

## Module Ownership

- `chronos_mcp.py`: startup orchestration only
- `chronos/db.py`: SQLite path, schema, connection lifecycle
- `chronos/uuid7.py`: ID generation
- `chronos/validation.py`: event validation
- `chronos/memory.py`: memory lifecycle
- `chronos/time_travel.py`: historical memory reconstruction
- `chronos/compression.py`: recall token-budget compression
- `chronos/tfidf.py`: keyword retrieval index
- `chronos/mem_embed.py`: structural memory vectors
- `chronos/geometry.py`: Poincare ball geometry and node embeddings
- `chronos/analyzers.py`: causal, graph structure, and dependency analysis
- `chronos/beliefs.py`: confidence and FSRS state logic
- `chronos/consolidation.py`: memory maintenance workflow
- `chronos/consolidation_config.py`: consolidation thresholds
- `chronos/*_tools.py`: MCP tool registration for each subsystem

When adding behavior, choose the owning module first. If no existing module owns
the behavior cleanly, create a focused new module.

## Dependencies

Runtime dependencies live in `pyproject.toml`:

```toml
mcp[cli]~=1.6
numpy~=2.2
```

Development-only tooling is declared in the `dev` extra. Keep runtime
dependencies sparse; prefer stdlib plus numpy unless a new package removes
meaningful complexity.

## Database Rules

- Call `init_db()` once at startup before registering tools.
- Use `get_db()` for per-operation connections.
- Commit DB writes before updating in-memory indexes.
- Keep schema migrations idempotent.
- Never write real secrets or tokens into tests, docs, or fixtures.

## Before Finishing Work

1. Run `python -m pytest -q`.
2. Run `python -m compileall -q chronos chronos_mcp.py tests`.
3. Check file line counts.
4. Review `git status --short`.
