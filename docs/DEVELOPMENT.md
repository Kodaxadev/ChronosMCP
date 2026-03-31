# CHRONOS v3.1 — Development Guide

## Prerequisites

- Python 3.10 or later
- pip

## Setup

```bash
git clone <repo>
cd ChronosMCP
pip install -r requirements.txt
```

That is the complete setup. There are no Docker containers, no external databases, no API keys, and no environment configuration required beyond `CHRONOS_DB_PATH` (optional).

## Running the Server

```bash
# Stdio transport (default — for MCP client connections)
python chronos_mcp.py

# Explicit stdio via MCP CLI
mcp run chronos_mcp.py
```

The server writes to `chronos.db` in the same directory as `chronos_mcp.py` by default. Override with:

```bash
CHRONOS_DB_PATH=/path/to/custom.db python chronos_mcp.py
```

## Running Tests

```bash
python test_integration.py
```

The test suite creates a temporary SQLite database (`chronos_test.db`), runs all 98 tests, and deletes the database when complete. Tests must be run from the `ChronosMCP/` root directory (not from `chronos/`).

```bash
# Expected output
Results: 98/98 passed  |  0 failed
```

If you see `ModuleNotFoundError: No module named 'chronos'`, you are running from the wrong directory.

### Audit Reproduction Tests

A separate reproduction script confirms the bugs found during the audit pass and verifies fixes:

```bash
python test_audit_repro.py
```

This script does not add to the integration test count — it is a diagnostic tool for validating specific bug fixes.

---

## Module Governance

These rules are enforced across all modules and must be maintained on any future changes.

### File Size Limits

| Threshold | Action |
|-----------|--------|
| 150–300 lines | Target range |
| 400 lines | Warning — review for split opportunities |
| 500 lines | Hard limit — do not add new logic; create a new module first |
| 50 lines per function | Maximum — extract helpers if exceeded |

Current file sizes (as of v3.1):

| File | Lines | Status |
|------|-------|--------|
| `chronos/memory.py` | 449 | ⚠️ Above warning threshold — `query_at()` is extraction candidate |
| `chronos/analyzers.py` | 265 | OK |
| `chronos/analysis_tools.py` | 245 | OK |
| `chronos/tfidf.py` | 217 | OK |
| `chronos/mem_embed.py` | 203 | OK |
| `chronos/tools.py` | 190 | OK |
| `chronos/geometry.py` | 185 | OK |
| `chronos/graph_tools.py` | 169 | OK |
| `chronos/db.py` | 131 | OK |
| `chronos/memory_tools.py` | 103 | OK |
| `chronos/validation.py` | 57 | OK |
| `chronos/uuid7.py` | 26 | OK |

### One Concern Per File

Each module has a single stated responsibility declared in its header comment. Do not combine:

- UI / routing logic with domain logic
- DB access with business rules
- Tool registration with analytics

When adding features, ask: "which module owns this?" If the answer is "none of them," create a new module before adding code to an existing one.

### Module Organization

```
chronos/
  db.py           — SQLite infrastructure only
  uuid7.py        — ID generation only
  validation.py   — Event schema validation only
  geometry.py     — Poincaré ball geometry + HyperbolicEmbedder only
  tfidf.py        — TF-IDF index only
  mem_embed.py    — MemoryEmbedder only
  memory.py       — MemoryStore lifecycle only
  analyzers.py    — CausalAnalyzer, StructureAnalyzer, ConstraintSolver only
  tools.py        — register() + remember/recall/forget/query_at tool registration
  graph_tools.py  — add_event/query_similar/add_constraint tool registration
  analysis_tools.py — analyze_*/suggest_next_tasks tool registration
  memory_tools.py — update_memory/query_similar_memories tool registration
```

If `memory.py` reaches 500 lines, extract `query_at()` into `chronos/time_travel.py`.

### Patch-First Editing

When modifying existing behavior, prefer targeted edits over rewrites. Always:

1. Read the file before editing
2. Make the smallest change that fixes the issue
3. Add a comment explaining the fix and what the old behavior was
4. Run `test_integration.py` after every change

### No Silent Coercion

Following `validation.py`'s design: never silently correct bad input. If a parameter is wrong, raise or return a structured error. Callers must know when their input was rejected.

---

## Adding a New Tool

1. Decide which registration file owns the tool (see module table above)
2. If that file is at or approaching 400 lines, create a new `chronos/X_tools.py` first
3. Add the tool function inside the registration function (closure pattern — captures singletons, avoids globals)
4. Add parameter validation at the top of the function; clamp numeric inputs to safe ranges
5. Update `tools.register()` in `tools.py` to call the new registration function if needed
6. Add integration tests in `test_integration.py` covering: normal case, empty/missing input, boundary inputs, error case
7. Update `docs/API_REFERENCE.md` with the new tool's parameter table and return shape

---

## Adding a New DB Table

1. Add the `CREATE TABLE IF NOT EXISTS` statement to `_DDL_STATEMENTS` in `db.py`
2. Add any supporting indexes immediately below the table definition
3. Keep the DDL comment aligned: what the table is for, which module writes it, which module reads it
4. Do not add migration logic — `CREATE TABLE IF NOT EXISTS` is idempotent; existing databases pick up new tables on next `init_db()` call
5. Document the schema in `docs/ARCHITECTURE.md` under the Database Schema section

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHRONOS_DB_PATH` | `<repo_root>/chronos.db` | Absolute path to the SQLite database file. Override to separate test and production databases or to place the DB in a known persistent location. |

---

## Dependencies and Versioning

`requirements.txt` uses compatible-release pins (`~=`) for reproducibility:

```
mcp[cli]~=1.6   # resolves to 1.26.x — ~= prevents v2 breaking changes
numpy~=2.2      # vector math; no sklearn, no scipy
```

**Why `~=` not `==`?** Exact pins (`==1.26.0`) cause friction when the host already has a compatible version installed. Compatible-release pins (`~=1.6`) allow patch updates while blocking breaking minor/major changes.

**Adding a new dependency:** Resist it. The "zero external services, two pip packages" constraint is a feature. Before adding a package, ask whether the functionality can be implemented with stdlib + numpy. If a package is genuinely needed, add it to `requirements.txt` with a compatible-release pin and document the reason in the module that uses it.

---

## Code Quality Standards

**Deterministic hashing:** Use `hashlib.sha256(s.encode()).hexdigest()` for any bucketing or identity hashing that must be stable across Python versions. Never use `hash()` — Python's built-in hash is randomized by default and differs between processes.

**Timestamp format:** Always write timestamps as `datetime.now().isoformat()` (tz-naive). Never write tz-aware timestamps to the DB unless you update all comparison and sorting logic to handle them.

**Parameterized queries:** Every SQL statement with user-supplied values must use `?` placeholders. No f-string or format-string SQL construction except for structural elements (table names, `IN (?, ?, ?)` placeholder counts generated programmatically from parameterized lists).

**Error propagation:** Never swallow error keys from downstream functions. If a called function can return `{"error": "..."}`, check for it and surface it to the caller. The causal analysis swallow bug is the canonical example of what not to do.

**Numpy safety:** Before calling `np.arccosh`, ensure the argument is `>= 1.0`. Before computing pairwise distances, ensure all vectors have the same dimension. The geometry module's `_clip_norm`, `max(1.0, arg)`, and `load_from_db` pad/truncate patterns are the reference implementations.
