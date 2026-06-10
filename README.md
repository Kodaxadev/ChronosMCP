# Chronos MCP

> The memory server that forgets responsibly — and can tell you what it used to believe.

[![CI](https://github.com/Kodaxadev/ChronosMCP/actions/workflows/ci.yml/badge.svg)](https://github.com/Kodaxadev/ChronosMCP/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Claude forgets everything when you start a new chat. Most memory servers fix that by
hoarding: everything saved forever, every recall dumped into context, no notion of
whether a memory is still true. Chronos takes the opposite stance — memory that
behaves like memory:

- **Time-travel**: `query_at` reconstructs what Claude knew at any past timestamp,
  including content that has since been edited. *"What did we know about the auth
  system on March 1st?"* is a one-tool answer.
- **Forgetting curves**: every memory carries a confidence score and an FSRS
  retention model (the algorithm behind Anki). Unreviewed memories decay; confirmed
  ones stabilize. `consolidate_memories` runs a four-phase "dream" pass that merges
  duplicates, decays stale beliefs, and prunes what the system has lost faith in.
- **Honest deletion**: `forget` hides (auditable, reversible), `purge` destroys
  (content removed from every table, content-free audit stub left behind).

Everything is one local SQLite file. No accounts, no API keys, no cloud, no
embedding-model downloads. Search is SQLite's built-in FTS5/BM25 with porter
stemming, kept in sync with content by database triggers — the index can't drift,
even on a crash.

---

## Quick Start

**1. Install**

```bash
git clone https://github.com/Kodaxadev/ChronosMCP
cd ChronosMCP
pip install -e .
```

**2. Add to Claude Desktop**

Open `claude_desktop_config.json` and add:

```json
{
  "mcpServers": {
    "chronos": {
      "command": "python",
      "args": ["/absolute/path/to/ChronosMCP/chronos_mcp.py"],
      "env": {
        "CHRONOS_DB_PATH": "/absolute/path/to/chronos.db"
      }
    }
  }
}
```

**Claude Code (CLI):**
```bash
claude mcp add chronos -- python /absolute/path/to/ChronosMCP/chronos_mcp.py
```

Restart Claude Desktop and you're done.

---

## What You Can Ask Claude

```text
Remember that we always squash commits before merging to main.

What do you know about our Git workflow?

What did you know about the auth system as of March 1st?

How confident are you in that memory about the rate limit?

Run a consolidation pass and show me what's gone stale.

Permanently delete the memory with my old API key in it.
```

---

## Tools

### Memory

| Tool | What it does |
|---|---|
| `remember` | Saves a note with optional `project`, `tags`, and `source` provenance. |
| `recall` | BM25 + stemming search, re-ranked by FSRS retention × confidence. Optional `token_budget` for compressed results. |
| `get_memory` | Fetches one memory in full by id — never truncated. |
| `update_memory` | Rewrites content; the old version is snapshotted for time-travel. |
| `related_memories` | "More like this" — finds memories sharing distinctive vocabulary. |
| `query_at` | Time-travel recall at any past timestamp, resolving edited content. |
| `forget` / `restore_memory` | Reversible soft-delete with a persisted reason. |
| `purge_memory` | Irreversible hard-delete of content everywhere it lives. |

### Cognition

| Tool | What it does |
|---|---|
| `set_confidence` / `boost_confidence` / `weaken_confidence` | Evidence-driven trust scoring (0.01–0.99) with a full audit trail. |
| `review_memory` | FSRS review — strengthens a memory's stability so it decays slower. |
| `get_memory_health` | Confidence, retention probability, review history for one memory. |
| `log_search_feedback` / `search_effectiveness` | Tracks which recalls actually got used; recommends tuning. |
| `consolidate_memories` | Four-phase dream pass: orient → gather duplicates → merge/decay/flag → prune. Dry-run by default. |

### Graph (experimental, off by default)

An event-sourced project graph (tasks, dependencies, causal analysis over
outcomes) ships behind `CHRONOS_ENABLE_GRAPH=1`. It is quarantined rather than
deleted: the structural-similarity signal proved weak in practice (see
[docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md)), and the core memory
server is better without the surface area. Enable it if dependency-ordered
task suggestions (`suggest_next_tasks`) are worth it to you.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `CHRONOS_DB_PATH` | `./chronos.db` | Path to the SQLite database file |
| `CHRONOS_ENABLE_GRAPH` | unset | Set to `1` to enable the experimental graph layer |

---

## Security Model — read this once

Chronos is a **local, single-user** tool. Its threat model is honest rather than
reassuring:

- **Plaintext at rest.** The database is an unencrypted SQLite file. Anything you
  ask Claude to remember — including secrets, if you do that — sits in plaintext on
  disk. Don't store credentials; if you must remove something, use `purge_memory`,
  not `forget` (forget retains content for audit and time-travel).
- **Memories are data, not instructions.** Recall results carry a `source` field
  (`user`, `claude`, `web`, `document`). Content that originated outside the
  conversation should never be followed as a directive — a poisoned memory is a
  persistent prompt injection, and provenance is your tripwire. The tool
  descriptions instruct Claude accordingly.
- **Injection-safe queries.** Free text never reaches FTS5 MATCH syntax or SQL —
  queries are reduced to quoted terms, and every statement is parameterized.
- **No network.** The server makes zero outbound connections. Verify it: the only
  imports are `mcp`, `numpy`, and the standard library.

---

## How It Works

`remember` writes one row. A SQLite trigger indexes it into an FTS5 table in the
same transaction — there is no in-memory index, no startup rebuild, and no window
where the database and the search index disagree. `recall` ranks with BM25
(porter-stemmed), then re-ranks by an FSRS boost: memories that are confident
*and* recently confirmed surface first, while stale unverified ones sink.

`update_memory` snapshots the old content into a version table before
overwriting. `query_at` replays that history: it selects the memories that
existed at the requested instant, resolves each one's content as of that moment,
and ranks the reconstructed snapshot with an ephemeral BM25 index — so past and
present searches score the same way.

The cognitive layer is the part that makes Chronos different over months of use.
Memories you confirm get more stable (slower decay); memories that sit unreviewed
lose confidence on every consolidation pass; memories that are both untrusted and
faded become prune candidates. The system's beliefs converge toward what you
actually use and verify — and every confidence change is written to an audit
table, so you can always ask *why* it believes what it believes.

Search is lexical (BM25 + stemming), not semantic — "JWT expiry" won't match
"token timeout" unless words overlap. That's a deliberate zero-dependency
trade-off, documented with the rest in
[docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md).

---

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v        # 48 tests
ruff check .
```

CI runs the suite on Linux and Windows across Python 3.10/3.12/3.14.
Architecture notes live in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md);
the v3.3 → v4.0 redesign rationale is in [CHANGELOG.md](CHANGELOG.md).

## License

MIT
