# Chronos MCP

> Gives Claude a long-term memory and the ability to map out your projects.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Claude forgets everything when you start a new chat. You end up pasting the same context, rules, and project states over and over. Chronos fixes this by giving Claude a permanent, local memory it can search — and a structured graph it can use to track complex projects. Everything stays on your machine. No accounts, no APIs, no cloud.

---

## Quick Start

**1. Install**

```bash
git clone https://github.com/you/ChronosMCP
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
      // Path to wherever you cloned ChronosMCP
      "args": ["/absolute/path/to/ChronosMCP/chronos_mcp.py"],
      "env": {
        // Where Chronos saves your database. Defaults to chronos.db next to the script.
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

Forget the note about the old staging URL.

What tasks are blocking the payments feature right now?

What should I work on next in the api-rewrite project?
```

---

## Tools

Chronos gives Claude two distinct pipelines: a memory layer for free-text notes,
and a graph layer for structured project tracking.

### Memory

| Tool | What it does |
|---|---|
| `remember` | Saves a note, decision, or code snippet. Accepts optional `project` and `tags`. |
| `recall` | Finds the most relevant memories for a query. Blends keyword relevance with recency. |
| `forget` | Hides a memory from future searches. The record stays in the DB for history. |
| `update_memory` | Rewrites a memory's content. The previous version is snapshotted automatically. |
| `query_at` | Time-travel recall — reconstructs what Claude knew at any past timestamp. |
| `query_similar_memories` | Finds conceptually related memories using embedding similarity, not just tags. |

### Graph

| Tool | What it does |
|---|---|
| `add_event` | Adds a structured node (task, decision, sprint) to the knowledge graph. |
| `add_constraint` | Links two nodes so Claude understands what must happen first. |
| `suggest_next_tasks` | Reads the dependency graph and surfaces what's unblocked and ready to work on. |
| `query_similar` | Finds related nodes by complexity, priority, and graph position. |
| `analyze_structure` | Spots bottlenecks, orphaned nodes, and dependency cycles in your project. |
| `analyze_causal` | Estimates the real-world impact of a change by analyzing outcome patterns across tasks. |

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `CHRONOS_DB_PATH` | `./chronos.db` | Path to the SQLite database file |
| `CHRONOS_EMBED_MODEL` | *(none)* | HuggingFace model to enable `query_similar_memories` |
| `CHRONOS_RECENCY_WEIGHT` | `0.3` | How much recent memories are boosted in `recall` (0.0 = pure relevance) |

---

## How It Works

When you ask Claude to remember something, Chronos writes it to a local SQLite file and
indexes it using TF-IDF — no dependencies, no model downloads required. `recall` ranks
results by relevance and applies a gentle recency boost so newer memories surface first
without burying highly relevant older ones.

For project tracking, Chronos organizes nodes in a hyperbolic space rather than a flat
list. This means closely related tasks cluster naturally together even as your graph
grows — the geometry handles the relationships so you don't have to label everything
manually.

`query_at` time-travels by reconstructing a snapshot of your memory database at any
past timestamp, including resolving content that's since been edited. The database is a
single local file. Back it up however you back up files.

---

## License

MIT
