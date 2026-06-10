# chronos/db.py
# Responsibility: SQLite connection management + one-time schema initialisation.
#
# v4.0: full-text search moved into the database itself. The memories_fts
# FTS5 virtual table is kept in sync with the memories table by triggers,
# so index updates commit atomically with the content writes they mirror.
# There is no in-memory index, no startup rebuild, and no consistency window.

import os
import sqlite3
from contextlib import contextmanager

_DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "chronos.db"
)


def db_path() -> str:
    """Resolve the DB path at call time (not import time) so tests and
    multi-environment setups can repoint CHRONOS_DB_PATH dynamically."""
    return os.environ.get("CHRONOS_DB_PATH", _DEFAULT_DB_PATH)


# ---------------------------------------------------------------------------
# Schema DDL — defined once, applied once via init_db()
# ---------------------------------------------------------------------------

_DDL_STATEMENTS = [
    # --- Graph layer (optional, enabled via CHRONOS_ENABLE_GRAPH) ---
    """CREATE TABLE IF NOT EXISTS events (
        id             TEXT PRIMARY KEY,
        aggregate_id   TEXT NOT NULL,
        event_type     TEXT NOT NULL,
        ts             TEXT NOT NULL,
        payload        TEXT NOT NULL,
        schema_version TEXT NOT NULL DEFAULT '2.3'
    )""",
    """CREATE TABLE IF NOT EXISTS embeddings (
        node_id  TEXT PRIMARY KEY,
        vector   BLOB NOT NULL,
        version  INTEGER NOT NULL,
        dim      INTEGER NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS causal_results (
        id         TEXT PRIMARY KEY,
        treatment  TEXT NOT NULL,
        outcome    TEXT NOT NULL,
        ate        REAL NOT NULL,
        n_samples  INTEGER NOT NULL,
        status     TEXT NOT NULL DEFAULT 'observational'
    )""",
    """CREATE TABLE IF NOT EXISTS constraints (
        id              TEXT PRIMARY KEY,
        node_id         TEXT NOT NULL,
        constraint_type TEXT NOT NULL,
        priority        INTEGER NOT NULL,
        data            TEXT NOT NULL
    )""",
    # Permanent tombstone table for graph nodes (§2.4 — never deleted)
    """CREATE TABLE IF NOT EXISTS tombstones (
        node_id    TEXT PRIMARY KEY,
        event_id   TEXT NOT NULL,
        deleted_at TEXT NOT NULL,
        reason     TEXT
    )""",
    # --- Memory layer — free-text content store for remember/recall/forget ---
    """CREATE TABLE IF NOT EXISTS memories (
        id         TEXT PRIMARY KEY,
        project    TEXT NOT NULL DEFAULT 'default',
        content    TEXT NOT NULL,
        tags       TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        forgotten  INTEGER NOT NULL DEFAULT 0
    )""",
    """CREATE INDEX IF NOT EXISTS idx_memories_project
       ON memories (project, forgotten)""",
    # Index for query_at() time-travel — filters on created_at for every call
    """CREATE INDEX IF NOT EXISTS idx_memories_created_at
       ON memories (created_at)""",
    # Memory content versions — enables true time-travel on query_at().
    # Each update_memory() call writes the OLD content here before overwriting.
    """CREATE TABLE IF NOT EXISTS memory_versions (
        id         TEXT PRIMARY KEY,
        memory_id  TEXT NOT NULL,
        content    TEXT NOT NULL,
        valid_from TEXT NOT NULL,
        valid_to   TEXT NOT NULL
    )""",
    """CREATE INDEX IF NOT EXISTS idx_memory_versions_lookup
       ON memory_versions (memory_id, valid_from, valid_to)""",
    # --- Cognitive subsystem tables (v3.2) ---
    # Audit trail for confidence changes — one row per adjustment.
    """CREATE TABLE IF NOT EXISTS belief_updates (
        id             TEXT PRIMARY KEY,
        memory_id      TEXT NOT NULL,
        old_confidence REAL NOT NULL,
        new_confidence REAL NOT NULL,
        reason         TEXT NOT NULL,
        updated_at     TEXT NOT NULL
    )""",
    """CREATE INDEX IF NOT EXISTS idx_belief_updates_memory
       ON belief_updates (memory_id, updated_at)""",
    # Search feedback for meta-learning — tracks which recall results get used.
    """CREATE TABLE IF NOT EXISTS search_feedback (
        id          TEXT PRIMARY KEY,
        query       TEXT NOT NULL,
        memory_id   TEXT NOT NULL,
        used        INTEGER NOT NULL DEFAULT 0,
        recalled_at TEXT NOT NULL
    )""",
    """CREATE INDEX IF NOT EXISTS idx_search_feedback_query
       ON search_feedback (recalled_at)""",
    # --- v4.0: hard-delete audit (no content — ids and timestamps only) ---
    """CREATE TABLE IF NOT EXISTS purge_log (
        id        TEXT PRIMARY KEY,
        memory_id TEXT NOT NULL,
        purged_at TEXT NOT NULL
    )""",
]

# v4.0: FTS5 full-text index + sync triggers.
# The triggers are the single write path into memories_fts — application code
# never inserts into it directly (except the one-time backfill in init_db).
# Because triggers run inside the writing transaction, the index can never
# drift from the memories table, even on crash.
_FTS_DDL_STATEMENTS = [
    """CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
        memory_id UNINDEXED,
        content,
        tokenize='porter unicode61'
    )""",
    """CREATE TRIGGER IF NOT EXISTS trg_memories_ai
       AFTER INSERT ON memories BEGIN
           INSERT INTO memories_fts (memory_id, content)
           SELECT new.id, new.content WHERE new.forgotten = 0;
       END""",
    # On any UPDATE (content edit, forget, restore, confidence change) the row
    # is re-indexed from scratch: delete then conditional re-insert. Slight
    # write amplification on metadata-only updates, in exchange for an index
    # that is correct by construction.
    """CREATE TRIGGER IF NOT EXISTS trg_memories_au
       AFTER UPDATE ON memories BEGIN
           DELETE FROM memories_fts WHERE memory_id = old.id;
           INSERT INTO memories_fts (memory_id, content)
           SELECT new.id, new.content WHERE new.forgotten = 0;
       END""",
    """CREATE TRIGGER IF NOT EXISTS trg_memories_ad
       AFTER DELETE ON memories BEGIN
           DELETE FROM memories_fts WHERE memory_id = old.id;
       END""",
]

# Columns added after initial release — applied via ALTER TABLE.
# Each entry: (table, column, type_default). Idempotent: skipped if exists.
_COLUMN_MIGRATIONS = [
    # v3.2 cognitive subsystem
    ("memories", "confidence",    "REAL DEFAULT 0.5"),
    ("memories", "stability",     "REAL DEFAULT 1.0"),
    ("memories", "difficulty",    "REAL DEFAULT 0.5"),
    ("memories", "last_reviewed", "TEXT"),
    ("memories", "review_count",  "INTEGER DEFAULT 0"),
    # v4.0 provenance + honest forget semantics
    ("memories", "source",        "TEXT DEFAULT 'user'"),
    ("memories", "forget_reason", "TEXT"),
]


def _column_exists(conn, table: str, column: str) -> bool:
    """Check if a column already exists on a table (for idempotent migrations)."""
    info = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row[1] == column for row in info)


def init_db() -> None:
    """
    Apply schema DDL exactly once at server startup.
    Must be called before any tool handler runs.
    Applies column migrations, creates the FTS5 index + triggers, and
    backfills the index for databases created before v4.0.
    """
    conn = sqlite3.connect(db_path())
    try:
        # WAL mode for concurrent read/write safety.
        conn.execute("PRAGMA journal_mode=WAL")
        for stmt in _DDL_STATEMENTS:
            conn.execute(stmt)
        for table, column, type_default in _COLUMN_MIGRATIONS:
            if not _column_exists(conn, table, column):
                conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} {type_default}"
                )
        # FTS5 index + triggers. Fail loudly if this Python's SQLite lacks FTS5
        # rather than degrading into a server whose recall returns nothing.
        try:
            for stmt in _FTS_DDL_STATEMENTS:
                conn.execute(stmt)
        except sqlite3.OperationalError as exc:
            raise RuntimeError(
                "This Python's SQLite build does not support FTS5, which "
                "Chronos v4 requires for search. Use a standard python.org "
                f"build (3.10+). Underlying error: {exc}"
            ) from exc
        # One-time backfill for pre-v4 databases. Idempotent.
        conn.execute(
            """INSERT INTO memories_fts (memory_id, content)
               SELECT id, content FROM memories
               WHERE forgotten = 0
                 AND id NOT IN (SELECT memory_id FROM memories_fts)"""
        )
        conn.commit()
    finally:
        conn.close()


@contextmanager
def get_db():
    """
    Yield an open SQLite connection with Row factory set.
    Schema is NOT initialised here — call init_db() at startup instead.
    """
    conn = sqlite3.connect(db_path())
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def get_tombstoned_ids(db) -> set:
    """Return set of node_ids currently recorded in the tombstones table."""
    rows = db.execute("SELECT node_id FROM tombstones").fetchall()
    return {r[0] for r in rows}
