# chronos/db.py
# Responsibility: SQLite connection context manager + one-time schema initialisation.
# Schema init (init_db) is called once at server startup — NOT on every connection.

import os
import sqlite3
from contextlib import contextmanager

DB_PATH: str = os.environ.get(
    "CHRONOS_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chronos.db"),
)

# ---------------------------------------------------------------------------
# Schema DDL — defined once, applied once via init_db()
# ---------------------------------------------------------------------------

_DDL_STATEMENTS = [
    # GAP-10 FIX: column renamed from `type` (reserved word) to `event_type`
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
    # GAP-03 FIX: Permanent tombstone table (§2.4 — never deleted)
    """CREATE TABLE IF NOT EXISTS tombstones (
        node_id    TEXT PRIMARY KEY,
        event_id   TEXT NOT NULL,
        deleted_at TEXT NOT NULL,
        reason     TEXT
    )""",
    # Memory layer — free-text content store for remember/recall/forget
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
    # Memory content vectors — hyperbolic embeddings of memory content features.
    # Kept separate from node embeddings (different entity type, different feature space).
    """CREATE TABLE IF NOT EXISTS memory_vectors (
        memory_id  TEXT PRIMARY KEY,
        vector     BLOB NOT NULL,
        dim        INTEGER NOT NULL,
        project    TEXT NOT NULL DEFAULT 'default'
    )""",
    """CREATE INDEX IF NOT EXISTS idx_memory_vectors_project
       ON memory_vectors (project)""",
    # FIX #4: Memory content versions — enables true time-travel on query_at().
    # Each update_memory() call writes the OLD content here before overwriting.
    # query_at() uses this to reconstruct content as it existed at any timestamp.
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
]

# Columns added in v3.2 — applied via ALTER TABLE after initial CREATE.
# Each entry: (table, column, type_default). Idempotent: skipped if exists.
_COLUMN_MIGRATIONS = [
    ("memories", "confidence",    "REAL DEFAULT 0.5"),
    ("memories", "stability",     "REAL DEFAULT 1.0"),
    ("memories", "difficulty",    "REAL DEFAULT 0.5"),
    ("memories", "last_reviewed", "TEXT"),
    ("memories", "review_count",  "INTEGER DEFAULT 0"),
]


def _column_exists(conn, table: str, column: str) -> bool:
    """Check if a column already exists on a table (for idempotent migrations)."""
    info = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row[1] == column for row in info)


def init_db() -> None:
    """
    Apply schema DDL exactly once at server startup.
    Must be called before any tool handler runs.
    Also applies column migrations (v3.2 cognitive subsystem).
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        # FIX #7: Enable WAL mode for concurrent read/write safety.
        # Without WAL, async tool handlers block each other on DB access.
        conn.execute("PRAGMA journal_mode=WAL")
        for stmt in _DDL_STATEMENTS:
            conn.execute(stmt)
        # v3.2 column migrations — idempotent ALTER TABLE additions.
        for table, column, type_default in _COLUMN_MIGRATIONS:
            if not _column_exists(conn, table, column):
                conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} {type_default}"
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
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def get_tombstoned_ids(db) -> set:
    """Return set of node_ids currently recorded in the tombstones table."""
    rows = db.execute("SELECT node_id FROM tombstones").fetchall()
    return {r[0] for r in rows}
