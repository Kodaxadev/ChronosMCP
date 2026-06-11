# chronos/db.py
# Responsibility: SQLite connection management + one-time schema initialisation.
#
# v4.0: full-text search moved into the database itself. The memories_fts
# FTS5 virtual table is kept in sync with the memories table by triggers,
# so index updates commit atomically with the content writes they mirror.
# There is no in-memory index, no startup rebuild, and no consistency window.

import hashlib
import os
import sqlite3
from contextlib import contextmanager

_DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "chronos.db"
)

_ENCRYPTION_HINT = (
    "CHRONOS_DB_KEY is set but the encryption driver is missing. "
    'Fix: pip install "chronosmcp[encryption]" '
    "(SQLCipher wheels — no compiler needed)"
)

_KEY_MISMATCH_HINT = (
    "Could not read the database with the current CHRONOS_DB_KEY setting. "
    "One of: (a) the key is wrong; (b) the database is plaintext but "
    "CHRONOS_DB_KEY is set — encrypt it first with "
    "`python scripts/encrypt_db.py encrypt`; (c) the database is encrypted "
    "but CHRONOS_DB_KEY is unset."
)


def db_path() -> str:
    """Resolve the DB path at call time (not import time) so tests and
    multi-environment setups can repoint CHRONOS_DB_PATH dynamically."""
    return os.environ.get("CHRONOS_DB_PATH", _DEFAULT_DB_PATH)


def db_key():
    """Passphrase from CHRONOS_DB_KEY, or None (plaintext mode)."""
    return os.environ.get("CHRONOS_DB_KEY") or None


def _driver():
    """stdlib sqlite3 in plaintext mode; sqlcipher3 when a key is set."""
    if db_key() is None:
        return sqlite3
    try:
        import sqlcipher3
        return sqlcipher3
    except ImportError as exc:
        raise RuntimeError(_ENCRYPTION_HINT) from exc


# (path, passphrase) -> PRAGMA key value, either ("raw", "x'<key><salt>'")
# or ("pass", passphrase). SQLCipher's PBKDF2 costs ~300ms per connection;
# Chronos opens a connection per operation, so we pay the KDF once here,
# derive the same 32-byte key locally from the salt stored in the file
# header, and reuse SQLCipher's raw key+salt form (~1ms) ever after.
_key_cache: dict = {}


def _pragma_quote(passphrase: str) -> str:
    return passphrase.replace("'", "''")


def _resolve_key(path: str, passphrase: str, drv):
    """Return the cached PRAGMA key value for this (path, passphrase),
    deriving and validating it on first use. Raises RuntimeError with
    guidance when the key cannot read the database."""
    cached = _key_cache.get((path, passphrase))
    if cached is not None:
        return cached

    # Slow path, once: open with the passphrase and prove readability.
    conn = drv.connect(path)
    try:
        conn.execute(f"PRAGMA key = '{_pragma_quote(passphrase)}'")
        try:
            # Also initialises the header (and salt) on a brand-new file.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone()
        except drv.DatabaseError as exc:
            raise RuntimeError(_KEY_MISMATCH_HINT) from exc
        kdf_iter = int(conn.execute("PRAGMA kdf_iter").fetchone()[0])
    finally:
        conn.close()

    # Derive the raw key from the header salt; verify before trusting.
    entry = ("pass", passphrase)  # safe fallback: KDF per connection
    try:
        with open(path, "rb") as fh:
            salt = fh.read(16)
        key = hashlib.pbkdf2_hmac(
            "sha512", passphrase.encode("utf-8"), salt, kdf_iter, dklen=32
        )
        raw = f"x'{key.hex()}{salt.hex()}'"
        probe = drv.connect(path)
        try:
            probe.execute(f'PRAGMA key = "{raw}"')
            probe.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone()
            entry = ("raw", raw)
        finally:
            probe.close()
    except (OSError, drv.DatabaseError):
        pass  # nonstandard cipher settings — keep the passphrase fallback

    _key_cache[(path, passphrase)] = entry
    return entry


def _connect():
    """Open a connection with the right driver and key. Returns (conn, drv)."""
    drv = _driver()
    conn = drv.connect(db_path())
    passphrase = db_key()
    if passphrase is not None:
        mode, value = _resolve_key(db_path(), passphrase, drv)
        if mode == "raw":
            conn.execute(f'PRAGMA key = "{value}"')
        else:
            conn.execute(f"PRAGMA key = '{_pragma_quote(value)}'")
    return conn, drv


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
    # --- v4.1: semantic vectors (optional, populated only when
    # CHRONOS_SEMANTIC=1). content_hash detects stale vectors after edits
    # made while the flag was off; model allows clean model switches. ---
    """CREATE TABLE IF NOT EXISTS memory_embeddings (
        memory_id    TEXT PRIMARY KEY,
        vector       BLOB NOT NULL,
        dim          INTEGER NOT NULL,
        model        TEXT NOT NULL,
        content_hash TEXT NOT NULL
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
    conn, drv = _connect()
    try:
        # WAL mode for concurrent read/write safety. Also the readability
        # probe: an encrypted database opened without its key (or a
        # plaintext one opened WITH a key) fails right here — fail loud
        # with guidance instead of erroring on the first tool call.
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except drv.DatabaseError as exc:
            raise RuntimeError(_KEY_MISMATCH_HINT) from exc
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
        except drv.OperationalError as exc:
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
    Yield an open SQLite connection with Row factory set (encrypted via
    SQLCipher when CHRONOS_DB_KEY is set — see _resolve_key for why this
    costs ~1ms, not a per-connection KDF).
    Schema is NOT initialised here — call init_db() at startup instead.
    """
    conn, drv = _connect()
    conn.row_factory = drv.Row
    try:
        yield conn
    finally:
        conn.close()


def get_tombstoned_ids(db) -> set:
    """Return set of node_ids currently recorded in the tombstones table."""
    rows = db.execute("SELECT node_id FROM tombstones").fetchall()
    return {r[0] for r in rows}
