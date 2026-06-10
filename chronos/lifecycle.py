# chronos/lifecycle.py
# Responsibility: memory lifecycle transitions — forget (soft-delete),
# restore (un-forget), purge (hard-delete).
#
# Extracted from memory.py to keep modules under the 400-line limit,
# following the same delegation pattern as time_travel.py. MemoryStore
# delegates here; these functions own the DB writes.
#
# Semantics (v4.0):
#   forget  — hides from search, content RETAINED for audit + time-travel.
#             The reason is persisted to memories.forget_reason.
#   restore — reverses forget; FTS trigger re-indexes in the same txn.
#   purge   — permanently deletes content everywhere it lives. Irreversible.
#             Leaves a content-free row in purge_log (id + timestamp only).

from datetime import datetime

from chronos.db import get_db
from chronos.uuid7 import uuid7


def forget(memory_id: str, reason: str = "manual") -> dict:
    """
    Soft-delete a memory. The FTS trigger removes it from search in the
    same transaction. Content is retained — use purge() for real removal.

    Returns: {id, status, reason} — status is 'forgotten', 'not_found',
    or 'already_forgotten'.
    """
    with get_db() as db:
        row = db.execute(
            "SELECT id, forgotten FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()

        if not row:
            return {"id": memory_id, "status": "not_found", "reason": reason}
        if row["forgotten"] == 1:
            return {"id": memory_id, "status": "already_forgotten",
                    "reason": reason}

        db.execute(
            """UPDATE memories SET forgotten = 1, forget_reason = ?,
               updated_at = ? WHERE id = ?""",
            (reason, datetime.now().isoformat(), memory_id),
        )
        db.commit()

    return {"id": memory_id, "status": "forgotten", "reason": reason}


def restore(memory_id: str) -> dict:
    """
    Un-forget a memory. The FTS trigger re-indexes it in the same
    transaction, so it reappears in recall immediately.

    Returns: {id, status} — 'restored', 'not_forgotten', or 'not_found'.
    """
    with get_db() as db:
        row = db.execute(
            "SELECT id, forgotten FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()

        if not row:
            return {"id": memory_id, "status": "not_found"}
        if row["forgotten"] == 0:
            return {"id": memory_id, "status": "not_forgotten"}

        db.execute(
            """UPDATE memories SET forgotten = 0, forget_reason = NULL,
               updated_at = ? WHERE id = ?""",
            (datetime.now().isoformat(), memory_id),
        )
        db.commit()

    return {"id": memory_id, "status": "restored"}


def purge(memory_id: str) -> dict:
    """
    Permanently delete a memory and every copy of its content: the memories
    row (FTS cleaned by trigger), all memory_versions snapshots,
    belief_updates audit rows, and search_feedback rows. A content-free
    record is written to purge_log.

    Irreversible. This is what 'actually forget it' means; forget() only hides.

    Returns: {id, status, versions_removed, purged_at}
    """
    with get_db() as db:
        row = db.execute(
            "SELECT id FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if not row:
            return {"id": memory_id, "status": "not_found"}

        now = datetime.now().isoformat()
        versions = db.execute(
            "DELETE FROM memory_versions WHERE memory_id = ?", (memory_id,)
        ).rowcount
        db.execute(
            "DELETE FROM belief_updates WHERE memory_id = ?", (memory_id,)
        )
        db.execute(
            "DELETE FROM search_feedback WHERE memory_id = ?", (memory_id,)
        )
        # Legacy v3.x structural-vector table, present only in old DBs
        legacy = db.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='memory_vectors'"
        ).fetchone()
        if legacy:
            db.execute(
                "DELETE FROM memory_vectors WHERE memory_id = ?", (memory_id,)
            )
        db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        db.execute(
            "INSERT INTO purge_log (id, memory_id, purged_at) VALUES (?, ?, ?)",
            (uuid7(), memory_id, now),
        )
        db.commit()

    return {
        "id":               memory_id,
        "status":           "purged",
        "versions_removed": versions,
        "purged_at":        now,
    }
