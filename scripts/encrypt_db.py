# scripts/encrypt_db.py
# Migrate a Chronos database between plaintext and SQLCipher encryption.
#
#   python scripts/encrypt_db.py encrypt [--db PATH]   plaintext -> encrypted
#   python scripts/encrypt_db.py decrypt [--db PATH]   encrypted -> plaintext
#   python scripts/encrypt_db.py rekey   [--db PATH]   change the passphrase
#
# The key is read from CHRONOS_DB_KEY, or prompted interactively (getpass —
# never appears in shell history). Requires the encryption extra:
#   pip install "chronosmcp[encryption]"
#
# Safety: the converted database is written to a temp file, row counts are
# verified against the original, the original is kept as <db>.pre-<op>.bak,
# and only then swapped into place. Stale -wal/-shm sidecars are removed
# after the swap. Delete the .bak yourself once you've confirmed the server
# starts — it still holds the pre-migration content.

import argparse
import getpass
import os
import sqlite3
import sys


def _key_from_env_or_prompt(prompt: str) -> str:
    key = os.environ.get("CHRONOS_DB_KEY")
    if key:
        return key
    key = getpass.getpass(prompt)
    if not key:
        print("error: empty passphrase", file=sys.stderr)
        sys.exit(2)
    return key


def _quote(passphrase: str) -> str:
    return passphrase.replace("'", "''")


def _checkpoint_plaintext(path: str) -> None:
    """Fold any WAL content into the main file before export."""
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()


def _count_memories(conn) -> int:
    try:
        return conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    except Exception:
        return -1  # pre-schema or non-chronos DB — counts not comparable


def _swap_into_place(path: str, tmp: str, op: str) -> str:
    bak = f"{path}.pre-{op}.bak"
    os.replace(path, bak)
    os.replace(tmp, path)
    for sidecar in (f"{path}-wal", f"{path}-shm"):
        if os.path.exists(sidecar):
            os.remove(sidecar)
    return bak


def encrypt_db(path: str, key: str) -> str:
    """Plaintext -> SQLCipher. Returns the backup path."""
    import sqlcipher3

    _checkpoint_plaintext(path)
    tmp = f"{path}.enc.tmp"
    if os.path.exists(tmp):
        os.remove(tmp)

    conn = sqlcipher3.connect(path)  # unkeyed sqlcipher reads plaintext
    try:
        n_before = _count_memories(conn)
        conn.execute("ATTACH DATABASE ? AS encrypted KEY ?", (tmp, key))
        conn.execute("SELECT sqlcipher_export('encrypted')")
        conn.execute("DETACH DATABASE encrypted")
    finally:
        conn.close()

    check = sqlcipher3.connect(tmp)
    try:
        check.execute(f"PRAGMA key = '{_quote(key)}'")
        n_after = _count_memories(check)
    finally:
        check.close()
    if n_after != n_before:
        os.remove(tmp)
        raise RuntimeError(
            f"verification failed: {n_before} memories before, {n_after} after"
        )

    return _swap_into_place(path, tmp, "encrypt")


def decrypt_db(path: str, key: str) -> str:
    """SQLCipher -> plaintext. Returns the backup path."""
    import sqlcipher3

    tmp = f"{path}.plain.tmp"
    if os.path.exists(tmp):
        os.remove(tmp)

    conn = sqlcipher3.connect(path)
    try:
        conn.execute(f"PRAGMA key = '{_quote(key)}'")
        n_before = _count_memories(conn)
        conn.execute("ATTACH DATABASE ? AS plaintext KEY ''", (tmp,))
        conn.execute("SELECT sqlcipher_export('plaintext')")
        conn.execute("DETACH DATABASE plaintext")
    finally:
        conn.close()

    check = sqlite3.connect(tmp)
    try:
        n_after = _count_memories(check)
    finally:
        check.close()
    if n_after != n_before:
        os.remove(tmp)
        raise RuntimeError(
            f"verification failed: {n_before} memories before, {n_after} after"
        )

    return _swap_into_place(path, tmp, "decrypt")


def rekey_db(path: str, old_key: str, new_key: str) -> None:
    """Change the passphrase in place (PRAGMA rekey)."""
    import sqlcipher3

    conn = sqlcipher3.connect(path)
    try:
        conn.execute(f"PRAGMA key = '{_quote(old_key)}'")
        conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()  # verify
        conn.execute(f"PRAGMA rekey = '{_quote(new_key)}'")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encrypt, decrypt, or rekey a Chronos database (SQLCipher)."
    )
    parser.add_argument("action", choices=["encrypt", "decrypt", "rekey"])
    parser.add_argument(
        "--db",
        default=os.environ.get(
            "CHRONOS_DB_PATH",
            os.path.join(os.path.dirname(__file__), "..", "chronos.db"),
        ),
        help="database path (default: CHRONOS_DB_PATH or ./chronos.db)",
    )
    args = parser.parse_args()

    try:
        import sqlcipher3  # noqa: F401
    except ImportError:
        print(
            'error: encryption driver missing — pip install "chronosmcp[encryption]"',
            file=sys.stderr,
        )
        sys.exit(2)

    if not os.path.exists(args.db):
        print(f"error: no database at {args.db}", file=sys.stderr)
        sys.exit(2)

    if args.action == "encrypt":
        key = _key_from_env_or_prompt("New passphrase: ")
        bak = encrypt_db(args.db, key)
        print(f"encrypted {args.db}")
        print(f"plaintext backup kept at {bak} — delete it once verified")
        print("start the server with CHRONOS_DB_KEY set to this passphrase")
    elif args.action == "decrypt":
        key = _key_from_env_or_prompt("Current passphrase: ")
        bak = decrypt_db(args.db, key)
        print(f"decrypted {args.db} (backup at {bak})")
        print("unset CHRONOS_DB_KEY before starting the server")
    else:
        old = _key_from_env_or_prompt("Current passphrase: ")
        new = getpass.getpass("New passphrase: ")
        if not new:
            print("error: empty new passphrase", file=sys.stderr)
            sys.exit(2)
        rekey_db(args.db, old, new)
        print(f"rekeyed {args.db} — update CHRONOS_DB_KEY to the new passphrase")


if __name__ == "__main__":
    main()
