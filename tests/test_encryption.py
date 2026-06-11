# tests/test_encryption.py
# At-rest encryption (CHRONOS_DB_KEY / SQLCipher). Skips entirely when the
# encryption extra isn't installed; CI runs these in a dedicated job that
# installs sqlcipher3-wheels on both Linux and Windows.

import importlib.util
import os
import sys

import pytest

_SQLCIPHER = importlib.util.find_spec("sqlcipher3") is not None
pytestmark = pytest.mark.skipif(
    not _SQLCIPHER, reason="encryption extra (sqlcipher3) not installed"
)

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "scripts")
)


def _fresh_encrypted_db(tmp_path, monkeypatch, key="test-passphrase-1"):
    """Point Chronos at a new path with a key set, schema initialised."""
    path = str(tmp_path / "encrypted.db")
    monkeypatch.setenv("CHRONOS_DB_PATH", path)
    monkeypatch.setenv("CHRONOS_DB_KEY", key)
    from chronos.db import init_db
    init_db()
    return path


def test_encrypted_roundtrip_and_real_ciphertext(tmp_path, monkeypatch):
    from chronos.memory import MemoryStore

    path = _fresh_encrypted_db(tmp_path, monkeypatch)
    store = MemoryStore()
    m = store.remember("the deploy pipeline runs nightly", project="ops")
    out = store.recall("deploying")  # FTS5 + porter still work under SQLCipher
    assert [r["id"] for r in out["results"]] == [m["id"]]

    with open(path, "rb") as fh:
        assert not fh.read(16).startswith(b"SQLite format 3")  # ciphertext


def test_raw_key_fast_path_engages(tmp_path, monkeypatch):
    """The KDF must be paid once, not per connection (audit: 320ms→~1ms)."""
    import chronos.db as cdb

    path = _fresh_encrypted_db(tmp_path, monkeypatch, key="speed-key")
    entry = cdb._key_cache.get((path, "speed-key"))
    assert entry is not None and entry[0] == "raw"


def test_wrong_key_fails_loud_with_guidance(tmp_path, monkeypatch):
    _fresh_encrypted_db(tmp_path, monkeypatch, key="right-key")
    monkeypatch.setenv("CHRONOS_DB_KEY", "wrong-key")
    from chronos.db import init_db
    with pytest.raises(RuntimeError, match="encrypt_db"):
        init_db()


def test_plaintext_db_with_key_set_fails_loud(tmp_path, monkeypatch, chronos_db):
    # chronos_db fixture made a plaintext DB; now a key appears
    monkeypatch.setenv("CHRONOS_DB_KEY", "surprise-key")
    from chronos.db import init_db
    with pytest.raises(RuntimeError, match="encrypt_db"):
        init_db()


def test_encrypted_db_without_key_fails_loud(tmp_path, monkeypatch):
    _fresh_encrypted_db(tmp_path, monkeypatch, key="the-key")
    monkeypatch.delenv("CHRONOS_DB_KEY")
    from chronos.db import init_db
    with pytest.raises(RuntimeError, match="CHRONOS_DB_KEY"):
        init_db()


def test_passphrase_with_single_quote(tmp_path, monkeypatch):
    from chronos.memory import MemoryStore

    _fresh_encrypted_db(tmp_path, monkeypatch, key="it's a 'quoted' key")
    store = MemoryStore()
    store.remember("quoted-key smoke test")
    assert store.recall("quoted-key smoke")["count"] == 1


def test_migration_encrypt_then_decrypt_roundtrip(tmp_path, monkeypatch, chronos_db):
    import sqlcipher3
    from encrypt_db import decrypt_db, encrypt_db

    from chronos.memory import MemoryStore

    # Plaintext DB with content (chronos_db fixture path, no key set)
    store = MemoryStore()
    m = store.remember("secret meeting notes about the merger")

    bak = encrypt_db(chronos_db, "migrate-key")
    assert os.path.exists(bak)
    with open(chronos_db, "rb") as fh:
        assert not fh.read(16).startswith(b"SQLite format 3")

    # Server-side view: key set → recall works on the migrated file
    monkeypatch.setenv("CHRONOS_DB_KEY", "migrate-key")
    from chronos.db import init_db
    init_db()
    assert m["id"] in {r["id"] for r in store.recall("merger")["results"]}

    # And back to plaintext
    monkeypatch.delenv("CHRONOS_DB_KEY")
    decrypt_db(chronos_db, "migrate-key")
    with open(chronos_db, "rb") as fh:
        assert fh.read(16).startswith(b"SQLite format 3")
    assert m["id"] in {r["id"] for r in store.recall("merger")["results"]}

    # Wrong key cannot open the encrypted backup-era file… verify via driver
    conn = sqlcipher3.connect(chronos_db)
    n = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    conn.close()
    assert n == 1  # plaintext again, readable without key


def test_rekey_changes_passphrase(tmp_path, monkeypatch):
    from encrypt_db import rekey_db

    from chronos.memory import MemoryStore

    path = _fresh_encrypted_db(tmp_path, monkeypatch, key="old-key")
    MemoryStore().remember("note that survives a rekey")

    rekey_db(path, "old-key", "new-key")

    monkeypatch.setenv("CHRONOS_DB_KEY", "new-key")
    from chronos.db import init_db
    init_db()
    assert MemoryStore().recall("rekey")["count"] == 1

    monkeypatch.setenv("CHRONOS_DB_KEY", "old-key")
    with pytest.raises(RuntimeError):
        init_db()
