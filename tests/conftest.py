# tests/conftest.py
# Every test gets a fresh, isolated database. db.db_path() resolves the
# env var at call time (v4.0), so monkeypatching the env is sufficient —
# no import-order tricks required.

import pytest


@pytest.fixture(autouse=True)
def chronos_db(tmp_path, monkeypatch):
    """Point every test at a fresh temp database with full schema applied."""
    monkeypatch.setenv("CHRONOS_DB_PATH", str(tmp_path / "test.db"))
    from chronos.db import init_db
    init_db()
    return str(tmp_path / "test.db")
