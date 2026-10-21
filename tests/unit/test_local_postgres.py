"""Native Postgres bootstrap. No Docker and no live server."""

from __future__ import annotations

from pathlib import Path

import pytest

from ndup.config import load_settings, write_settings
from ndup.db.local import (
    create_database_sql,
    ensure_local_database,
    parse_local_postgres,
    pg_isready_argv,
)
from ndup.errors import NdUpError


class _Result:
    def __init__(self, row: object) -> None:
        self._row = row

    def fetchone(self) -> object:
        return self._row


class _Conn:
    def __init__(self, sink: list[tuple[str, object]], select_row: object) -> None:
        self.sink = sink
        self.select_row = select_row

    def execute(self, sql: str, params: object = None) -> _Result:
        self.sink.append((sql, params))
        if sql.startswith("SELECT"):
            return _Result(self.select_row)
        return _Result(None)

    def __enter__(self) -> _Conn:
        return self

    def __exit__(self, *args: object) -> bool:
        return False


@pytest.mark.unit
def test_pg_isready_argv_omits_password() -> None:
    target = parse_local_postgres("postgresql://ndup:secret@localhost:5432/ndup")
    assert target is not None
    argv = pg_isready_argv(target)
    assert argv == ["pg_isready", "-h", "localhost", "-p", "5432"]
    assert "secret" not in " ".join(argv)
    assert create_database_sql("ndup") == 'CREATE DATABASE "ndup"'


@pytest.mark.unit
def test_ensure_creates_database_and_extension() -> None:
    sink: list[tuple[str, object]] = []
    seen: list[list[str]] = []

    def connect(dsn: str) -> _Conn:
        del dsn
        return _Conn(sink, None)

    message = ensure_local_database(
        "postgresql://ndup:secret@127.0.0.1:5432/ndup",
        which=lambda _name: "/usr/bin/pg_isready",
        run=lambda argv: seen.append(argv) or 0,
        connect=connect,
    )
    assert message == "Created database ndup and the vector extension."
    assert "secret" not in (message or "")
    assert seen == [["pg_isready", "-h", "127.0.0.1", "-p", "5432"]]
    sql = [item[0] for item in sink]
    assert 'CREATE DATABASE "ndup"' in sql
    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql


@pytest.mark.unit
def test_ensure_skips_create_when_database_exists() -> None:
    sink: list[tuple[str, object]] = []

    def connect(dsn: str) -> _Conn:
        del dsn
        return _Conn(sink, (1,))

    message = ensure_local_database(
        "postgresql://ndup:secret@localhost:5432/ndup",
        which=lambda _name: "/usr/bin/pg_isready",
        run=lambda _argv: 0,
        connect=connect,
    )
    assert message is not None
    assert "ready" in message
    assert not any(sql.startswith("CREATE DATABASE") for sql, _params in sink)
    assert any(sql.startswith("CREATE EXTENSION") for sql, _params in sink)


@pytest.mark.unit
def test_ensure_skips_memory_remote_and_down_server() -> None:
    def explode(_name: str) -> str:
        raise AssertionError("pg_isready should not be consulted")

    assert ensure_local_database("memory://.ndup-store", which=explode) is None
    assert ensure_local_database("postgresql://ndup:ndup@db.internal:5432/ndup", which=explode) is None
    called = {"connect": 0}

    def connect(_dsn: str) -> _Conn:
        called["connect"] += 1
        return _Conn([], None)

    assert (
        ensure_local_database(
            "postgresql://ndup:ndup@localhost:5432/ndup",
            which=lambda _name: None,
            connect=connect,
        )
        is None
    )
    assert (
        ensure_local_database(
            "postgresql://ndup:ndup@localhost:5432/ndup",
            which=lambda _name: "/usr/bin/pg_isready",
            run=lambda _argv: 2,
            connect=connect,
        )
        is None
    )
    assert called["connect"] == 0


@pytest.mark.unit
def test_database_name_must_be_an_identifier() -> None:
    with pytest.raises(NdUpError):
        ensure_local_database(
            "postgresql://ndup:ndup@localhost:5432/bad-name",
            which=lambda _name: "/usr/bin/pg_isready",
            run=lambda _argv: 0,
            connect=lambda _dsn: _Conn([], None),
        )


@pytest.mark.unit
def test_database_url_env_overrides_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "config.toml"
    write_settings(path, load_settings(profile="solo"))
    monkeypatch.setenv("NDUP_STORAGE__DATABASE_URL", "postgresql://localhost:5432/nested")
    monkeypatch.setenv("NDUP_DATABASE_URL", "postgresql://localhost:5432/fromenv")
    loaded = load_settings(path)
    assert loaded.storage.database_url == "postgresql://localhost:5432/fromenv"


@pytest.mark.unit
def test_readme_local_setup_has_no_docker_or_runpod_steps() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    quickstart = Path("docs/quickstart.md").read_text(encoding="utf-8")
    local = readme.split("## Local setup", 1)[1].split("\n## ", 1)[0]
    quick_local = quickstart.split("## Postgres", 1)[0]
    commands = [
        "uv run ndup init --database-url memory://.ndup-store --model fake-bow-v1",
        "uv run ndup sync",
        "uv run ndup embed",
        "uv run ndup analyze --full",
        "uv run ndup show clusters",
        "uv run ndup run",
        "https://docs.astral.sh/uv/getting-started/installation/",
    ]
    for command in commands:
        assert command in local
        assert command in quick_local
    assert "main" in local
    assert "git checkout" not in local
    assert "docker compose" not in local.lower()
    assert "ndup runpod" not in local
    assert "apt install postgresql" in quickstart
    assert "brew install postgresql@17 pgvector" in quickstart
    assert "NDUP_DATABASE_URL" in quickstart
    assert "pg_isready" in quickstart
    assert "docker compose" not in quickstart.lower()
