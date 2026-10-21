"""Prepare a local Postgres database when ``pg_isready`` sees a server.

Remote URLs and ``memory://`` are left alone. The laptop install is the OS
package (apt or Homebrew). This module does not call Docker.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, unquote, urlparse

from ndup.errors import NdUpError


def _url(user: str, password: str, host: str, port: int, database: str) -> str:
    auth = quote(user, safe="")
    if password:
        auth = f"{auth}:{quote(password, safe='')}"
    return f"postgresql://{auth}@{host}:{port}/{quote(database, safe='')}"


_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
EXTENSION_SQL = "CREATE EXTENSION IF NOT EXISTS vector"

Runner = Callable[[list[str]], int]
Connector = Callable[[str], Any]


@dataclass(frozen=True)
class LocalPostgres:
    user: str
    password: str
    host: str
    port: int
    database: str

    def maintenance_url(self) -> str:
        return _url(self.user, self.password, self.host, self.port, "postgres")

    def database_url(self) -> str:
        return _url(self.user, self.password, self.host, self.port, self.database)


def parse_local_postgres(url: str) -> LocalPostgres | None:
    """Return a localhost target, or None for memory and remote URLs."""
    if not url.startswith(("postgresql://", "postgres://")):
        return None
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if host not in _LOCAL_HOSTS:
        return None
    database = unquote(parsed.path.lstrip("/"))
    if not database:
        return None
    return LocalPostgres(
        user=unquote(parsed.username or "postgres"),
        password=unquote(parsed.password or ""),
        host=host,
        port=parsed.port or 5432,
        database=database,
    )


def pg_isready_argv(target: LocalPostgres) -> list[str]:
    return ["pg_isready", "-h", target.host, "-p", str(target.port)]


def quote_ident(name: str) -> str:
    if not _IDENT.fullmatch(name):
        raise NdUpError(f"Refusing to create a database named {name!r}")
    return f'"{name}"'


def create_database_sql(name: str) -> str:
    return f"CREATE DATABASE {quote_ident(name)}"


def ensure_local_database(
    url: str,
    *,
    which: Callable[[str], str | None] = shutil.which,
    run: Runner | None = None,
    connect: Connector | None = None,
) -> str | None:
    """Create the database and ``vector`` extension when a local server answers.

    Returns None when there is nothing to do: a memory URL, a remote host, or
    ``pg_isready`` is missing or reports the server down. Callers then open
    the URL as usual.
    """
    target = parse_local_postgres(url)
    if target is None or which("pg_isready") is None:
        return None
    argv = pg_isready_argv(target)
    code = run(argv) if run is not None else _default_run(argv)
    if code != 0:
        return None
    opener = connect or _psycopg_connect
    try:
        created = _create_database(opener, target)
        with opener(target.database_url()) as database:
            database.execute(EXTENSION_SQL)
    except NdUpError:
        raise
    except Exception as exc:
        message = str(exc)
        if target.password:
            message = message.replace(target.password, "***")
        raise NdUpError(f"Could not prepare the local database {target.database}: {message}") from exc
    if created:
        return f"Created database {target.database} and the vector extension."
    return f"Database {target.database} is ready and the vector extension is installed."


def _create_database(opener: Connector, target: LocalPostgres) -> bool:
    with opener(target.maintenance_url()) as admin:
        row = admin.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (target.database,),
        ).fetchone()
        if row is not None:
            return False
        admin.execute(create_database_sql(target.database))
    return True


def _default_run(argv: list[str]) -> int:
    completed = subprocess.run(argv, check=False, capture_output=True)
    return int(completed.returncode)


def _psycopg_connect(dsn: str) -> Any:
    import psycopg

    return psycopg.connect(dsn, autocommit=True)
