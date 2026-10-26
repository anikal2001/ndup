"""Forward-only SQL migrations."""

from __future__ import annotations

from pathlib import Path

import psycopg
from psycopg import Connection
from psycopg.rows import dict_row

from ndup.db.local import EXTENSION_SQL
from ndup.errors import NdUpError

MIGRATION_DIR = Path(__file__).resolve().parent / "migrations"
ROLES_SQL = Path(__file__).resolve().parent / "roles.sql"


def connect(database_url: str) -> Connection[dict[str, object]]:
    try:
        from pgvector.psycopg import register_vector
    except ImportError as exc:
        raise NdUpError("pgvector is required to connect") from exc
    conn = psycopg.connect(database_url, row_factory=dict_row)
    try:
        register_vector(conn)
    except psycopg.ProgrammingError:
        # Fresh database: 001_schema.sql creates the extension, but the type
        # adapter has to exist before any migration can run.
        conn.execute(EXTENSION_SQL)
        conn.commit()
        register_vector(conn)
    return conn


def applied_versions(conn: Connection[dict[str, object]]) -> list[str]:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migration (
          version text PRIMARY KEY,
          applied_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    rows = conn.execute("SELECT version FROM schema_migration ORDER BY version").fetchall()
    conn.commit()
    return [str(row["version"]) for row in rows]


def migrate(conn: Connection[dict[str, object]], *, with_roles: bool = True) -> list[str]:
    """Apply any migrations that are not recorded yet. Returns newly applied names."""
    done = set(applied_versions(conn))
    fresh: list[str] = []
    for path in sorted(MIGRATION_DIR.glob("*.sql")):
        if path.name in done:
            continue
        sql = path.read_text(encoding="utf-8")
        with conn.transaction():
            conn.execute(sql)
            conn.execute(
                "INSERT INTO schema_migration (version) VALUES (%s)",
                (path.name,),
            )
        fresh.append(path.name)
    if with_roles and ROLES_SQL.is_file():
        conn.execute(ROLES_SQL.read_text(encoding="utf-8"))
        conn.commit()
    return fresh


def migration_status(conn: Connection[dict[str, object]]) -> list[tuple[str, bool]]:
    done = set(applied_versions(conn))
    return [(path.name, path.name in done) for path in sorted(MIGRATION_DIR.glob("*.sql"))]
