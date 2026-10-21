"""Health checks. Each failed check is a reason for a non-zero exit."""

from __future__ import annotations

import ast
import time
from dataclasses import dataclass
from pathlib import Path

from ndup.embed.base import EmbeddingProvider


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


@dataclass
class DoctorContext:
    profile: str = "solo"
    database_ok: bool = True
    pgvector_version: str | None = "0.8.0"
    migrations_ok: bool = True
    model_registered: bool = True
    chunks_missing_hash: int = 0
    token_present: bool = True
    lease_held: bool | None = None
    volume_writable: bool | None = None
    gpu_visible: bool | None = None
    expect_gpu: bool = False
    tunnel_connected: bool | None = None
    postgres_tcp: bool | None = None
    backup_age_hours: float | None = None
    false_neighbor_rate: float | None = None
    require_pod_checks: bool = False


def parse_version(value: str) -> tuple[int, int, int]:
    parts = value.split(".")
    numbers = [int(part) for part in parts[:3]]
    while len(numbers) < 3:
        numbers.append(0)
    return numbers[0], numbers[1], numbers[2]


def pgvector_ok(version: str | None) -> bool:
    if version is None:
        return False
    return parse_version(version) >= (0, 8, 0)


def evaluate(ctx: DoctorContext) -> list[Check]:
    checks = [
        Check("database", ctx.database_ok, "connected" if ctx.database_ok else "unreachable"),
        Check(
            "pgvector",
            pgvector_ok(ctx.pgvector_version),
            ctx.pgvector_version or "missing",
        ),
        Check("migrations", ctx.migrations_ok, "current" if ctx.migrations_ok else "pending"),
        Check("model", ctx.model_registered, "registered" if ctx.model_registered else "missing"),
        Check(
            "chunk-hashes",
            ctx.chunks_missing_hash == 0,
            f"{ctx.chunks_missing_hash} chunks missing text_hash or embed_hash",
        ),
        Check("notion-token", ctx.token_present, "present" if ctx.token_present else "missing"),
        Check(
            "false-neighbors",
            ctx.false_neighbor_rate is None or ctx.false_neighbor_rate <= 0.2,
            "n/a" if ctx.false_neighbor_rate is None else f"{ctx.false_neighbor_rate:.3f}",
        ),
    ]
    if ctx.profile == "pod" or ctx.require_pod_checks:
        checks.extend(
            [
                Check("lease", bool(ctx.lease_held), "held" if ctx.lease_held else "not held"),
                Check(
                    "volume",
                    bool(ctx.volume_writable),
                    "writable" if ctx.volume_writable else "not writable",
                ),
                Check(
                    "gpu",
                    (not ctx.expect_gpu) or bool(ctx.gpu_visible),
                    "visible" if ctx.gpu_visible else "missing",
                ),
                Check(
                    "tunnel",
                    bool(ctx.tunnel_connected),
                    "connected" if ctx.tunnel_connected else "down",
                ),
                Check(
                    "postgres-socket",
                    ctx.postgres_tcp is False,
                    "socket only" if ctx.postgres_tcp is False else "tcp listener",
                ),
                Check(
                    "backup-age",
                    ctx.backup_age_hours is not None and ctx.backup_age_hours <= 48,
                    "unknown" if ctx.backup_age_hours is None else f"{ctx.backup_age_hours:.1f} h",
                ),
            ]
        )
    return checks


def failed(checks: list[Check]) -> list[Check]:
    return [check for check in checks if not check.ok]


def provider_imports_http(root: Path | None = None) -> list[str]:
    """AST scan of embed provider modules. They must not import an HTTP client."""
    base = root or Path(__file__).resolve().parent / "embed"
    banned = {"httpx", "requests", "urllib", "aiohttp", "http"}
    offenders: list[str] = []
    for path in sorted(base.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            for name in names:
                if name in banned:
                    offenders.append(f"{path.name}:{name}")
    return offenders


def bench_provider(provider: EmbeddingProvider, texts: int = 32) -> float:
    """Chunks per second. Uses whatever provider is injected; tests pass the fake one."""
    payload = [f"local bench sentence {index} about duplication" for index in range(texts)]
    started = time.perf_counter()
    provider.embed(payload)
    elapsed = max(time.perf_counter() - started, 1e-6)
    return texts / elapsed
