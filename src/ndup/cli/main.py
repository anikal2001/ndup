"""ndup command line. Phase 2 commands exit 2 without starting a server."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any
from uuid import UUID

import typer
from rich.console import Console
from rich.table import Table

from ndup import __version__
from ndup.analysis.calibrate import fit_thresholds
from ndup.analysis.relations import EvidenceView
from ndup.config import Settings, load_settings, redacted_dump, write_settings
from ndup.db.local import ensure_local_database
from ndup.doctor import DoctorContext, bench_provider, evaluate, failed, provider_imports_http
from ndup.embed.registry import build_provider
from ndup.errors import NdUpError
from ndup.model import order_pair
from ndup.notion.estimate import Estimate, build_estimate, format_estimate
from ndup.runpod import (
    RunPodClient,
    RunPodError,
    bring_up,
    connect_account,
    format_report,
    pod_status,
    stop_pod,
)
from ndup.secrets import install_redaction, known_secret_values, resolve_secret, rotate_notion_token
from ndup.service import (
    analyze_store,
    embed_store,
    ensure_model,
    open_store,
    sync_store,
    utcnow,
)
from ndup.tui.tables import (
    cluster_payload,
    clusters_payload,
    doc_payload,
    export_metadata,
    pair_payload,
)

app = typer.Typer(no_args_is_help=True, help="Notion semantic duplication detector.")
config_app = typer.Typer(no_args_is_help=True, help="Show the resolved configuration.")
db_app = typer.Typer(no_args_is_help=True, help="Database lifecycle.")
sync_app = typer.Typer(help="Read pages from Notion.")
show_app = typer.Typer(no_args_is_help=True, help="Inspect clusters, documents, and pairs.")
ignore_app = typer.Typer(no_args_is_help=True, help="Ignore a pair, document, or root.")
secrets_app = typer.Typer(no_args_is_help=True, help="Rotate secrets. Values are never printed.")
runpod_app = typer.Typer(
    no_args_is_help=True,
    help="Secure Cloud volume and pod. The laptop calls the API and does not build an image.",
)
console = Console()
err_console = Console(stderr=True)

PHASE2 = "This command is Phase 2 (web app, workers, SSO). It is not in this build."


def compose_argv(action: str, *, solo: bool) -> list[str]:
    filename = "docker-compose.solo.yml" if solo else "docker-compose.yml"
    if action == "up":
        return ["docker", "compose", "-f", filename, "up", "-d"]
    if action == "down":
        return ["docker", "compose", "-f", filename, "down"]
    raise NdUpError(f"Unknown compose action {action}")


class _State:
    def __init__(self, settings: Settings, config_path: Path | None = None) -> None:
        self.settings = settings
        self.config_path = config_path


@app.callback()
def main(
    ctx: typer.Context,
    config: Path | None = typer.Option(None, "--config", envvar="NDUP_CONFIG"),
    profile: str | None = typer.Option(None, "--profile"),
) -> None:
    """Local embeddings only. Nothing is written back to Notion."""
    if ctx.resilient_parsing:
        return
    chosen = profile if profile in {"solo", "team", "pod"} else None
    try:
        settings = load_settings(config, profile=chosen)  # type: ignore[arg-type]
    except NdUpError as exc:
        err_console.print(str(exc))
        raise typer.Exit(code=1) from exc
    install_redaction(known_secret_values())
    ctx.obj = _State(settings, config)


def _state(ctx: typer.Context) -> _State:
    if not isinstance(ctx.obj, _State):
        raise typer.Exit(code=1)
    return ctx.obj


def _open(settings: Settings, *, migrate: bool = True) -> Any:
    try:
        store = open_store(settings.storage.database_url)
        if migrate:
            store.migrate()
    except NdUpError as exc:
        err_console.print(str(exc))
        raise typer.Exit(code=1) from exc
    return store


def _token() -> str:
    token = resolve_secret("notion_token")
    if not token:
        err_console.print("Set NDUP_NOTION_TOKEN or /run/secrets/notion_token.")
        raise typer.Exit(code=1)
    return token


def _print_json(payload: object) -> None:
    console.print(json.dumps(payload, indent=2, default=str))


@config_app.command("show")
def config_show(
    ctx: typer.Context,
    redacted: bool = typer.Option(True, "--redacted/--raw"),
) -> None:
    """Print the resolved config. Secrets and database passwords are redacted."""
    settings = _state(ctx).settings
    payload = redacted_dump(settings, known_secret_values()) if redacted else settings.model_dump(mode="json")
    _print_json(payload)


@app.command("init")
def init_cmd(
    ctx: typer.Context,
    database_url: str | None = typer.Option(None, "--database-url"),
    model: str | None = typer.Option(None, "--model"),
    config_out: Path | None = typer.Option(None, "--config-out"),
) -> None:
    """Write config, migrate, register the local model, and print a sync estimate."""
    settings = _state(ctx).settings
    if database_url:
        settings.storage.database_url = database_url
    if model:
        settings.embedding.model = model
    destination = config_out or _default_config_path()
    write_settings(destination, settings)
    console.print(f"Wrote {destination}")
    try:
        note = ensure_local_database(settings.storage.database_url)
    except NdUpError as exc:
        err_console.print(str(exc))
        raise typer.Exit(code=1) from exc
    if note:
        console.print(note)
    store = _open(settings)
    try:
        ensure_model(store, settings)
        token = resolve_secret("notion_token")
        if token:
            estimate = asyncio.run(_live_estimate(token, settings.notion.api_version))
        else:
            estimate = build_estimate(
                page_requests=[],
                row_requests=[],
                empty_flags=[],
                token_counts=[],
                pages=0,
                rows=0,
            )
            console.print("No Notion token yet. Estimate uses the default 3–6 requests/page prior.")
        store.set_state("estimate", estimate.to_json())
        console.print(format_estimate(estimate))
    finally:
        store.close()


@sync_app.callback(invoke_without_command=True)
def sync_cmd(
    ctx: typer.Context,
    estimate: bool = typer.Option(False, "--estimate"),
    discover_only: bool = typer.Option(False, "--discover-only"),
    root: str | None = typer.Option(None, "--root"),
) -> None:
    """Discover pages and fetch them. Search is a hint; the block tree is ground truth."""
    if ctx.invoked_subcommand is not None:
        return
    settings = _state(ctx).settings
    if estimate:
        _print_estimate(settings)
        return
    root_id = UUID(root) if root else None
    store = _open(settings)
    try:
        counts = asyncio.run(
            sync_store(
                store,
                settings,
                _token(),
                discover_only=discover_only,
                root=root_id,
            )
        )
        _print_json(counts)
    finally:
        store.close()


@sync_app.command("status")
def sync_status(ctx: typer.Context) -> None:
    """Queue depth by state."""
    settings = _state(ctx).settings
    store = _open(settings, migrate=False)
    try:
        payload = {"tasks": store.task_counts(), "estimate": store.get_state("estimate")}
        _print_json(payload)
    finally:
        store.close()


@app.command("embed")
def embed_cmd(ctx: typer.Context) -> None:
    """Embed new chunk hashes with the local model. Cached hashes are skipped."""
    settings = _state(ctx).settings
    store = _open(settings)
    try:
        embedded, hits = embed_store(store, settings)
        _print_json({"embedded": embedded, "cache_hits": hits})
    finally:
        store.close()


@app.command("analyze")
def analyze_cmd(
    ctx: typer.Context,
    full: bool = typer.Option(True, "--full/--incremental"),
) -> None:
    """Cluster near-copies. Incremental analysis is Phase 2; --full is the Phase 1 path."""
    if not full:
        err_console.print("Incremental analysis is Phase 2. Re-run with --full.")
        raise typer.Exit(code=2)
    settings = _state(ctx).settings
    store = _open(settings)
    try:
        result = analyze_store(store, settings)
        _print_json(
            {
                "pairs": len(result.pairs),
                "clusters": len(result.clusters),
                "other_language_docs": result.other_language_docs,
                "stub_docs": result.stub_docs,
            }
        )
    finally:
        store.close()


@app.command("run")
def run_cmd(ctx: typer.Context) -> None:
    """Sync, embed, and analyze once."""
    settings = _state(ctx).settings
    store = _open(settings)
    try:
        counts = asyncio.run(sync_store(store, settings, _token()))
        embedded, hits = embed_store(store, settings)
        result = analyze_store(store, settings)
        _print_json(
            {
                "sync": counts,
                "embedded": embedded,
                "cache_hits": hits,
                "clusters": len(result.clusters),
            }
        )
    finally:
        store.close()


def _docs(store: Any) -> dict[UUID, Any]:
    return {doc.id: doc for doc in store.list_documents()}


def _clusters(store: Any, area: str | None, sort: str, limit: int) -> list[Any]:
    clusters: list[Any] = list(store.list_clusters())
    if area:
        clusters = [cluster for cluster in clusters if area in cluster.areas]
    if sort == "active":
        clusters.sort(key=lambda cluster: (-cluster.active_copies, -cluster.size))
    elif sort == "newest":
        clusters.sort(key=lambda cluster: str(cluster.newest_doc or ""))
    else:
        clusters.sort(key=lambda cluster: (-cluster.size, str(cluster.stable_key)))
    return clusters[:limit]


@show_app.command("clusters")
def show_clusters(
    ctx: typer.Context,
    as_json: bool = typer.Option(False, "--json"),
    sort: str = typer.Option("size", "--sort"),
    area: str | None = typer.Option(None, "--area"),
    limit: int = typer.Option(50, "--limit"),
) -> None:
    settings = _state(ctx).settings
    store = _open(settings, migrate=False)
    try:
        docs = _docs(store)
        chosen = _clusters(store, area, sort, limit)
        other = sum(1 for doc in docs.values() if doc.lang not in {None, "en"} and doc.status == "active")
        stubs = sum(1 for doc in docs.values() if doc.is_stub)
        groups = _stub_groups(list(docs.values()))
        payload = clusters_payload(
            chosen,
            docs,
            mode=settings.web.content_mode,
            stub_groups=groups,
            other_language_docs=other,
            stub_docs=stubs,
        )
        if as_json:
            _print_json(payload)
            return
        table = Table(title="clusters")
        table.add_column("key")
        table.add_column("label")
        table.add_column("size", justify="right")
        table.add_column("areas")
        for cluster in payload["clusters"]:
            table.add_row(
                cluster["stable_key"],
                cluster["label"] or "",
                str(cluster["size"]),
                ", ".join(cluster["areas"]),
            )
        console.print(table)
    finally:
        store.close()


@show_app.command("cluster")
def show_cluster(
    ctx: typer.Context,
    key: str,
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    settings = _state(ctx).settings
    store = _open(settings, migrate=False)
    try:
        cluster = store.get_cluster(UUID(key))
        if cluster is None:
            err_console.print("Cluster not found")
            raise typer.Exit(code=1)
        payload = cluster_payload(cluster, _docs(store), mode=settings.web.content_mode)
        _print_json(payload) if as_json else console.print(payload["label"], payload["size"])
    finally:
        store.close()


@show_app.command("doc")
def show_doc(
    ctx: typer.Context,
    doc_id: str,
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    settings = _state(ctx).settings
    store = _open(settings, migrate=False)
    try:
        doc = store.get_document(UUID(doc_id))
        if doc is None:
            err_console.print("Document not found")
            raise typer.Exit(code=1)
        payload = doc_payload(doc, mode=settings.web.content_mode)
        if as_json:
            _print_json(payload)
        else:
            console.print(payload["title"] or "(untitled)")
    finally:
        store.close()


@show_app.command("pair")
def show_pair(
    ctx: typer.Context,
    doc_a: str,
    doc_b: str,
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    settings = _state(ctx).settings
    store = _open(settings, migrate=False)
    try:
        left, right = order_pair(UUID(doc_a), UUID(doc_b))
        pair = next((item for item in store.list_pairs() if item.doc_a == left and item.doc_b == right), None)
        if pair is None:
            err_console.print("Pair not found")
            raise typer.Exit(code=1)
        payload = pair_payload(pair, _docs(store), mode=settings.web.content_mode)
        _print_json(payload) if as_json else console.print(payload["relation"], payload["score"])
    finally:
        store.close()


@app.command("review")
def review_cmd(
    ctx: typer.Context,
    key: str,
    status: str = typer.Option("reviewed", "--status"),
    note: str | None = typer.Option(None, "--note"),
) -> None:
    settings = _state(ctx).settings
    store = _open(settings, migrate=False)
    try:
        store.review(UUID(key), status, note, os.environ.get("USER"), utcnow())
        console.print(status)
    finally:
        store.close()


@ignore_app.command("pair")
def ignore_pair(ctx: typer.Context, doc_a: str, doc_b: str) -> None:
    _ignore(ctx, "pair", UUID(doc_a), UUID(doc_b))


@ignore_app.command("doc")
def ignore_doc(ctx: typer.Context, doc_id: str) -> None:
    _ignore(ctx, "doc", UUID(doc_id), None)


@ignore_app.command("root")
def ignore_root(ctx: typer.Context, root_id: str) -> None:
    _ignore(ctx, "root", UUID(root_id), None)


def _ignore(ctx: typer.Context, kind: str, left: UUID, right: UUID | None) -> None:
    ordered = order_pair(left, right) if right is not None else (left, None)
    store = _open(_state(ctx).settings, migrate=False)
    try:
        store.ignore(kind, ordered[0], ordered[1], os.environ.get("USER"), utcnow())
    finally:
        store.close()


@app.command("export")
def export_cmd(
    ctx: typer.Context,
    metadata_only: bool = typer.Option(False, "--metadata-only"),
) -> None:
    """Export cluster metadata. Team profile allows only metadata."""
    settings = _state(ctx).settings
    if settings.profile == "team" and not metadata_only:
        err_console.print("Team profile exports metadata only. Pass --metadata-only.")
        raise typer.Exit(code=1)
    store = _open(settings, migrate=False)
    try:
        _print_json(export_metadata(store.list_clusters(), _docs(store)))
    finally:
        store.close()


@app.command("purge")
def purge_cmd(
    ctx: typer.Context,
    doc: str = typer.Option(..., "--doc"),
) -> None:
    """Forget one document. Text, vectors, and pair rows go in one transaction."""
    from ndup.db.purge import purge_document

    settings = _state(ctx).settings
    store = _open(settings, migrate=False)
    try:
        purge_document(store, UUID(doc), utcnow())
        console.print("purged")
    finally:
        store.close()


@app.command("calibrate")
def calibrate_cmd(
    ctx: typer.Context,
    fit: bool = typer.Option(False, "--fit"),
    doc_a: str | None = typer.Option(None, "--doc-a"),
    doc_b: str | None = typer.Option(None, "--doc-b"),
    label: str | None = typer.Option(None, "--label"),
) -> None:
    """Record a label or refit thresholds from stored labels."""
    settings = _state(ctx).settings
    store = _open(settings, migrate=False)
    try:
        if doc_a and doc_b and label:
            left, right = order_pair(UUID(doc_a), UUID(doc_b))
            store.add_label(left, right, label, os.environ.get("USER"), utcnow())
            console.print("recorded")
            return
        labels = store.list_labels()
        if not fit:
            console.print(f"{len(labels)} labels stored. Pass --fit to refit thresholds.")
            return
        if len(labels) < 2:
            err_console.print("Need at least two labels to fit. Shipped thresholds stay in place.")
            raise typer.Exit(code=1)
        examples = _examples_from_labels(store, labels)
        fitted = fit_thresholds(examples)
        model = store.active_model()
        if model is not None:
            store.register_model(model.model, model.dims, model.runtime, fitted.to_json())
        _print_json(fitted.to_json())
    finally:
        store.close()


@secrets_app.command("rotate")
def secrets_rotate(
    name: str = typer.Argument(...),
    token: str = typer.Option(..., "--token", prompt=False),
) -> None:
    """Validate and store a replacement secret. The value is not logged."""
    if name != "notion":
        err_console.print("Only Notion token rotation is implemented.")
        raise typer.Exit(code=1)

    def validate(value: str) -> bool:
        return bool(value.strip())

    try:
        path = rotate_notion_token(token, validate)
    except NdUpError as exc:
        err_console.print(str(exc))
        raise typer.Exit(code=1) from exc
    console.print(f"Stored notion token at {path}")


@app.command("models")
def models_cmd(ctx: typer.Context) -> None:
    store = _open(_state(ctx).settings, migrate=False)
    try:
        _print_json(
            [
                {
                    "id": model.id,
                    "model": model.model,
                    "dims": model.dims,
                    "runtime": model.runtime,
                    "is_active": model.is_active,
                    "thresholds": model.thresholds,
                }
                for model in store.list_models()
            ]
        )
    finally:
        store.close()


@app.command("doctor")
def doctor_cmd(
    ctx: typer.Context,
    bench: bool = typer.Option(False, "--bench"),
) -> None:
    """Exit non-zero when a required check fails."""
    settings = _state(ctx).settings
    offenders = provider_imports_http()
    context = DoctorContext(profile=settings.profile, token_present=resolve_secret("notion_token") is not None)
    store = None
    try:
        store = open_store(settings.storage.database_url)
        store.migrate()
        context.database_ok = True
        version = getattr(store, "pgvector_version", lambda: "0.8.0")()
        context.pgvector_version = version or "0.8.0"
        context.migrations_ok = True
        context.model_registered = store.active_model() is not None or settings.embedding.model != ""
        context.chunks_missing_hash = int(getattr(store, "chunks_missing_hash", lambda: 0)())
    except Exception as exc:
        context.database_ok = False
        context.migrations_ok = False
        context.pgvector_version = None
        err_console.print(str(exc))
    finally:
        if store is not None:
            store.close()
    if settings.profile == "pod":
        context.require_pod_checks = True
        context.lease_held = False
        context.volume_writable = False
        context.tunnel_connected = False
        context.postgres_tcp = True
        context.backup_age_hours = None
    checks = evaluate(context)
    if offenders:
        from ndup.doctor import Check

        checks.append(Check("embed-imports", False, ", ".join(offenders)))
    for check in checks:
        mark = "ok" if check.ok else "FAIL"
        console.print(f"{mark:4} {check.name}: {check.detail}")
    if bench:
        try:
            provider = build_provider(settings.embedding.model, settings.embedding.device)
            rate = bench_provider(provider)
            console.print(f"bench {rate:.1f} chunks/s ({provider.model})")
        except NdUpError as exc:
            err_console.print(str(exc))
            raise typer.Exit(code=1) from exc
    if failed(checks):
        raise typer.Exit(code=1)


@db_app.command("up")
def db_up(solo: bool = typer.Option(True, "--solo/--team")) -> None:
    _compose("up", solo=solo)


@db_app.command("down")
def db_down(solo: bool = typer.Option(True, "--solo/--team")) -> None:
    _compose("down", solo=solo)


@db_app.command("migrate")
def db_migrate(ctx: typer.Context) -> None:
    store = _open(_state(ctx).settings)
    try:
        applied = store.migrate()
        _print_json({"applied": applied or []})
    finally:
        store.close()


@db_app.command("status")
def db_status(ctx: typer.Context) -> None:
    store = _open(_state(ctx).settings, migrate=False)
    try:
        status = store.migration_status() if hasattr(store, "migration_status") else []
        _print_json({"migrations": status, "tasks": store.task_counts()})
    finally:
        store.close()


@db_app.command("psql")
def db_psql(ctx: typer.Context) -> None:
    url = _state(ctx).settings.storage.database_url
    if url.startswith("memory:"):
        err_console.print("The memory store has no psql shell.")
        raise typer.Exit(code=1)
    program = shutil.which("psql")
    if program is None:
        err_console.print(f"psql is not installed. DATABASE_URL is configured for {url.split('@')[-1]}")
        raise typer.Exit(code=1)
    raise typer.Exit(subprocess.call([program, url]))


@db_app.command("backup")
def db_backup(
    ctx: typer.Context,
    state_only: bool = typer.Option(False, "--state-only"),
    full: bool = typer.Option(False, "--full"),
    output: Path = typer.Argument(Path("ndup-backup.json")),
) -> None:
    if state_only == full:
        err_console.print("Pass exactly one of --state-only or --full.")
        raise typer.Exit(code=1)
    store = _open(_state(ctx).settings, migrate=False)
    try:
        payload = store.dump(state_only=state_only)
    finally:
        store.close()
    output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    console.print(str(output))


@db_app.command("reindex")
def db_reindex(ctx: typer.Context) -> None:
    store = _open(_state(ctx).settings, migrate=False)
    try:
        count = store.reindex() if hasattr(store, "reindex") else 0
        _print_json({"models": count})
    finally:
        store.close()


@app.command("web")
def web_cmd() -> None:
    """Phase 2. The report server is not in this build."""
    err_console.print(PHASE2)
    raise typer.Exit(code=2)


@app.command("worker")
def worker_cmd(role: str = typer.Option("all", "--role")) -> None:
    """Phase 2. Scheduled workers are not in this build."""
    del role
    err_console.print(PHASE2)
    raise typer.Exit(code=2)


@app.command("version")
def version_cmd() -> None:
    console.print(__version__)


def _compose(action: str, *, solo: bool) -> None:
    argv = compose_argv(action, solo=solo)
    if shutil.which("docker") is None:
        err_console.print(
            "Postgres is selected with NDUP_DATABASE_URL. Compose is optional. Command: " + " ".join(argv)
        )
        raise typer.Exit(code=1)
    raise typer.Exit(subprocess.call(argv))


def _default_config_path() -> Path:
    override = os.environ.get("NDUP_CONFIG")
    if override:
        return Path(override)
    return Path.home() / ".config" / "ndup" / "config.toml"


def _print_estimate(settings: Settings) -> None:
    token = resolve_secret("notion_token")
    if token:
        estimate = asyncio.run(_live_estimate(token, settings.notion.api_version))
    else:
        estimate = build_estimate(
            page_requests=[],
            row_requests=[],
            empty_flags=[],
            token_counts=[],
            pages=0,
            rows=0,
        )
    console.print(format_estimate(estimate))


async def _live_estimate(token: str, api_version: str) -> Estimate:
    from ndup.notion.client import NotionClient

    async with NotionClient(token, api_version=api_version) as client:
        await client.me()
        pages = 0
        async for _page in client.search_pages():
            pages += 1
            if pages >= 20:
                break
        return build_estimate(
            page_requests=[4] * min(pages, 5),
            row_requests=[1],
            empty_flags=[False] * min(pages, 5),
            token_counts=[200] * min(pages, 5),
            pages=pages,
            rows=0,
        )


def _stub_groups(docs: list[Any]) -> list[list[UUID]]:
    grouped: dict[bytes, list[UUID]] = {}
    for doc in docs:
        if doc.is_stub and doc.title_hash and doc.status == "active":
            grouped.setdefault(doc.title_hash, []).append(doc.id)
    return [members for members in grouped.values() if len(members) >= 2]


def _examples_from_labels(store: Any, labels: list[tuple[UUID, UUID, str]]) -> list[Any]:
    from ndup.analysis.calibrate import LabeledExample

    by_pair = {(pair.doc_a, pair.doc_b): pair for pair in store.list_pairs()}
    examples = []
    for doc_a, doc_b, label in labels:
        pair = by_pair.get((doc_a, doc_b))
        if pair is None:
            continue
        examples.append(
            LabeledExample(
                evidence=EvidenceView(
                    cov_ab=pair.cov_ab,
                    cov_ba=pair.cov_ba,
                    lex_jaccard=pair.lex_jaccard,
                    lex_cont_ab=pair.lex_cont_ab,
                    lex_cont_ba=pair.lex_cont_ba,
                    doc_sim=pair.doc_sim,
                    median_match=pair.median_match,
                    shared_hash=pair.lex_jaccard >= 0.8,
                    weighted_chunks_a=2,
                    weighted_chunks_b=2,
                ),
                label=label,
                doc_sim=pair.doc_sim,
                mutual=True,
                shared_hash=pair.lex_jaccard >= 0.8,
            )
        )
    return examples


def _runpod_key() -> str:
    key = resolve_secret("runpod_api_key")
    if not key:
        err_console.print("Set RUNPOD_API_KEY or run ndup runpod connect.")
        raise typer.Exit(code=1)
    return key


def _save_pod(ctx: typer.Context, settings: Settings, volume_id: str, pod_id: str) -> None:
    settings.deploy.target = "pod"
    settings.deploy.pod.volume_id = volume_id
    settings.deploy.pod.pod_id = pod_id
    path = _state(ctx).config_path or _default_config_path()
    write_settings(path, settings)


@runpod_app.command("connect")
def runpod_connect(
    api_key: str | None = typer.Option(None, "--api-key", envvar="RUNPOD_API_KEY"),
) -> None:
    """Store the API key and check the account. The key is never printed."""
    if not api_key:
        err_console.print("Set RUNPOD_API_KEY. It is stored in the OS keychain.")
        raise typer.Exit(code=1)
    try:
        where = connect_account(api_key)
    except RunPodError as exc:
        err_console.print(str(exc))
        raise typer.Exit(code=1) from exc
    console.print(f"RunPod account ok. Key stored in {where}.")


@runpod_app.command("up")
def runpod_up(
    ctx: typer.Context,
    cpu: bool = typer.Option(False, "--cpu/--gpu"),
) -> None:
    """Create or reuse a Secure Cloud volume and pod, start it, and print the URL."""
    settings = _state(ctx).settings
    key = _runpod_key()
    try:
        with RunPodClient(key) as client:
            report = bring_up(client, settings, cpu=cpu)
    except RunPodError as exc:
        err_console.print(str(exc))
        raise typer.Exit(code=1) from exc
    _save_pod(ctx, settings, report.volume_id, report.pod_id)
    console.print(format_report(report))


@runpod_app.command("status")
def runpod_status(ctx: typer.Context) -> None:
    """Show the recorded pod."""
    settings = _state(ctx).settings
    key = _runpod_key()
    try:
        with RunPodClient(key) as client:
            report = pod_status(client, settings)
    except RunPodError as exc:
        err_console.print(str(exc))
        raise typer.Exit(code=1) from exc
    console.print(format_report(report))


@runpod_app.command("down")
def runpod_down(ctx: typer.Context) -> None:
    """Stop the recorded pod. The network volume is kept."""
    settings = _state(ctx).settings
    key = _runpod_key()
    try:
        with RunPodClient(key) as client:
            report = stop_pod(client, settings)
    except RunPodError as exc:
        err_console.print(str(exc))
        raise typer.Exit(code=1) from exc
    console.print(format_report(report))


app.add_typer(config_app, name="config")
app.add_typer(db_app, name="db")
app.add_typer(sync_app, name="sync")
app.add_typer(show_app, name="show")
app.add_typer(ignore_app, name="ignore")
app.add_typer(secrets_app, name="secrets")
app.add_typer(runpod_app, name="runpod")


if __name__ == "__main__":
    app()
