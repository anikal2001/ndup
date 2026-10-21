"""Discover and fetch. Search is a hint; the block tree is ground truth."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, Protocol
from uuid import UUID

from ndup.db.queue import priority_for
from ndup.model import DataSourceRow, DocumentDraft, RootRow, SyncTask
from ndup.notion.client import NotionClient, NotionError
from ndup.notion.discovery.ancestry import is_excluded, parent_of
from ndup.notion.discovery.data_sources import data_sources_of
from ndup.notion.discovery.search import edited_at, past_watermark
from ndup.notion.discovery.tree import child_targets
from ndup.notion.fetcher import fetch_document
from ndup.notion.render import page_title
from ndup.text.boilerplate import template_title


class SyncStore(Protocol):
    def get_state(self, key: str) -> dict[str, Any] | None: ...
    def set_state(self, key: str, value: dict[str, Any]) -> None: ...
    def upsert_root(self, root: RootRow) -> None: ...
    def list_roots(self) -> list[RootRow]: ...
    def upsert_data_source(self, source: DataSourceRow) -> None: ...
    def list_data_sources(self) -> list[DataSourceRow]: ...
    def upsert_task(self, task: SyncTask) -> None: ...
    def claim_task(self, now: datetime, lease_for: timedelta) -> SyncTask | None: ...
    def complete_task(self, doc_id: UUID) -> None: ...
    def fail_task(self, doc_id: UUID, error: str, now: datetime) -> None: ...
    def exclude_document(self, doc_id: UUID, discovered_via: str, now: datetime) -> None: ...
    def commit_document(self, doc: DocumentDraft, now: datetime) -> None: ...
    def get_document(self, doc_id: UUID) -> DocumentDraft | None: ...
    def list_tasks(self) -> list[SyncTask]: ...
    def known_ids(self) -> set[UUID]: ...


async def discover(
    client: NotionClient,
    store: SyncStore,
    *,
    excluded: set[UUID],
    now: datetime | None = None,
    lag: timedelta = timedelta(minutes=15),
    root: UUID | None = None,
) -> set[UUID]:
    """Search, then walk block trees so pages search omitted are still found."""
    moment = now or datetime.now(UTC)
    watermark = _watermark(store)
    found: set[UUID] = set()
    if root is not None:
        store.upsert_task(
            SyncTask(
                doc_id=root,
                kind="page",
                priority=priority_for(kind="page", discovered_via="tree"),
                discovered_via="tree",
            )
        )
        found.add(root)
        await _walk_trees(client, store, excluded, moment, found)
        return found
    async for page in client.search_pages():
        if past_watermark(edited_at(page), watermark, lag):
            break
        doc_id = _consider(store, page, kind="page", excluded=excluded, now=moment, via="search")
        if doc_id is not None:
            found.add(doc_id)
    async for database in client.search_databases():
        if past_watermark(edited_at(database), watermark, lag):
            break
        doc_id = _consider(store, database, kind="database", excluded=excluded, now=moment, via="search")
        if doc_id is not None:
            found.add(doc_id)
            await _enqueue_rows(client, store, doc_id, excluded, moment)
    await _walk_trees(client, store, excluded, moment, found)
    store.set_state(
        "search_watermark",
        {"at": moment.isoformat(), "docs": len(store.known_ids())},
    )
    return found


async def run_sync(
    client: NotionClient,
    store: SyncStore,
    *,
    excluded: set[UUID],
    now: datetime | None = None,
    discover_only: bool = False,
    root: UUID | None = None,
    lease_for: timedelta = timedelta(minutes=5),
    drop_titles: dict[UUID, str] | None = None,
) -> dict[str, int]:
    moment = now or datetime.now(UTC)
    await discover(client, store, excluded=excluded, now=moment, root=root)
    if discover_only:
        counts = _count(store)
        return counts
    titles = drop_titles or {}
    while True:
        task = store.claim_task(moment, lease_for)
        if task is None:
            break
        if task.kind == "database":
            store.complete_task(task.doc_id)
            continue
        try:
            doc = await fetch_document(
                client,
                task.doc_id,
                drop_title=titles.get(task.doc_id) is not None
                and titles.get(task.doc_id) == _task_title(store, task.doc_id),
                kind="db_row" if task.kind == "db_row" else "page",
                discovered_via=task.discovered_via,
            )
            if doc.data_source_id is None and task.kind == "db_row":
                doc.kind = "db_row"
            store.commit_document(doc, moment)
            for child in doc.child_page_ids:
                if child in excluded:
                    store.exclude_document(child, "tree", moment)
                    continue
                store.upsert_task(
                    SyncTask(
                        doc_id=child,
                        kind="page",
                        priority=priority_for(kind="page", discovered_via="tree"),
                        discovered_via="tree",
                    )
                )
            store.complete_task(task.doc_id)
        except Exception as exc:
            store.fail_task(task.doc_id, str(exc), moment)
            if isinstance(exc, NotionError) and exc.status == 404:
                store.exclude_document(task.doc_id, task.discovered_via or "fetch", moment)
    return _count(store)


def _consider(
    store: SyncStore,
    payload: dict[str, Any],
    *,
    kind: str,
    excluded: set[UUID],
    now: datetime,
    via: str,
) -> UUID | None:
    raw_id = payload.get("id")
    if not isinstance(raw_id, str):
        return None
    try:
        doc_id = UUID(raw_id)
    except ValueError:
        return None
    parent = parent_of(payload)
    ancestors = [parent.id] if parent.id is not None else []
    if is_excluded(doc_id, ancestors, excluded):
        store.exclude_document(doc_id, via, now)
        return None
    if parent.type == "workspace" or parent.id is None:
        label = page_title(payload) or "Untitled"
        store.upsert_root(RootRow(id=doc_id, label=label, kind="discovered", created_at=now))
    substantial = kind == "db_row" and bool(page_title(payload))
    store.upsert_task(
        SyncTask(
            doc_id=doc_id,
            kind="page" if kind == "page" else kind,
            priority=priority_for(kind="page" if kind == "page" else kind, discovered_via=via, substantial=substantial),
            remote_last_edited=edited_at(payload),
            discovered_via=via,
        )
    )
    return doc_id


async def _enqueue_rows(
    client: NotionClient,
    store: SyncStore,
    database_id: UUID,
    excluded: set[UUID],
    now: datetime,
) -> None:
    if database_id in excluded:
        return
    try:
        database = await client.database(str(database_id))
    except NotionError as exc:
        if exc.status == 404:
            store.upsert_root(RootRow(id=database_id, label="Untitled", kind="discovered"))
            return
        raise
    sources = data_sources_of(database)
    if not sources:
        sources = [(database_id, page_title(database) or None)]
    for source_id, name in sources:
        rows = await client.query_data_source(str(source_id))
        titles = [page_title(row) for row in rows]
        store.upsert_data_source(
            DataSourceRow(
                id=source_id,
                database_id=database_id,
                title=name,
                row_count=len(rows),
                template_title=template_title(titles),
            )
        )
        for row in rows:
            row_id = _consider(store, row, kind="db_row", excluded=excluded, now=now, via="data_source")
            if row_id is None:
                continue


async def _walk_trees(
    client: NotionClient,
    store: SyncStore,
    excluded: set[UUID],
    now: datetime,
    found: set[UUID],
) -> None:
    walked: set[UUID] = set()
    while True:
        pending = [
            task.doc_id
            for task in store.list_tasks()
            if task.kind == "page" and task.doc_id not in walked and task.doc_id not in excluded
        ]
        if not pending:
            return
        for doc_id in pending:
            walked.add(doc_id)
            existing = store.get_document(doc_id)
            if existing is not None and existing.status == "excluded":
                continue
            try:
                blocks = await _deep_children(client, str(doc_id))
            except NotionError as exc:
                if exc.status == 404:
                    continue
                raise
            pages, databases = child_targets(blocks)
            for child in pages:
                if is_excluded(child, [doc_id], excluded):
                    store.exclude_document(child, "tree", now)
                    continue
                store.upsert_task(
                    SyncTask(
                        doc_id=child,
                        kind="page",
                        priority=priority_for(kind="page", discovered_via="tree"),
                        discovered_via="tree",
                    )
                )
                found.add(child)
            for child in databases:
                store.upsert_task(
                    SyncTask(
                        doc_id=child,
                        kind="database",
                        priority=priority_for(kind="database", discovered_via="tree"),
                        discovered_via="tree",
                    )
                )
                found.add(child)


async def _deep_children(client: NotionClient, block_id: str) -> list[dict[str, Any]]:
    blocks = await client.block_children(block_id)
    for block in blocks:
        kind = block.get("type")
        if block.get("has_children") and kind not in {"child_page", "child_database"}:
            block["children"] = await _deep_children(client, str(block.get("id")))
    return blocks


def _watermark(store: SyncStore) -> datetime | None:
    state = store.get_state("search_watermark")
    if not state:
        return None
    raw = state.get("at")
    if not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _count(store: SyncStore) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in store.list_tasks():
        counts[task.state] = counts.get(task.state, 0) + 1
    return counts


def _task_title(store: SyncStore, doc_id: UUID) -> str | None:
    doc = store.get_document(doc_id)
    if doc is None:
        return None
    return doc.title
