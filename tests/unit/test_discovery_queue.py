"""WP1.2 and WP1.3 discovery, exclusion, and the durable queue."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

import httpx
import pytest
import respx

from ndup.db.memory import MemoryStore
from ndup.db.queue import choose_task, fail_task
from ndup.model import SyncTask
from ndup.notion.client import NotionClient
from ndup.notion.discovery.ancestry import ParentRef, resolve_ancestry
from ndup.notion.sync import discover, fetch_document, run_sync

NOW = datetime(2024, 6, 1, tzinfo=UTC)


def _uuid(number: int) -> UUID:
    return UUID(int=number)


def _page(doc_id: UUID, parent: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "object": "page",
        "id": str(doc_id),
        "url": f"https://www.notion.so/{doc_id.hex}",
        "created_time": "2024-01-01T00:00:00.000Z",
        "last_edited_time": "2024-06-01T00:00:00.000Z",
        "parent": parent or {"type": "workspace", "workspace": True},
        "properties": {"title": {"type": "title", "title": [{"plain_text": "Notes", "annotations": {}}]}},
    }


def _paragraph(block_id: UUID, text: str) -> dict[str, object]:
    return {
        "object": "block",
        "id": str(block_id),
        "type": "paragraph",
        "has_children": False,
        "paragraph": {"rich_text": [{"plain_text": text, "annotations": {}}]},
    }


@pytest.mark.unit
async def test_search_miss_is_found_by_the_tree() -> None:
    root = _uuid(10)
    children = [_uuid(100 + index) for index in range(10)]
    omitted = children[-1]
    search_ids = {root, *children[:-1]}

    def search(request: httpx.Request) -> httpx.Response:
        body = request.read()
        if b"database" in body:
            return httpx.Response(200, json={"results": [], "has_more": False})
        results = [_page(doc_id) for doc_id in search_ids]
        return httpx.Response(200, json={"results": results, "has_more": False})

    def blocks(request: httpx.Request) -> httpx.Response:
        if str(root) in str(request.url):
            payload = [
                {
                    "object": "block",
                    "id": str(child),
                    "type": "child_page",
                    "has_children": False,
                    "child_page": {"title": "child"},
                }
                for child in children
            ]
        else:
            payload = []
        return httpx.Response(200, json={"results": payload, "has_more": False})

    store = MemoryStore()
    with respx.mock(assert_all_called=False) as router:
        router.post("https://api.notion.com/v1/search").mock(side_effect=search)
        router.get(url__regex=r".*/blocks/.*/children").mock(side_effect=blocks)
        async with NotionClient("token") as client:
            found = await discover(client, store, excluded=set(), now=NOW)
    assert omitted in found
    assert set(children) <= store.known_ids()


@pytest.mark.unit
async def test_unshared_ancestor_becomes_root() -> None:
    child = _uuid(20)
    parent = _uuid(21)
    with respx.mock() as router:
        router.get(f"https://api.notion.com/v1/pages/{parent}").mock(
            return_value=httpx.Response(404, json={"object": "error"})
        )
        async with NotionClient("token") as client:
            ancestry = await resolve_ancestry(
                client,
                child,
                ParentRef("page_id", parent),
                {},
            )
    assert ancestry.missing_parent
    assert ancestry.is_root
    assert ancestry.ancestors == []


@pytest.mark.unit
async def test_excluded_page_is_not_fetched() -> None:
    excluded = _uuid(30)
    hidden = _uuid(31)
    fetched: list[str] = []

    def blocks(request: httpx.Request) -> httpx.Response:
        fetched.append(str(request.url))
        return httpx.Response(200, json={"results": [], "has_more": False})

    store = MemoryStore()
    with respx.mock(assert_all_called=False) as router:
        router.post("https://api.notion.com/v1/search").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        _page(hidden, {"type": "page_id", "page_id": str(excluded)}),
                    ],
                    "has_more": False,
                },
            )
        )
        router.get(url__regex=r".*/blocks/.*/children").mock(side_effect=blocks)
        async with NotionClient("token") as client:
            await discover(client, store, excluded={excluded}, now=NOW)
    stored = store.get_document(hidden)
    assert stored is not None
    assert stored.title is None
    assert stored.status == "excluded"
    assert all(str(hidden) not in url for url in fetched)


@pytest.mark.unit
async def test_kill_resume_and_no_partial_page(monkeypatch: pytest.MonkeyPatch) -> None:
    doc_id = _uuid(40)
    calls = {"n": 0}

    async def flaky(*args: object, **kwargs: object) -> object:
        del args, kwargs
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("killed")
        from ndup.model import DocumentDraft

        return DocumentDraft(id=doc_id, kind="page", title="Kept", markdown="body", content_hash=b"abc")

    monkeypatch.setattr("ndup.notion.sync.fetch_document", flaky)
    store = MemoryStore()

    def search(request: httpx.Request) -> httpx.Response:
        if b"database" in request.content:
            return httpx.Response(200, json={"results": [], "has_more": False})
        return httpx.Response(200, json={"results": [_page(doc_id)], "has_more": False})

    with respx.mock(assert_all_called=False) as router:
        router.post("https://api.notion.com/v1/search").mock(side_effect=search)
        router.get(url__regex=r".*/blocks/.*/children").mock(
            return_value=httpx.Response(200, json={"results": [], "has_more": False})
        )
        async with NotionClient("token") as client:
            await run_sync(client, store, excluded=set(), now=NOW)
            assert store.get_document(doc_id) is None
            await run_sync(client, store, excluded=set(), now=NOW + timedelta(hours=1))
    stored = store.get_document(doc_id)
    assert stored is not None
    assert stored.title == "Kept"
    assert stored.markdown == "body"


@pytest.mark.unit
def test_lease_expiry_and_duplicate_tasks() -> None:
    now = NOW
    doc_id = _uuid(50)
    task = SyncTask(
        doc_id=doc_id,
        kind="page",
        priority=1,
        state="in_progress",
        leased_until=now - timedelta(seconds=1),
        attempts=1,
    )
    chosen = choose_task([task], now)
    assert chosen is not None
    assert chosen.doc_id == doc_id
    store = MemoryStore()
    store.upsert_task(SyncTask(doc_id=doc_id, kind="page", priority=1))
    store.upsert_task(SyncTask(doc_id=doc_id, kind="page", priority=1, discovered_via="tree"))
    assert len(store.list_tasks()) == 1
    failed = SyncTask(doc_id=doc_id, kind="page", priority=1, attempts=6, state="in_progress")
    fail_task(failed, "nope", now)
    assert failed.state == "failed"


@pytest.mark.unit
def test_fetch_symbol_is_patchable() -> None:
    assert fetch_document is not None
