"""Fetch one page's block tree and turn it into a document draft.

The caller commits the draft in a single transaction. A crash before that
commit leaves no partial page.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from ndup.model import DocumentDraft
from ndup.notion.client import NotionClient
from ndup.notion.discovery.tree import child_targets
from ndup.notion.render import page_title, render_page
from ndup.text.chunker import chunk_blocks, content_hash, title_hash
from ndup.text.langid import majority_language
from ndup.text.tokens import simple_token_count


async def fetch_document(
    client: NotionClient,
    doc_id: UUID,
    *,
    title: str | None = None,
    drop_title: bool = False,
    kind: str = "page",
    parent_type: str | None = None,
    parent_id: UUID | None = None,
    data_source_id: UUID | None = None,
    root_id: UUID | None = None,
    area: str | None = None,
    discovered_via: str | None = None,
    count_tokens: Any = simple_token_count,
) -> DocumentDraft:
    page = await client.page(str(doc_id))
    blocks = await _block_tree(client, str(doc_id))
    rendered = render_page(page, blocks)
    page_label = title if title is not None else page_title(page)
    chunks = chunk_blocks(
        rendered.source_blocks,
        title=page_label,
        drop_title=drop_title,
        count_tokens=count_tokens,
    )
    token_count = sum(span.tokens for chunk in chunks for span in chunk.spans)
    lang = majority_language([(chunk.lang, max(chunk.token_count, 0)) for chunk in chunks])
    created = _time(page.get("created_time"))
    edited = _time(page.get("last_edited_time"))
    child_pages, child_databases = child_targets(blocks)
    # render_page also records children; prefer the walked tree so nested
    # containers are included, and union with the renderer's list.
    for extra in rendered.child_page_ids:
        if extra not in child_pages:
            child_pages.append(extra)
    for extra in rendered.child_database_ids:
        if extra not in child_databases:
            child_databases.append(extra)
    return DocumentDraft(
        id=doc_id,
        kind=kind,
        parent_type=parent_type,
        parent_id=parent_id,
        data_source_id=data_source_id,
        root_id=root_id,
        area=area,
        title=page_label or None,
        title_hash=title_hash(page_label),
        url=page.get("url") if isinstance(page.get("url"), str) else f"https://www.notion.so/{doc_id.hex}",
        created_time=created,
        created_by=_user(page.get("created_by")),
        last_edited_time=edited,
        last_edited_by=_user(page.get("last_edited_by")),
        properties=rendered.properties,
        markdown=rendered.markdown,
        content_hash=content_hash(rendered.markdown or ""),
        token_count=token_count,
        lang=lang,
        contributors=rendered.contributors,
        is_stub=token_count < 40,
        status="active",
        discovered_via=discovered_via,
        chunks=chunks,
        blocks=rendered.blocks,
        links=rendered.links,
        child_page_ids=child_pages,
        child_database_ids=child_databases,
    )


async def _block_tree(client: NotionClient, block_id: str) -> list[dict[str, Any]]:
    children = await client.block_children(block_id)
    for child in children:
        kind = child.get("type")
        if kind in {"child_page", "child_database"}:
            continue
        if child.get("has_children"):
            child["children"] = await _block_tree(client, str(child.get("id")))
    return children


def _time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _user(value: object) -> UUID | None:
    if isinstance(value, dict) and isinstance(value.get("id"), str):
        try:
            return UUID(value["id"])
        except ValueError:
            return None
    return None


def utcnow() -> datetime:
    return datetime.now(UTC)
