"""Parent chain, area labels, and exclusion before any block fetch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from ndup.notion.client import NotionClient, NotionError


@dataclass
class ParentRef:
    type: str | None
    id: UUID | None


@dataclass
class Ancestry:
    ancestors: list[UUID]
    root_id: UUID | None
    is_root: bool
    missing_parent: bool


def parent_of(payload: dict[str, Any]) -> ParentRef:
    parent = payload.get("parent")
    if not isinstance(parent, dict):
        return ParentRef(None, None)
    kind = parent.get("type")
    if kind == "workspace":
        return ParentRef("workspace", None)
    if kind == "page_id":
        return ParentRef("page_id", _uuid(parent.get("page_id")))
    if kind == "database_id":
        return ParentRef("database_id", _uuid(parent.get("database_id")))
    if kind == "data_source_id":
        return ParentRef("data_source_id", _uuid(parent.get("data_source_id")))
    if kind == "block_id":
        return ParentRef("block_id", _uuid(parent.get("block_id")))
    return ParentRef(str(kind) if kind else None, None)


async def resolve_ancestry(
    client: NotionClient,
    doc_id: UUID,
    parent: ParentRef,
    known_titles: dict[UUID, str | None],
    *,
    fetch_missing: bool = True,
) -> Ancestry:
    """Walk parents. A 404 means the parent is not shared, so this doc is a root."""
    ancestors: list[UUID] = []
    seen: set[UUID] = {doc_id}
    current = parent
    missing = False
    while current.id is not None and current.id not in seen:
        seen.add(current.id)
        ancestors.append(current.id)
        if current.id in known_titles or current.type == "workspace":
            break
        if not fetch_missing:
            break
        try:
            if current.type == "database_id":
                payload = await client.database(str(current.id))
            else:
                payload = await client.page(str(current.id))
        except NotionError as exc:
            if exc.status == 404:
                missing = True
                ancestors.pop()
                break
            raise
        title = _title(payload)
        known_titles.setdefault(current.id, title)
        current = parent_of(payload)
    is_root = missing or parent.type == "workspace" or not ancestors
    root_id = ancestors[-1] if ancestors else doc_id
    return Ancestry(ancestors=ancestors, root_id=root_id, is_root=is_root, missing_parent=missing)


def is_excluded(doc_id: UUID, ancestors: list[UUID], excluded: set[UUID]) -> bool:
    if doc_id in excluded:
        return True
    return any(ancestor in excluded for ancestor in ancestors)


def area_label(
    doc_id: UUID,
    ancestors: list[UUID],
    roots: dict[UUID, str],
    titles: dict[UUID, str | None],
) -> str | None:
    """Label of the nearest configured root, else the top ancestor's title."""
    chain = [doc_id, *ancestors]
    for item in chain:
        if item in roots:
            return roots[item]
    for item in reversed(chain):
        title = titles.get(item)
        if title:
            return title
    return None


def _title(payload: dict[str, Any]) -> str | None:
    from ndup.notion.render import page_title

    title = page_title(payload)
    return title or None


def _uuid(value: object) -> UUID | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return UUID(value)
    except ValueError:
        return None
