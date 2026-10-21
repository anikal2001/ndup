"""Child page and database ids discovered while walking a block tree."""

from __future__ import annotations

from typing import Any
from uuid import UUID


def child_targets(blocks: list[dict[str, Any]]) -> tuple[list[UUID], list[UUID]]:
    pages: list[UUID] = []
    databases: list[UUID] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        try:
            block_id = UUID(str(block.get("id")))
        except ValueError:
            block_id = None
        if kind == "child_page" and block_id is not None:
            pages.append(block_id)
        elif kind == "child_database" and block_id is not None:
            databases.append(block_id)
        children = block.get("children")
        if isinstance(children, list):
            nested_pages, nested_databases = child_targets(children)
            pages.extend(nested_pages)
            databases.extend(nested_databases)
    return pages, databases
