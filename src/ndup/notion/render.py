"""Render a Notion block tree to Markdown plus an offset map.

Property text on a database row is prepended and carries no block id. Its
deep link is the page itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

from ndup.model import BlockRecord, SourceBlock


@dataclass
class RenderResult:
    markdown: str
    blocks: list[BlockRecord] = field(default_factory=list)
    source_blocks: list[SourceBlock] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)
    child_page_ids: list[UUID] = field(default_factory=list)
    child_database_ids: list[UUID] = field(default_factory=list)
    links: list[UUID] = field(default_factory=list)
    contributors: list[UUID] = field(default_factory=list)


def render_page(page: dict[str, Any], blocks: list[dict[str, Any]]) -> RenderResult:
    result = RenderResult(markdown="")
    property_text, properties = _properties(page)
    result.properties = properties
    pieces: list[str] = []
    if property_text:
        start = 0
        pieces.append(property_text + "\n\n")
        end = len(property_text)
        result.source_blocks.append(
            SourceBlock(
                block_id=None,
                text=property_text,
                plain=property_text,
                heading_level=None,
                is_code=False,
                is_synced=False,
                is_property=True,
                created_time=_time(page.get("created_time")),
                start_char=start,
                end_char=end,
            )
        )
    _render_blocks(blocks, result, pieces, synced=False)
    result.markdown = "".join(pieces).strip() + ("\n" if pieces else "")
    return result


def page_title(page: dict[str, Any]) -> str:
    properties = page.get("properties")
    if not isinstance(properties, dict):
        return ""
    for prop in properties.values():
        if isinstance(prop, dict) and prop.get("type") == "title":
            return rich_text(prop.get("title")).strip()
    title = page.get("title")
    if isinstance(title, list):
        return rich_text(title).strip()
    return ""


def rich_text(value: object) -> str:
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        text = str(item.get("plain_text") or "")
        annotations_raw = item.get("annotations")
        annotations = annotations_raw if isinstance(annotations_raw, dict) else {}
        if annotations.get("code"):
            text = f"`{text}`"
        if annotations.get("bold"):
            text = f"**{text}**"
        if annotations.get("italic"):
            text = f"*{text}*"
        href = item.get("href")
        if isinstance(href, str) and href:
            text = f"[{text}]({href})"
        parts.append(text)
    return "".join(parts)


def _properties(page: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Title and rich_text are embedded. Other types stay structured."""
    properties = page.get("properties")
    if not isinstance(properties, dict):
        return "", {}
    lines: list[str] = []
    title = page_title(page)
    if title:
        lines.append(title)
    structured: dict[str, Any] = {}
    for name, prop in properties.items():
        if not isinstance(prop, dict):
            continue
        kind = prop.get("type")
        if kind == "title":
            continue
        if kind == "rich_text":
            text = rich_text(prop.get("rich_text")).strip()
            if text:
                lines.append(f"{name}: {text}")
            continue
        structured[str(name)] = _plain_property(prop)
    return "\n".join(lines), structured


def _plain_property(prop: dict[str, Any]) -> Any:
    kind = prop.get("type")
    value = prop.get(kind) if isinstance(kind, str) else None
    if isinstance(value, dict) and "name" in value:
        return value.get("name")
    if isinstance(value, list):
        names = []
        for item in value:
            if isinstance(item, dict) and "name" in item:
                names.append(item.get("name"))
            else:
                names.append(item)
        return names
    return value


def _render_blocks(
    blocks: list[dict[str, Any]],
    result: RenderResult,
    pieces: list[str],
    *,
    synced: bool,
) -> None:
    number = 0
    for block in blocks:
        if not isinstance(block, dict):
            continue
        kind = str(block.get("type") or "")
        if kind == "numbered_list_item":
            number += 1
        else:
            number = 0
        _render_one(block, result, pieces, synced=synced, number=number)


def _render_one(
    block: dict[str, Any],
    result: RenderResult,
    pieces: list[str],
    *,
    synced: bool,
    number: int,
) -> None:
    kind = str(block.get("type") or "unsupported")
    payload_raw = block.get(kind)
    payload: dict[str, Any] = payload_raw if isinstance(payload_raw, dict) else {}
    children_raw = block.get("children")
    children = [child for child in children_raw if isinstance(child, dict)] if isinstance(children_raw, list) else []
    block_id = _uuid(block.get("id"))
    created = _time(block.get("created_time"))
    edited = _time(block.get("last_edited_time"))
    created_by = _user(block.get("created_by"))
    edited_by = _user(block.get("last_edited_by"))
    is_synced = synced or kind == "synced_block"
    start = sum(len(part) for part in pieces)

    if kind == "synced_block":
        _render_blocks(children, result, pieces, synced=True)
        return

    if kind in {"column_list", "column", "toggle"}:
        text = rich_text(payload.get("rich_text")) if kind == "toggle" else ""
        if text:
            _append_source(result, pieces, block_id, text + "\n", text, None, False, is_synced, created, start)
        _render_blocks(children, result, pieces, synced=is_synced)
        _record_block(result, block_id, kind, start, pieces, created, created_by, edited, edited_by, is_synced)
        return

    if kind == "child_page":
        title = str(payload.get("title") or "Untitled")
        target = block_id
        if target is not None:
            result.child_page_ids.append(target)
        link = f"[{title}](https://www.notion.so/{_notion_ref(block.get('id'))})"
        _append_source(result, pieces, block_id, link + "\n\n", title, None, False, is_synced, created, start)
        _record_block(result, block_id, kind, start, pieces, created, created_by, edited, edited_by, is_synced)
        return

    if kind == "child_database":
        title = str(payload.get("title") or "Database")
        if block_id is not None:
            result.child_database_ids.append(block_id)
        link = f"[{title}](https://www.notion.so/{_notion_ref(block.get('id'))})"
        _append_source(result, pieces, block_id, link + "\n\n", title, None, False, is_synced, created, start)
        _record_block(result, block_id, kind, start, pieces, created, created_by, edited, edited_by, is_synced)
        return

    if kind == "link_to_page":
        link_id = _link_target(payload)
        title = "linked page"
        if link_id is not None:
            result.links.append(link_id)
        link = f"[{title}](https://www.notion.so/{link_id})" if link_id else title
        _append_source(result, pieces, block_id, link + "\n\n", title, None, False, is_synced, created, start)
        _record_block(result, block_id, kind, start, pieces, created, created_by, edited, edited_by, is_synced)
        return

    if kind == "table":
        rendered = _table(children)
        _append_source(result, pieces, block_id, rendered, rendered, None, False, is_synced, created, start)
        _record_block(result, block_id, kind, start, pieces, created, created_by, edited, edited_by, is_synced)
        return

    text, plain, level, is_code = _inline(kind, payload, number)
    if text:
        _append_source(result, pieces, block_id, text, plain, level, is_code, is_synced, created, start)
    if children and kind not in {"child_page", "child_database"}:
        _render_blocks(children, result, pieces, synced=is_synced)
    if block_id is not None:
        _record_block(result, block_id, kind, start, pieces, created, created_by, edited, edited_by, is_synced)


def _inline(kind: str, payload: dict[str, Any], number: int) -> tuple[str, str, int | None, bool]:
    rich = rich_text(payload.get("rich_text"))
    if kind == "heading_1":
        return f"# {rich}\n\n", rich, 1, False
    if kind == "heading_2":
        return f"## {rich}\n\n", rich, 2, False
    if kind == "heading_3":
        return f"### {rich}\n\n", rich, 3, False
    if kind == "paragraph":
        return f"{rich}\n\n", rich, None, False
    if kind == "quote":
        return f"> {rich}\n\n", rich, None, False
    if kind == "callout":
        return f"> {rich}\n\n", rich, None, False
    if kind == "bulleted_list_item":
        return f"- {rich}\n", rich, None, False
    if kind == "numbered_list_item":
        return f"{number}. {rich}\n", rich, None, False
    if kind == "to_do":
        mark = "x" if payload.get("checked") else " "
        return f"- [{mark}] {rich}\n", rich, None, False
    if kind == "code":
        language = str(payload.get("language") or "")
        return f"```{language}\n{rich}\n```\n\n", rich, None, True
    if kind == "equation":
        expression = str(payload.get("expression") or rich)
        return f"$$\n{expression}\n$$\n\n", expression, None, False
    if kind in {"bookmark", "embed", "image", "video", "file", "pdf", "audio"}:
        caption = rich_text(payload.get("caption"))
        return (f"{caption}\n\n", caption, None, False) if caption else ("", "", None, False)
    if kind == "divider":
        return "---\n\n", "", None, False
    if rich:
        return f"{rich}\n\n", rich, None, False
    return "", "", None, False


def _table(children: list[dict[str, Any]]) -> str:
    rows: list[list[str]] = []
    for child in children:
        if not isinstance(child, dict) or child.get("type") != "table_row":
            continue
        row_raw = child.get("table_row")
        payload = row_raw if isinstance(row_raw, dict) else {}
        cells_raw = payload.get("cells")
        cells = cells_raw if isinstance(cells_raw, list) else []
        rows.append([rich_text(cell).replace("\n", " ") for cell in cells])
    if not rows:
        return ""
    width = max(len(row) for row in rows)

    def line(row: list[str]) -> str:
        padded = row + [""] * (width - len(row))
        return "| " + " | ".join(padded) + " |\n"

    rendered = [line(rows[0]), "| " + " | ".join("---" for _ in range(width)) + " |\n"]
    rendered.extend(line(row) for row in rows[1:])
    rendered.append("\n")
    return "".join(rendered)


def _append_source(
    result: RenderResult,
    pieces: list[str],
    block_id: UUID | None,
    text: str,
    plain: str,
    level: int | None,
    is_code: bool,
    is_synced: bool,
    created: datetime | None,
    start: int,
) -> None:
    if not text:
        return
    pieces.append(text)
    result.source_blocks.append(
        SourceBlock(
            block_id=block_id,
            text=text.strip("\n"),
            plain=plain,
            heading_level=level,
            is_code=is_code,
            is_synced=is_synced,
            is_property=False,
            created_time=created,
            start_char=start,
            end_char=start + len(text),
        )
    )


def _record_block(
    result: RenderResult,
    block_id: UUID | None,
    kind: str,
    start: int,
    pieces: list[str],
    created: datetime | None,
    created_by: UUID | None,
    edited: datetime | None,
    edited_by: UUID | None,
    is_synced: bool,
) -> None:
    if block_id is None:
        return
    end = sum(len(part) for part in pieces)
    result.blocks.append(
        BlockRecord(
            id=block_id,
            type=kind,
            start_char=start,
            end_char=end,
            created_time=created,
            created_by=created_by,
            last_edited_time=edited,
            last_edited_by=edited_by,
            is_synced=is_synced,
        )
    )
    for user in (created_by, edited_by):
        if user is not None and user not in result.contributors:
            result.contributors.append(user)


def _uuid(value: object) -> UUID | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return UUID(value)
    except ValueError:
        return None


def _user(value: object) -> UUID | None:
    if isinstance(value, dict):
        return _uuid(value.get("id"))
    return None


def _time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _link_target(payload: dict[str, Any]) -> UUID | None:
    for key in ("page_id", "database_id"):
        found = _uuid(payload.get(key))
        if found is not None:
            return found
    return None


def _notion_ref(value: object) -> str:
    return str(value or "").replace("-", "")
