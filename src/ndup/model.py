"""Domain records shared by sync, storage, and analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

import numpy as np


@dataclass
class TokenSpan:
    span_id: str
    block_id: UUID | None
    tokens: int


@dataclass
class SourceBlock:
    """One rendered block, ready for the chunker."""

    block_id: UUID | None
    text: str
    plain: str
    heading_level: int | None
    is_code: bool
    is_synced: bool
    is_property: bool
    created_time: datetime | None
    start_char: int
    end_char: int


@dataclass
class BlockRecord:
    id: UUID
    type: str
    start_char: int
    end_char: int
    created_time: datetime | None
    created_by: UUID | None
    last_edited_time: datetime | None
    last_edited_by: UUID | None
    is_synced: bool


@dataclass
class ChunkDraft:
    ord: int
    heading_path: list[str]
    text: str
    embed_text: str
    start_char: int
    end_char: int
    block_ids: list[UUID]
    spans: list[TokenSpan]
    token_count: int
    text_hash: bytes
    embed_hash: bytes
    minhash: bytes
    lang: str
    weight: float
    is_boilerplate: bool
    boilerplate_reason: str | None
    is_synced: bool
    earliest_block_time: datetime | None
    vector: np.ndarray | None = None
    chunk_id: int | None = None


@dataclass
class DocumentDraft:
    id: UUID
    kind: str
    parent_type: str | None = None
    parent_id: UUID | None = None
    data_source_id: UUID | None = None
    root_id: UUID | None = None
    area: str | None = None
    breadcrumb: list[str] = field(default_factory=list)
    title: str | None = None
    title_hash: bytes | None = None
    url: str | None = None
    created_time: datetime | None = None
    created_by: UUID | None = None
    last_edited_time: datetime | None = None
    last_edited_by: UUID | None = None
    last_content_change: datetime | None = None
    properties: dict[str, Any] = field(default_factory=dict)
    markdown: str | None = None
    content_hash: bytes | None = None
    token_count: int = 0
    lang: str | None = None
    contributors: list[UUID] = field(default_factory=list)
    is_stub: bool = False
    status: str = "active"
    discovered_via: str | None = None
    first_seen_at: datetime | None = None
    last_synced_at: datetime | None = None
    chunks: list[ChunkDraft] = field(default_factory=list)
    blocks: list[BlockRecord] = field(default_factory=list)
    links: list[UUID] = field(default_factory=list)
    child_page_ids: list[UUID] = field(default_factory=list)
    child_database_ids: list[UUID] = field(default_factory=list)
    doc_vector: np.ndarray | None = None


@dataclass
class SyncTask:
    doc_id: UUID
    kind: str
    priority: int
    remote_last_edited: datetime | None = None
    discovered_via: str | None = None
    state: str = "pending"
    attempts: int = 0
    next_attempt_at: datetime | None = None
    last_error: str | None = None
    leased_until: datetime | None = None


@dataclass
class RootRow:
    id: UUID
    label: str
    kind: str
    excluded: bool = False
    created_at: datetime | None = None


@dataclass
class DataSourceRow:
    id: UUID
    database_id: UUID | None = None
    title: str | None = None
    row_count: int = 0
    template_title: str | None = None
    poll_interval_seconds: int = 3600
    next_poll_at: datetime | None = None
    last_polled_at: datetime | None = None
    last_change_seen: datetime | None = None
    watermark: datetime | None = None


@dataclass
class UserRow:
    id: UUID
    type: str | None
    name: str | None
    email: str | None
    avatar_url: str | None
    updated_at: datetime | None = None


@dataclass
class ModelRow:
    id: int
    model: str
    dims: int
    runtime: str
    thresholds: dict[str, Any]
    is_active: bool = True


@dataclass
class IgnoreRule:
    kind: str
    a: UUID
    b: UUID | None
    created_by: str | None = None
    created_at: datetime | None = None


@dataclass
class ReviewRow:
    stable_key: UUID
    status: str
    reviewed_member_hash: bytes | None
    note: str | None
    reviewed_by: str | None
    reviewed_at: datetime | None


def order_pair(a: UUID, b: UUID) -> tuple[UUID, UUID]:
    if a.int == b.int:
        raise ValueError("A pair requires two documents")
    return (a, b) if a.int < b.int else (b, a)
