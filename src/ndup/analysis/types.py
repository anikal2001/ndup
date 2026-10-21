"""Analysis outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID


@dataclass(frozen=True)
class PassageMatch:
    chunk_a: int
    chunk_b: int
    sim: float
    lex: float


@dataclass
class PairRecord:
    doc_a: UUID
    doc_b: UUID
    doc_sim: float
    cov_ab: float
    cov_ba: float
    lex_jaccard: float
    lex_cont_ab: float
    lex_cont_ba: float
    score: float
    relation: str
    median_match: float
    passages: list[PassageMatch] = field(default_factory=list)


@dataclass
class MemberRecord:
    doc_id: UUID
    relation_to_cluster: str
    is_satellite: bool


@dataclass
class ClusterRecord:
    stable_key: UUID
    label: str
    size: int
    member_hash: bytes
    oldest_doc: UUID | None
    newest_doc: UUID | None
    active_copies: int
    areas: list[str]
    members: list[MemberRecord]
    passage_origin_doc: UUID | None = None
    passage_origin_time: datetime | None = None
    created_at: datetime | None = None


@dataclass
class ClusterEventRecord:
    stable_key: UUID
    doc_id: UUID | None
    event: str
    other_key: UUID | None = None


@dataclass
class PrevCluster:
    stable_key: UUID
    members: set[UUID]
    created_at: datetime
    note: str | None = None


@dataclass
class AnalysisResult:
    pairs: list[PairRecord]
    clusters: list[ClusterRecord]
    events: list[ClusterEventRecord]
    stub_groups: list[list[UUID]]
    other_language_docs: int
    stub_docs: int
    stats: dict[str, Any]
    thresholds: dict[str, Any]
    review_notes: dict[UUID, str] = field(default_factory=dict)
