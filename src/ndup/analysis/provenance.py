"""Oldest, newest, and active copies. No canonical score."""

from __future__ import annotations

from datetime import datetime, timedelta
from uuid import UUID

from ndup.analysis.types import PairRecord
from ndup.model import DocumentDraft


def oldest_doc(docs: list[DocumentDraft]) -> DocumentDraft | None:
    if not docs:
        return None
    return min(docs, key=_oldest_key)


def newest_doc(docs: list[DocumentDraft]) -> DocumentDraft | None:
    if not docs:
        return None
    return max(docs, key=_newest_key)


def content_time(doc: DocumentDraft) -> datetime | None:
    """Observed content change, otherwise Notion's last edit (any touch)."""
    return doc.last_content_change or doc.last_edited_time


def active_copy_count(docs: list[DocumentDraft], now: datetime, window_days: int) -> int:
    cutoff = now - timedelta(days=window_days)
    count = 0
    for doc in docs:
        moment = content_time(doc)
        if moment is not None and moment >= cutoff:
            count += 1
    return count


def passage_origin(
    pairs: list[PairRecord],
    member_ids: set[UUID],
    docs: dict[UUID, DocumentDraft],
) -> tuple[UUID | None, datetime | None]:
    """Earliest block time among matched passages inside the cluster."""
    best_doc: UUID | None = None
    best_time: datetime | None = None
    chunk_time = _chunk_times(docs)
    for pair in pairs:
        if pair.doc_a not in member_ids or pair.doc_b not in member_ids:
            continue
        if pair.relation not in {"near_copy", "duplicate", "subset", "partial"}:
            continue
        for passage in pair.passages:
            for doc_id, chunk_ord in ((pair.doc_a, passage.chunk_a), (pair.doc_b, passage.chunk_b)):
                moment = chunk_time.get((doc_id, chunk_ord))
                if moment is None:
                    continue
                if best_time is None or moment < best_time:
                    best_time = moment
                    best_doc = doc_id
    return best_doc, best_time


def cluster_label(docs: list[DocumentDraft]) -> str:
    """Top terms from core titles and headings, deterministic on ties."""
    stop = {
        "the",
        "and",
        "of",
        "to",
        "in",
        "a",
        "for",
        "on",
        "with",
        "is",
        "page",
        "notes",
    }
    counts: dict[str, int] = {}
    for doc in docs:
        texts = [doc.title or ""]
        for chunk in doc.chunks:
            texts.extend(chunk.heading_path)
        for text in texts:
            for raw in text.lower().split():
                word = "".join(char for char in raw if char.isalnum())
                if len(word) < 3 or word in stop or word.isdigit():
                    continue
                counts[word] = counts.get(word, 0) + 1
    if not counts:
        return "Untitled cluster"
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:3]
    return " ".join(word for word, _count in ranked)


def _oldest_key(doc: DocumentDraft) -> tuple[datetime, datetime, str]:
    created = doc.created_time or datetime.max.replace(tzinfo=None)
    first_block = _first_block_time(doc) or created
    return (created, first_block, str(doc.id))


def _newest_key(doc: DocumentDraft) -> tuple[datetime, str]:
    moment = content_time(doc) or datetime.min.replace(tzinfo=None)
    return (moment, str(doc.id))


def _first_block_time(doc: DocumentDraft) -> datetime | None:
    times = [block.created_time for block in doc.blocks if block.created_time is not None]
    chunk_times = [chunk.earliest_block_time for chunk in doc.chunks if chunk.earliest_block_time]
    times.extend(chunk_times)
    if not times:
        return None
    return min(times)


def _chunk_times(docs: dict[UUID, DocumentDraft]) -> dict[tuple[UUID, int], datetime]:
    found: dict[tuple[UUID, int], datetime] = {}
    for doc in docs.values():
        for chunk in doc.chunks:
            if chunk.earliest_block_time is not None:
                found[(doc.id, chunk.ord)] = chunk.earliest_block_time
    return found
