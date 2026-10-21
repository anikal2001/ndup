"""Hash and row-template boilerplate detectors.

The density detector lives in ``analysis.templates`` because it needs chunk kNN.
"""

from __future__ import annotations

from collections import defaultdict
from uuid import UUID

from ndup.model import ChunkDraft, DocumentDraft


def hash_threshold(doc_count: int) -> float:
    """``max(20, 0.05% of docs)``."""
    return max(20.0, 0.0005 * doc_count)


def apply_hash_boilerplate(docs: list[DocumentDraft], doc_count: int | None = None) -> None:
    total = doc_count if doc_count is not None else len(docs)
    threshold = hash_threshold(total)
    owners: dict[bytes, set[UUID]] = defaultdict(set)
    for doc in docs:
        seen: set[bytes] = set()
        for chunk in doc.chunks:
            if chunk.is_synced or chunk.text_hash in seen:
                continue
            seen.add(chunk.text_hash)
            owners[chunk.text_hash].add(doc.id)
    for doc in docs:
        for chunk in doc.chunks:
            if len(owners.get(chunk.text_hash, ())) >= threshold:
                _mark(chunk, "hash")


def apply_row_template(docs: list[DocumentDraft]) -> None:
    grouped: dict[UUID, list[DocumentDraft]] = defaultdict(list)
    for doc in docs:
        if doc.kind == "db_row" and doc.data_source_id is not None and doc.status == "active":
            grouped[doc.data_source_id].append(doc)
    for rows in grouped.values():
        if len(rows) < 10:
            continue
        owners: dict[bytes, set[UUID]] = defaultdict(set)
        for doc in rows:
            for chunk in doc.chunks:
                if not chunk.is_synced:
                    owners[chunk.text_hash].add(doc.id)
        for doc in rows:
            for chunk in doc.chunks:
                if chunk.is_boilerplate:
                    continue
                if len(owners.get(chunk.text_hash, ())) / len(rows) >= 0.30:
                    _mark(chunk, "row_template")


def template_title(titles: list[str]) -> str | None:
    """Title that covers at least 5% of the rows in one data source."""
    cleaned = [title.strip() for title in titles if title and title.strip()]
    if not cleaned:
        return None
    counts: dict[str, int] = defaultdict(int)
    for title in cleaned:
        counts[title] += 1
    title, count = max(counts.items(), key=lambda item: (item[1], item[0]))
    if count / len(cleaned) >= 0.05:
        return title
    return None


def _mark(chunk: ChunkDraft, reason: str) -> None:
    chunk.is_boilerplate = True
    chunk.boilerplate_reason = reason
    chunk.weight = 0.0
