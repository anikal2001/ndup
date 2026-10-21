"""Stable cluster keys: largest intersection, ties to the older key."""

from __future__ import annotations

import hashlib
from datetime import datetime
from uuid import UUID, uuid5

from ndup.analysis.types import ClusterEventRecord, PrevCluster

_NAMESPACE = UUID("6f1d8d1a-0b7a-4e7c-9c1a-2a7e0d5b9c11")


def member_hash(doc_ids: set[UUID] | list[UUID]) -> bytes:
    payload = "\n".join(sorted(str(doc_id) for doc_id in doc_ids)).encode()
    return hashlib.sha256(payload).digest()


def review_after(status: str | None, reviewed_hash: bytes | None, new_hash: bytes) -> str:
    if status == "reviewed" and reviewed_hash is not None and reviewed_hash != new_hash:
        return "changed"
    return status or "open"


def assign_stable_keys(
    groups: list[set[UUID]],
    previous: list[PrevCluster],
    now: datetime,
) -> tuple[list[UUID], list[ClusterEventRecord], dict[UUID, str]]:
    """Return a key per group, events, and notes copied onto merge survivors."""
    order = sorted(
        range(len(groups)),
        key=lambda index: (-len(groups[index]), sorted(str(item) for item in groups[index])),
    )
    wishes: list[tuple[int, PrevCluster | None]] = []
    for index in order:
        members = groups[index]
        best: PrevCluster | None = None
        best_overlap = 0
        for prev in previous:
            overlap = len(members & prev.members)
            if overlap <= 0:
                continue
            if (
                best is None
                or overlap > best_overlap
                or (overlap == best_overlap and prev.created_at < best.created_at)
            ):
                best = prev
                best_overlap = overlap
        wishes.append((index, best))

    assigned: dict[int, UUID] = {}
    used: set[UUID] = set()
    events: list[ClusterEventRecord] = []
    for index, best in wishes:
        if best is not None and best.stable_key not in used:
            assigned[index] = best.stable_key
            used.add(best.stable_key)
            joined = groups[index] - best.members
            left = best.members - groups[index]
            for doc_id in sorted(joined, key=lambda item: item.int):
                events.append(ClusterEventRecord(best.stable_key, doc_id, "joined"))
            for doc_id in sorted(left, key=lambda item: item.int):
                events.append(ClusterEventRecord(best.stable_key, doc_id, "left"))
        elif best is not None:
            fresh = _new_key(groups[index], now, index)
            assigned[index] = fresh
            events.append(ClusterEventRecord(fresh, None, "split_from", other_key=best.stable_key))
            for doc_id in sorted(groups[index], key=lambda item: item.int):
                events.append(ClusterEventRecord(fresh, doc_id, "joined"))
        else:
            fresh = _stable_new_key(groups[index])
            assigned[index] = fresh
            for doc_id in sorted(groups[index], key=lambda item: item.int):
                events.append(ClusterEventRecord(fresh, doc_id, "joined"))

    inherited = set(assigned.values())
    notes: dict[UUID, str] = {}
    for prev in previous:
        if prev.stable_key in inherited:
            continue
        target: UUID | None = None
        best_overlap = 0
        for index, members in enumerate(groups):
            overlap = len(members & prev.members)
            if overlap > best_overlap:
                best_overlap = overlap
                target = assigned[index]
        if target is not None and best_overlap > 0:
            events.append(ClusterEventRecord(prev.stable_key, None, "merged_into", other_key=target))
            if prev.note:
                existing = notes.get(target, "")
                extra = f"merged from {prev.stable_key}: {prev.note}"
                notes[target] = f"{existing}\n{extra}".strip()
    keys = [assigned[index] for index in range(len(groups))]
    return keys, events, notes


def _stable_new_key(members: set[UUID]) -> UUID:
    material = ",".join(sorted(str(doc_id) for doc_id in members))
    return uuid5(_NAMESPACE, material)


def _new_key(members: set[UUID], now: datetime, index: int) -> UUID:
    material = f"{_stable_new_key(members)}:{now.isoformat()}:{index}"
    return uuid5(_NAMESPACE, material)
