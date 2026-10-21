"""Exact word shingles for Jaccard and containment."""

from __future__ import annotations


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def word_shingles(text: str, n: int = 5) -> set[tuple[str, ...]]:
    words = normalize(text).split()
    if not words:
        return set()
    if len(words) <= n:
        return {tuple(words)}
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def jaccard(left: set[tuple[str, ...]], right: set[tuple[str, ...]]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    union = len(left | right)
    if union == 0:
        return 0.0
    return len(left & right) / union


def containment(left: set[tuple[str, ...]], right: set[tuple[str, ...]]) -> float:
    """Share of ``left`` that also appears in ``right``."""
    if not left:
        return 0.0
    return len(left & right) / len(left)
