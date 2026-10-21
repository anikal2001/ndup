"""Per-surface redaction. One function, applied before text leaves the process."""

from __future__ import annotations

from typing import Literal

ContentMode = Literal["full", "snippets", "metadata"]
SNIPPET_CHARS = 300


def redact_text(text: str | None, mode: ContentMode) -> str | None:
    if text is None or mode == "metadata":
        return None
    if mode == "full":
        return text
    if len(text) <= SNIPPET_CHARS:
        return text
    return text[:SNIPPET_CHARS]


def redact_pair_passage(text: str | None, mode: ContentMode) -> str | None:
    """Pair diff and doc lens never return more than 300 characters in snippets mode."""
    return redact_text(text, mode)
