"""Token counting used when the model tokenizer is not installed."""

from __future__ import annotations


def simple_token_count(text: str) -> int:
    """Whitespace tokens. Empty text is zero; any other text is at least one."""
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, len(stripped.split()))
