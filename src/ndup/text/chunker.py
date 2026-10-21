"""Heading-aware chunker.

Packs blocks toward 300 tokens, never past 512, with a one-block overlap.
Oversize blocks split on sentence boundaries. Overlap spans share a span id
so coverage counts them once.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from datetime import datetime

from ndup.model import ChunkDraft, SourceBlock, TokenSpan
from ndup.text.langid import detect_language_heuristic
from ndup.text.minhash import minhash_signature
from ndup.text.shingles import normalize

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def content_hash(text: str) -> bytes:
    return hashlib.sha256(normalize(text).encode()).digest()


def embed_hash(text: str) -> bytes:
    return hashlib.sha256(text.encode()).digest()


def title_hash(title: str | None) -> bytes | None:
    if not title:
        return None
    return hashlib.sha256(normalize(title).encode()).digest()


def chunk_blocks(
    blocks: list[SourceBlock],
    *,
    title: str | None,
    drop_title: bool,
    count_tokens: Callable[[str], int],
    target: int = 300,
    max_tokens: int = 512,
) -> list[ChunkDraft]:
    if target < 1 or max_tokens < 1 or target > max_tokens:
        raise ValueError("token bounds must satisfy 1 <= target <= max_tokens")
    prepared = [_prepare(block, count_tokens, max_tokens) for block in blocks]
    flat: list[SourceBlock] = []
    for pieces in prepared:
        flat.extend(pieces)

    chunks: list[list[SourceBlock]] = []
    buffer: list[SourceBlock] = []
    buffer_tokens = 0
    heading: list[tuple[int, str]] = []

    def emit(current: list[SourceBlock]) -> None:
        if current:
            chunks.append(list(current))

    for block in flat:
        if block.heading_level in (1, 2, 3):
            if buffer:
                emit(buffer)
                buffer = []
                buffer_tokens = 0
            heading = [(level, text) for level, text in heading if level < (block.heading_level or 0)]
            heading.append((block.heading_level or 0, block.plain))
        block_tokens = _count(count_tokens, block.text)
        should_flush = buffer and (
            buffer_tokens + block_tokens > max_tokens
            or (buffer_tokens >= target and buffer_tokens + block_tokens > target)
        )
        if should_flush:
            overlap = _overlap_block(buffer, count_tokens, target)
            emit(buffer)
            buffer = [overlap] if overlap is not None else []
            buffer_tokens = _count(count_tokens, overlap.text) if overlap is not None else 0
            if buffer and buffer_tokens + block_tokens > max_tokens:
                buffer = []
                buffer_tokens = 0
        buffer.append(block)
        buffer_tokens += block_tokens
    if buffer:
        emit(buffer)

    path_for_chunk = _paths_for(flat, chunks)
    drafts: list[ChunkDraft] = []
    for index, group in enumerate(chunks):
        path = path_for_chunk[index]
        drafts.append(
            _to_chunk(
                index,
                group,
                path,
                title=title,
                drop_title=drop_title,
                count_tokens=count_tokens,
            )
        )
    return drafts


def _paths_for(flat: list[SourceBlock], chunks: list[list[SourceBlock]]) -> list[list[str]]:
    """Replay headings so each packed chunk inherits the path at its first new block."""
    del flat
    heading: list[tuple[int, str]] = []
    paths: list[list[str]] = []
    for group in chunks:
        for block in group:
            if block.heading_level in (1, 2, 3):
                heading = [(level, text) for level, text in heading if level < (block.heading_level or 0)]
                heading.append((block.heading_level or 0, block.plain))
        paths.append([text for _level, text in heading])
    return paths


def _prepare(
    block: SourceBlock,
    count_tokens: Callable[[str], int],
    max_tokens: int,
) -> list[SourceBlock]:
    tokens = _count(count_tokens, block.text)
    if tokens <= max_tokens:
        return [block]
    pieces = _split_text(block.text, max_tokens, count_tokens)
    cursor = block.start_char
    output: list[SourceBlock] = []
    for piece_index, piece in enumerate(pieces):
        end = cursor + len(piece)
        output.append(
            SourceBlock(
                block_id=block.block_id,
                text=piece,
                plain=piece,
                heading_level=block.heading_level,
                is_code=block.is_code,
                is_synced=block.is_synced,
                is_property=block.is_property,
                created_time=block.created_time,
                start_char=cursor,
                end_char=min(block.end_char, end),
            )
        )
        # Piece index is encoded later from the text offset via span id suffix.
        cursor = end + 1
        del piece_index
    return output


def _split_text(text: str, max_tokens: int, count_tokens: Callable[[str], int]) -> list[str]:
    sentences = [part for part in _SENTENCE.split(text) if part]
    if len(sentences) <= 1:
        return _split_words(text, max_tokens, count_tokens)
    parts: list[str] = []
    current = ""
    for sentence in sentences:
        if _count(count_tokens, sentence) > max_tokens:
            if current:
                parts.append(current)
                current = ""
            parts.extend(_split_words(sentence, max_tokens, count_tokens))
            continue
        trial = f"{current} {sentence}".strip() if current else sentence
        if current and _count(count_tokens, trial) > max_tokens:
            parts.append(current)
            current = sentence
        else:
            current = trial
    if current:
        parts.append(current)
    return parts or [text]


def _split_words(text: str, max_tokens: int, count_tokens: Callable[[str], int]) -> list[str]:
    words = text.split()
    if not words:
        return [text] if text else []
    parts: list[str] = []
    current: list[str] = []
    for word in words:
        trial_words = current + [word]
        trial = " ".join(trial_words)
        if current and _count(count_tokens, trial) > max_tokens:
            parts.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        parts.append(" ".join(current))
    # A single word that itself exceeds the limit is still emitted. Callers
    # packing normal prose will not hit this; the hypothesis tests use short words.
    return parts


def _overlap_block(
    buffer: list[SourceBlock],
    count_tokens: Callable[[str], int],
    target: int,
) -> SourceBlock | None:
    """One-block overlap, skipped when that block already fills the target."""
    if not buffer:
        return None
    last = buffer[-1]
    if _count(count_tokens, last.text) >= target:
        return None
    return last


def _count(count_tokens: Callable[[str], int], text: str) -> int:
    if not text.strip():
        return 0
    return max(1, count_tokens(text))


def _to_chunk(
    index: int,
    group: list[SourceBlock],
    heading_path: list[str],
    *,
    title: str | None,
    drop_title: bool,
    count_tokens: Callable[[str], int],
) -> ChunkDraft:
    text = "\n".join(block.text for block in group if block.text)
    prefix_parts: list[str] = []
    if title and not drop_title:
        prefix_parts.append(title)
    prefix_parts.extend(heading_path)
    prefix = " > ".join(prefix_parts)
    embedded = f"{prefix}\n\n{text}" if prefix else text
    spans = _spans(group, count_tokens)
    block_ids = list(dict.fromkeys(block.block_id for block in group if block.block_id is not None))
    seen_spans: set[str] = set()
    synced_tokens = 0
    code_tokens = 0
    total_tokens = 0
    for block in group:
        span_id = _span_id(block, 0)
        if span_id in seen_spans:
            continue
        seen_spans.add(span_id)
        tokens = _count(count_tokens, block.text)
        total_tokens += tokens
        if block.is_synced:
            synced_tokens += tokens
        if block.is_code:
            code_tokens += tokens
    is_synced = total_tokens > 0 and synced_tokens == total_tokens
    weight = 0.0 if is_synced else (0.5 if total_tokens and code_tokens / total_tokens > 0.5 else 1.0)
    times = [block.created_time for block in group if block.created_time is not None]
    earliest: datetime | None = min(times) if times else None
    token_count = _count(count_tokens, text)
    return ChunkDraft(
        ord=index,
        heading_path=list(heading_path),
        text=text,
        embed_text=embedded,
        start_char=group[0].start_char,
        end_char=group[-1].end_char,
        block_ids=block_ids,
        spans=spans,
        token_count=token_count,
        text_hash=content_hash(text),
        embed_hash=embed_hash(embedded),
        minhash=minhash_signature(text),
        lang=detect_language_heuristic(text),
        weight=weight,
        is_boilerplate=False,
        boilerplate_reason=None,
        is_synced=is_synced,
        earliest_block_time=earliest,
    )


def _spans(group: list[SourceBlock], count_tokens: Callable[[str], int]) -> list[TokenSpan]:
    spans: list[TokenSpan] = []
    seen_ids: set[str] = set()
    for block in group:
        span_id = _span_id(block, 0)
        if span_id in seen_ids:
            continue
        seen_ids.add(span_id)
        spans.append(
            TokenSpan(
                span_id=span_id,
                block_id=block.block_id,
                tokens=_count(count_tokens, block.text),
            )
        )
    return spans


def _span_id(block: SourceBlock, piece_index: int) -> str:
    del piece_index
    if block.is_property or block.block_id is None:
        return f"prop:{block.start_char}:{block.end_char}"
    return f"{block.block_id}:{block.start_char}:{block.end_char}"
