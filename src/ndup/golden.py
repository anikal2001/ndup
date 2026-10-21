"""Labeled synthetic workspace used to fit and gate detection quality.

Live Notion content is not required. Planted pairs use disjoint token families
so the fake embedder's cosine tracks the intended relation.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

from ndup.embed.fake_provider import FakeEmbeddingProvider
from ndup.model import DocumentDraft, SourceBlock
from ndup.text.chunker import chunk_blocks, content_hash, title_hash
from ndup.text.langid import detect_language_heuristic, majority_language
from ndup.text.tokens import simple_token_count

Label = tuple[UUID, UUID, str]
_EPOCH = datetime(2024, 1, 1, tzinfo=UTC)
_FUNCTION = ("the", "and", "of", "to", "in")


def build_golden_workspace() -> tuple[list[DocumentDraft], list[Label]]:
    """About 500 documents, with labeled near-copies, duplicates, subsets, and partials."""
    docs: list[DocumentDraft] = []
    labels: list[Label] = []
    cursor = 1

    def take() -> UUID:
        nonlocal cursor
        value = UUID(int=cursor)
        cursor += 1
        return value

    for index in range(24):
        left = _english_doc(
            take(),
            title=f"Near copy {index}",
            sections=_sections(f"near{index}a", 2, 320),
            created=_EPOCH + timedelta(days=index),
            edited=_EPOCH + timedelta(days=index, hours=1),
            area="Engineering" if index % 2 == 0 else "Sales",
        )
        right = _english_doc(
            take(),
            title=f"Near copy {index}",
            sections=_sections(f"near{index}a", 2, 320),
            created=_EPOCH + timedelta(days=index, hours=2),
            edited=_EPOCH + timedelta(days=30 + index),
            area=left.area or "Engineering",
        )
        docs.extend((left, right))
        labels.append(_label(left.id, right.id, "near_copy"))

    for index in range(24):
        shared = _sections(f"dup{index}a", 3, 320)
        left = _english_doc(
            take(),
            title=f"Duplicate {index}",
            sections=shared + _sections(f"dup{index}l", 1, 320),
            created=_EPOCH + timedelta(days=100 + index),
            edited=_EPOCH + timedelta(days=100 + index),
            area="Engineering",
        )
        right = _english_doc(
            take(),
            title=f"Duplicate {index}",
            sections=shared + _sections(f"dup{index}r", 1, 320),
            created=_EPOCH + timedelta(days=100 + index, hours=3),
            edited=_EPOCH + timedelta(days=140 + index),
            area="Engineering",
        )
        docs.extend((left, right))
        labels.append(_label(left.id, right.id, "duplicate"))

    for index in range(12):
        full = _sections(f"sub{index}", 4, 280)
        large = _english_doc(
            take(),
            title=f"Superset {index}",
            sections=full,
            created=_EPOCH + timedelta(days=200 + index),
            edited=_EPOCH + timedelta(days=200 + index),
            area="Sales",
        )
        small = _english_doc(
            take(),
            title=f"Subset {index}",
            sections=full[:1],
            created=_EPOCH + timedelta(days=201 + index),
            edited=_EPOCH + timedelta(days=201 + index),
            area="Sales",
        )
        docs.extend((large, small))
        labels.append(_label(large.id, small.id, "subset"))

    for index in range(12):
        shared = _sections(f"partshare{index}", 1, 280)
        left = _english_doc(
            take(),
            title=f"Partial left {index}",
            sections=shared + _sections(f"partl{index}", 3, 280),
            created=_EPOCH + timedelta(days=300 + index),
            edited=_EPOCH + timedelta(days=300 + index),
            area="Engineering",
        )
        right = _english_doc(
            take(),
            title=f"Partial right {index}",
            sections=shared + _sections(f"partr{index}", 3, 280),
            created=_EPOCH + timedelta(days=301 + index),
            edited=_EPOCH + timedelta(days=301 + index),
            area="Engineering",
        )
        docs.extend((left, right))
        labels.append(_label(left.id, right.id, "partial"))

    for index in range(16):
        words = _word_run(f"short{index}", 50)
        left = _english_doc(
            take(),
            title=f"Short copy {index}",
            sections=[("Note", words)],
            created=_EPOCH + timedelta(days=400 + index),
            edited=_EPOCH + timedelta(days=400 + index),
            area="Sales",
        )
        right = _english_doc(
            take(),
            title=f"Short copy {index}",
            sections=[("Note", list(words))],
            created=_EPOCH + timedelta(days=400 + index, hours=4),
            edited=_EPOCH + timedelta(days=410 + index),
            area="Sales",
        )
        docs.extend((left, right))
        labels.append(_label(left.id, right.id, "near_copy"))

    template_source = UUID(int=900_001)
    template_words = _word_run("templatebody", 320)
    for index in range(12):
        docs.append(
            _row_doc(
                take(),
                title="Weekly status",
                sections=[("Template", list(template_words)), ("Notes", _word_run(f"note{index}", 40))],
                data_source_id=template_source,
                created=_EPOCH + timedelta(days=500 + index),
                drop_title=True,
            )
        )

    density_source = UUID(int=900_002)
    density_base = _word_run("densitybody", 60)
    for index in range(12):
        words = list(density_base) + [f"uniq{index}"]
        docs.append(
            _row_doc(
                take(),
                title=f"Density {index}",
                sections=[("Row", words)],
                data_source_id=density_source,
                created=_EPOCH + timedelta(days=600 + index),
                drop_title=False,
            )
        )

    for index in range(8):
        docs.append(
            _plain_doc(
                take(),
                title=f"Informe {index}",
                body=(
                    "el la de que y en los del se las un por con una su para es al "
                    f"lo como más pero sus le ya fue tema{index} "
                ),
                lang_hint="es",
                created=_EPOCH + timedelta(days=700 + index),
            )
        )
    for index in range(4):
        docs.append(
            _plain_doc(
                take(),
                title=f"预算 {index}",
                body="这是一个重复的页面关于预算和计划" * 8,
                lang_hint="zh",
                created=_EPOCH + timedelta(days=720 + index),
            )
        )

    for index in range(5):
        for _copy in range(2):
            docs.append(
                _stub(
                    take(),
                    title=f"Stub Report {index}",
                    created=_EPOCH + timedelta(days=800 + index),
                )
            )

    # Fill toward the ~500 doc workspace without creating accidental copies.
    target = 500
    filler_index = 0
    while len(docs) < target:
        docs.append(
            _english_doc(
                take(),
                title=f"Filler {filler_index}",
                sections=[("Body", _word_run(f"fill{filler_index}", 45))],
                created=_EPOCH + timedelta(days=900 + filler_index),
                edited=_EPOCH + timedelta(days=900 + filler_index),
                area="Engineering" if filler_index % 2 == 0 else "Sales",
            )
        )
        filler_index += 1
    return docs, labels


def embed_golden(docs: list[DocumentDraft]) -> FakeEmbeddingProvider:
    from ndup.analysis.pipeline import embed_documents, prepare_documents

    prepare_documents(docs)
    provider = FakeEmbeddingProvider()
    embed_documents(docs, provider, {})
    return provider


def _sections(prefix: str, count: int, size: int) -> list[tuple[str, list[str]]]:
    return [(f"Section {index}", _word_run(f"{prefix}s{index}", size)) for index in range(count)]


def _word_run(prefix: str, size: int) -> list[str]:
    words: list[str] = []
    for index in range(size):
        if index % 17 == 0:
            words.append(_FUNCTION[index % len(_FUNCTION)])
        else:
            words.append(f"{prefix}w{index}")
    return words


def _paraphrase(words: list[str]) -> list[str]:
    rewritten: list[str] = []
    for index, word in enumerate(words):
        if word in _FUNCTION or index % 4 != 3:
            rewritten.append(word)
        else:
            rewritten.append(f"alt{word}")
    return rewritten


def _english_doc(
    doc_id: UUID,
    *,
    title: str,
    sections: list[tuple[str, list[str]]],
    created: datetime,
    edited: datetime,
    area: str,
    drop_title: bool = False,
    kind: str = "page",
    data_source_id: UUID | None = None,
) -> DocumentDraft:
    blocks = _section_blocks(sections)
    return _from_blocks(
        doc_id,
        title=title,
        blocks=blocks,
        created=created,
        edited=edited,
        area=area,
        drop_title=drop_title,
        kind=kind,
        data_source_id=data_source_id,
    )


def _row_doc(
    doc_id: UUID,
    *,
    title: str,
    sections: list[tuple[str, list[str]]],
    data_source_id: UUID,
    created: datetime,
    drop_title: bool,
) -> DocumentDraft:
    return _english_doc(
        doc_id,
        title=title,
        sections=sections,
        created=created,
        edited=created,
        area="Operations",
        drop_title=drop_title,
        kind="db_row",
        data_source_id=data_source_id,
    )


def _plain_doc(
    doc_id: UUID,
    *,
    title: str,
    body: str,
    lang_hint: str,
    created: datetime,
) -> DocumentDraft:
    block = SourceBlock(
        block_id=UUID(int=doc_id.int + 10_000_000),
        text=body,
        plain=body,
        heading_level=None,
        is_code=False,
        is_synced=False,
        is_property=False,
        created_time=created,
        start_char=0,
        end_char=len(body),
    )
    doc = _from_blocks(
        doc_id,
        title=title,
        blocks=[block],
        created=created,
        edited=created,
        area="Archive",
        drop_title=False,
        kind="page",
        data_source_id=None,
    )
    doc.lang = lang_hint
    for chunk in doc.chunks:
        chunk.lang = lang_hint
    return doc


def _stub(doc_id: UUID, *, title: str, created: datetime) -> DocumentDraft:
    doc = _english_doc(
        doc_id,
        title=title,
        sections=[("TBD", ["pending"])],
        created=created,
        edited=created,
        area="Archive",
    )
    doc.is_stub = True
    doc.token_count = 2
    return doc


def _from_blocks(
    doc_id: UUID,
    *,
    title: str,
    blocks: list[SourceBlock],
    created: datetime,
    edited: datetime,
    area: str,
    drop_title: bool,
    kind: str,
    data_source_id: UUID | None,
) -> DocumentDraft:
    chunks = chunk_blocks(
        blocks,
        title=title,
        drop_title=drop_title,
        count_tokens=simple_token_count,
    )
    markdown = "\n".join(block.text for block in blocks)
    token_count = sum(span.tokens for chunk in chunks for span in chunk.spans)
    lang = majority_language([(chunk.lang, chunk.token_count) for chunk in chunks])
    if lang == "und":
        lang = detect_language_heuristic(markdown)
    doc = DocumentDraft(
        id=doc_id,
        kind=kind,
        data_source_id=data_source_id,
        area=area,
        title=title,
        title_hash=title_hash(title),
        url=f"https://www.notion.so/{doc_id.hex}",
        created_time=created,
        last_edited_time=edited,
        last_content_change=edited,
        markdown=markdown,
        content_hash=content_hash(markdown),
        token_count=token_count,
        lang=lang,
        is_stub=token_count < 40,
        status="active",
        discovered_via="fixture",
        chunks=chunks,
    )
    return doc


def _section_blocks(sections: list[tuple[str, list[str]]]) -> list[SourceBlock]:
    blocks: list[SourceBlock] = []
    cursor = 0
    for index, (heading, words) in enumerate(sections):
        heading_text = f"## {heading}"
        blocks.append(
            SourceBlock(
                block_id=UUID(int=1_000_000 + len(blocks) + index * 17),
                text=heading_text,
                plain=heading,
                heading_level=2,
                is_code=False,
                is_synced=False,
                is_property=False,
                created_time=None,
                start_char=cursor,
                end_char=cursor + len(heading_text),
            )
        )
        cursor += len(heading_text) + 1
        body = " ".join(words)
        blocks.append(
            SourceBlock(
                block_id=UUID(int=2_000_000 + len(blocks) + index * 19),
                text=body,
                plain=body,
                heading_level=None,
                is_code=False,
                is_synced=False,
                is_property=False,
                created_time=None,
                start_char=cursor,
                end_char=cursor + len(body),
            )
        )
        cursor += len(body) + 1
    return blocks


def _label(left: UUID, right: UUID, label: str) -> Label:
    from ndup.model import order_pair

    doc_a, doc_b = order_pair(left, right)
    return doc_a, doc_b, label
