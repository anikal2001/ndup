"""WP1.4 renderer, WP1.5 chunker, WP0.4 and WP1.6 embeddings."""

from __future__ import annotations

import socket
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ndup.analysis.pipeline import embed_documents
from ndup.db.memory import MemoryStore
from ndup.doctor import provider_imports_http
from ndup.embed.fake_provider import FAKE_MODEL, FakeEmbeddingProvider
from ndup.embed.registry import hnsw_index_sql, runtime_for
from ndup.model import DocumentDraft, SourceBlock
from ndup.notion.render import render_page
from ndup.text.boilerplate import template_title
from ndup.text.chunker import chunk_blocks, content_hash, embed_hash
from ndup.text.langid import detect_language_heuristic
from ndup.text.tokens import simple_token_count


def _block(kind: str, text: str, block_id: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "object": "block",
        "id": block_id,
        "type": kind,
        "has_children": False,
    }
    rich = [{"plain_text": text, "annotations": {"bold": True}}]
    if kind == "code":
        payload[kind] = {"rich_text": [{"plain_text": text, "annotations": {}}], "language": "python"}
    elif kind == "child_page":
        payload[kind] = {"title": text}
    else:
        payload[kind] = {"rich_text": rich}
    return payload


@pytest.mark.unit
def test_renderer_golden_and_property_has_no_block_id() -> None:
    page_id = "11111111-1111-1111-1111-111111111111"
    page = {
        "id": page_id,
        "url": "https://www.notion.so/example",
        "created_time": "2024-01-01T00:00:00.000Z",
        "properties": {
            "Name": {"type": "title", "title": [{"plain_text": "Row", "annotations": {}}]},
            "Notes": {
                "type": "rich_text",
                "rich_text": [{"plain_text": "owned by ada", "annotations": {}}],
            },
        },
    }
    blocks = [
        _block("heading_1", "Overview", "22222222-2222-2222-2222-222222222222"),
        _block("paragraph", "The body.", "33333333-3333-3333-3333-333333333333"),
        _block("code", "print(1)", "44444444-4444-4444-4444-444444444444"),
        _block("child_page", "Nested", "55555555-5555-5555-5555-555555555555"),
    ]
    rendered = render_page(page, blocks)
    assert "owned by ada" in rendered.markdown
    assert "# Overview" in rendered.markdown or "Overview" in rendered.markdown
    assert "The body." in rendered.markdown
    assert "print(1)" in rendered.markdown
    property_blocks = [block for block in rendered.source_blocks if block.is_property]
    assert property_blocks
    assert all(block.block_id is None for block in property_blocks)
    assert property_blocks[0].start_char == 0
    assert UUID("55555555-5555-5555-5555-555555555555") in rendered.child_page_ids


@pytest.mark.unit
@settings(max_examples=25, deadline=None)
@given(
    st.lists(
        st.lists(st.text(alphabet="abcdefg ", min_size=1, max_size=12), min_size=1, max_size=6),
        min_size=1,
        max_size=8,
    )
)
def test_chunker_invariants(groups: list[list[str]]) -> None:
    blocks: list[SourceBlock] = []
    cursor = 0
    for index, words in enumerate(groups):
        text = " ".join(word.strip() or "a" for word in words)
        blocks.append(
            SourceBlock(
                block_id=UUID(int=index + 1),
                text=text,
                plain=text,
                heading_level=None,
                is_code=False,
                is_synced=False,
                is_property=False,
                created_time=None,
                start_char=cursor,
                end_char=cursor + len(text),
            )
        )
        cursor += len(text) + 2

    def count(text: str) -> int:
        return len(text.split())

    chunks = chunk_blocks(blocks, title="Title", drop_title=False, count_tokens=count, target=4, max_tokens=8)
    for chunk in chunks:
        assert chunk.token_count <= 8 or " " not in chunk.text
        assert chunk.start_char <= chunk.end_char
    seen: dict[str, int] = {}
    for chunk in chunks:
        for span in chunk.spans:
            seen[span.span_id] = seen.get(span.span_id, 0) + 1
    assert all(count_spans <= 2 for count_spans in seen.values())
    blob = "\n".join(chunk.text for chunk in chunks)
    for block in blocks:
        assert block.text.split()[0] in blob


@pytest.mark.unit
def test_hashes_title_suppression_and_languages() -> None:
    block = SourceBlock(
        block_id=uuid4(),
        text="alpha beta gamma delta",
        plain="alpha beta gamma delta",
        heading_level=None,
        is_code=False,
        is_synced=False,
        is_property=False,
        created_time=datetime(2024, 1, 1, tzinfo=UTC),
        start_char=0,
        end_char=22,
    )
    kept = chunk_blocks([block], title="Weekly status", drop_title=False, count_tokens=simple_token_count)
    dropped = chunk_blocks([block], title="Weekly status", drop_title=True, count_tokens=simple_token_count)
    assert kept[0].text_hash == content_hash(kept[0].text)
    assert kept[0].embed_hash == embed_hash(kept[0].embed_text)
    assert "Weekly status" in kept[0].embed_text
    assert "Weekly status" not in dropped[0].embed_text
    assert kept[0].embed_hash != dropped[0].embed_hash
    assert template_title(["Weekly status"] * 5 + ["other"]) == "Weekly status"
    assert detect_language_heuristic("el documento de la oficina y los archivos") == "es"
    assert detect_language_heuristic("这是一个中文页面") == "zh"
    assert detect_language_heuristic("the notes and the page of the team") == "en"


@pytest.mark.unit
def test_embed_contract_cache_and_model_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = FakeEmbeddingProvider()
    first = provider.embed(["alpha beta", "alpha beta"])
    again = provider.embed(["alpha beta"])
    assert first.shape == (2, provider.dims)
    assert abs(float((first[0] ** 2).sum()) - 1.0) < 1e-5
    assert (first[0] == again[0]).all()
    assert provider.model == FAKE_MODEL
    assert provider_imports_http() == []

    def blocked(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("embedding opened a socket")

    monkeypatch.setattr(socket, "socket", blocked)
    provider.embed(["offline sentence"])

    left = _doc("same", "Body text alpha beta gamma delta epsilon zeta eta theta")
    right = _doc("same", "Body text alpha beta gamma delta epsilon zeta eta theta")
    embedded, hits = embed_documents([left, right], provider, {})
    assert embedded == 1
    assert hits == 0
    cache = {chunk.embed_hash: chunk.vector for chunk in left.chunks if chunk.vector is not None}
    embedded_again, hits_again = embed_documents([left, right], provider, cache)
    assert embedded_again == 0
    assert hits_again == 2
    retitled = _doc("renamed title", "Body text alpha beta gamma delta epsilon zeta eta theta")
    changed, kept = embed_documents([retitled, right], provider, cache)
    assert changed == len(retitled.chunks)
    assert kept >= 1

    store = MemoryStore()
    cpu = store.register_model("BAAI/bge-base-en-v1.5", 768, runtime_for("cpu"), {})
    gpu = store.register_model("BAAI/bge-base-en-v1.5", 768, runtime_for("cuda"), {})
    assert cpu != gpu
    sql = hnsw_index_sql(table="embedding", model_id=cpu, dims=768)
    assert "halfvec(768)" in sql
    assert f"model_id = {cpu}" in sql


def _doc(title: str, text: str) -> DocumentDraft:
    block = SourceBlock(
        block_id=uuid4(),
        text=text,
        plain=text,
        heading_level=None,
        is_code=False,
        is_synced=False,
        is_property=False,
        created_time=None,
        start_char=0,
        end_char=len(text),
    )
    chunks = chunk_blocks([block], title=title, drop_title=False, count_tokens=simple_token_count)
    return DocumentDraft(id=uuid4(), kind="page", title=title, lang="en", chunks=chunks, status="active")
