"""Operations shared by the CLI commands."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from ndup.analysis.calibrate import p999_floor
from ndup.analysis.pipeline import analyze_documents, embed_documents, prepare_documents
from ndup.analysis.thresholds import apply_floor, apply_preset, thresholds_from_json
from ndup.config import Settings
from ndup.db.memory import MemoryStore
from ndup.embed.registry import build_provider, initial_thresholds, runtime_for
from ndup.errors import NdUpError
from ndup.model import IgnoreRule
from ndup.notion.client import NotionClient
from ndup.notion.sync import run_sync


def utcnow() -> datetime:
    return datetime.now(UTC)


def open_store(database_url: str) -> Any:
    if database_url.startswith("memory:"):
        return MemoryStore.open(database_url)
    from ndup.db.postgres import PostgresStore

    return PostgresStore.open(database_url)


def ensure_model(store: Any, settings: Settings) -> Any:
    runtime = runtime_for(settings.embedding.device, settings.embedding.runtime, settings.embedding.model)
    provider = build_provider(settings.embedding.model, settings.embedding.device, runtime)
    store.register_model(
        provider.model,
        provider.dims,
        runtime,
        initial_thresholds(provider.model),
    )
    model = store.get_model(provider.model, runtime)
    if model is None:
        raise NdUpError("Model registration did not persist")
    return model


def embed_store(store: Any, settings: Settings) -> tuple[int, int]:
    model = ensure_model(store, settings)
    docs = [doc for doc in store.list_documents() if doc.status == "active"]
    prepare_documents(docs)
    provider = build_provider(model.model, settings.embedding.device, model.runtime)
    cache = store.embedding_cache(model.id)
    embedded, hits = embed_documents(docs, provider, cache)
    store.put_embeddings(model.id, cache)
    for doc in docs:
        store.remember_source_hash(model.id, doc.id, _source_hash(doc), getattr(doc, "doc_vector", None))
    store.gc_embeddings(model.id)
    return embedded, hits


def _source_hash(doc: Any) -> bytes:
    from ndup.analysis.pipeline import embedding_source_hash

    return embedding_source_hash(doc)


def analyze_store(store: Any, settings: Settings, now: datetime | None = None) -> Any:
    moment = now or utcnow()
    model = ensure_model(store, settings)
    docs = store.list_documents()
    _attach_cached_vectors(store, model.id, docs)
    thresholds = apply_preset(thresholds_from_json(model.thresholds), settings.analysis.preset)
    floor = p999_floor([doc for doc in docs if doc.lang == "en" and not doc.is_stub])
    thresholds = apply_floor(thresholds, floor)
    ignored_docs, ignored_roots, ignored_pairs = _ignores(store.list_ignores())
    result = analyze_documents(
        docs,
        thresholds,
        now=moment,
        previous=store.load_previous_clusters(),
        ignored_docs=ignored_docs,
        ignored_roots=ignored_roots,
        ignored_pairs=ignored_pairs,
        active_window_days=settings.analysis.active_window_days,
    )
    store.save_analysis(result, moment)
    model.thresholds = thresholds.to_json()
    store.register_model(model.model, model.dims, model.runtime, model.thresholds)
    return result


def _attach_cached_vectors(store: Any, model_id: int, docs: list[Any]) -> None:
    cache = store.embedding_cache(model_id)
    for doc in docs:
        for chunk in doc.chunks:
            cached = cache.get(chunk.embed_hash)
            if cached is not None and chunk.vector is None:
                chunk.vector = cached


def _ignores(rules: list[IgnoreRule]) -> tuple[set[UUID], set[UUID], set[tuple[UUID, UUID]]]:
    docs: set[UUID] = set()
    roots: set[UUID] = set()
    pairs: set[tuple[UUID, UUID]] = set()
    for rule in rules:
        if rule.kind == "doc":
            docs.add(rule.a)
        elif rule.kind == "root":
            roots.add(rule.a)
        elif rule.kind == "pair" and rule.b is not None:
            pairs.add((rule.a, rule.b) if rule.a.int < rule.b.int else (rule.b, rule.a))
    return docs, roots, pairs


async def sync_store(
    store: Any,
    settings: Settings,
    token: str,
    *,
    discover_only: bool = False,
    root: UUID | None = None,
    now: datetime | None = None,
) -> dict[str, int]:
    excluded = _excluded_ids(settings, store)
    async with NotionClient(token, api_version=settings.notion.api_version) as client:
        return await run_sync(
            client,
            store,
            excluded=excluded,
            now=now,
            discover_only=discover_only,
            root=root,
        )


def sync_store_blocking(
    store: Any,
    settings: Settings,
    token: str,
    *,
    discover_only: bool = False,
    root: UUID | None = None,
) -> dict[str, int]:
    return asyncio.run(sync_store(store, settings, token, discover_only=discover_only, root=root))


def _excluded_ids(settings: Settings, store: Any) -> set[UUID]:
    found: set[UUID] = set()
    for root in store.list_roots():
        if root.excluded:
            found.add(root.id)
    for raw in settings.notion.exclude_roots:
        parsed = _maybe_uuid(raw)
        if parsed is not None:
            found.add(parsed)
    return found


def _maybe_uuid(value: str) -> UUID | None:
    token = value.rstrip("/").split("-")[-1]
    compact = value.replace("-", "")
    for candidate in (value, token, compact):
        try:
            return UUID(candidate)
        except ValueError:
            continue
    if len(compact) == 32:
        try:
            return UUID(hex=compact)
        except ValueError:
            return None
    return None
