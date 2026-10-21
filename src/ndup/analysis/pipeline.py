"""In-memory analysis used by the CLI and the golden-set gate.

Exact kNN is the Phase 1 default. Postgres can also answer kNN from the HNSW
indexes created at model registration; both paths share this evidence code.
"""

from __future__ import annotations

import hashlib
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

import numpy as np

from ndup.analysis.clustering import build_clusters
from ndup.analysis.provenance import (
    active_copy_count,
    cluster_label,
    newest_doc,
    oldest_doc,
    passage_origin,
)
from ndup.analysis.relations import EvidenceView, classify
from ndup.analysis.stable_keys import assign_stable_keys, member_hash
from ndup.analysis.templates import apply_density_templates
from ndup.analysis.thresholds import Thresholds
from ndup.analysis.types import (
    AnalysisResult,
    ClusterRecord,
    PairRecord,
    PassageMatch,
    PrevCluster,
)
from ndup.embed.fake_provider import l2_normalize
from ndup.metrics import (
    analysis_seconds,
    chunks_embedded,
    clusters_current,
    embed_cache_hits,
    pairs_current,
)
from ndup.model import ChunkDraft, DocumentDraft, order_pair
from ndup.text.boilerplate import apply_hash_boilerplate, apply_row_template
from ndup.text.shingles import containment, jaccard, word_shingles


@dataclass
class _SpanBest:
    tokens: int
    best_sim: float


@dataclass
class _PassageRaw:
    chunk_a: int
    chunk_b: int
    sim: float


@dataclass
class _PairRaw:
    doc_a: UUID
    doc_b: UUID
    doc_sim: float
    spans_a: list[_SpanBest]
    spans_b: list[_SpanBest]
    lex_jaccard: float
    lex_cont_ab: float
    lex_cont_ba: float
    mutual_sims: list[float]
    shared_hash: bool
    weighted_a: int
    weighted_b: int
    passages: list[_PassageRaw] = field(default_factory=list)


@dataclass
class PreparedCorpus:
    docs: dict[UUID, DocumentDraft]
    pairs: list[_PairRaw]
    doc_neighbors: dict[UUID, list[tuple[UUID, float]]]
    eligible_ids: list[UUID]


def prepare_documents(docs: list[DocumentDraft]) -> None:
    """Hash and row-template boilerplate. Call this before embedding."""
    apply_hash_boilerplate(docs)
    apply_row_template(docs)


def embed_documents(
    docs: list[DocumentDraft],
    provider: Any,
    cache: dict[bytes, np.ndarray] | None = None,
) -> tuple[int, int]:
    """Embed unique ``embed_hash`` values. Returns ``(embedded, cache_hits)``."""
    store = cache if cache is not None else {}
    needed: dict[bytes, str] = {}
    hits = 0
    for doc in docs:
        for chunk in doc.chunks:
            if chunk.weight <= 0 or chunk.is_synced:
                cached = store.get(chunk.embed_hash)
                if cached is not None:
                    chunk.vector = cached
                continue
            cached = store.get(chunk.embed_hash)
            if cached is not None:
                chunk.vector = cached
                hits += 1
                continue
            needed.setdefault(chunk.embed_hash, chunk.embed_text)
    embedded = 0
    if needed:
        hashes = list(needed)
        vectors = provider.embed([needed[digest] for digest in hashes])
        for digest, vector in zip(hashes, vectors, strict=True):
            store[digest] = np.asarray(vector, dtype=np.float32)
        embedded = len(hashes)
        chunks_embedded.inc(embedded)
    embed_cache_hits.inc(hits)
    for doc in docs:
        for chunk in doc.chunks:
            cached = store.get(chunk.embed_hash)
            if cached is not None:
                chunk.vector = cached
    for doc in docs:
        _attach_doc_vector(doc)
    return embedded, hits


def embedding_source_hash(doc: DocumentDraft) -> bytes:
    parts = [
        f"{chunk.embed_hash.hex()}:{chunk.weight:.4f}:{chunk.token_count}"
        for chunk in sorted(doc.chunks, key=lambda item: item.ord)
    ]
    return hashlib.sha256("\n".join(parts).encode()).digest()


def analyze_documents(
    docs: list[DocumentDraft],
    thresholds: Thresholds,
    *,
    now: datetime,
    previous: list[PrevCluster] | None = None,
    ignored_docs: set[UUID] | None = None,
    ignored_roots: set[UUID] | None = None,
    ignored_pairs: set[tuple[UUID, UUID]] | None = None,
    active_window_days: int = 90,
) -> AnalysisResult:
    """Run density detection, evidence, relations, and clustering."""
    with analysis_seconds.labels(stage="density").time():
        apply_density_templates(docs, thresholds.tau_chunk)
        for doc in docs:
            _attach_doc_vector(doc)
    eligible = [doc for doc in docs if _eligible(doc, ignored_docs or set(), ignored_roots or set())]
    with analysis_seconds.labels(stage="candidates").time():
        prepared = _prepare(eligible)
    with analysis_seconds.labels(stage="evidence").time():
        pairs = [
            pair
            for raw in prepared.pairs
            if (
                materialized := _materialize(
                    raw,
                    prepared.doc_neighbors,
                    thresholds,
                    ignored_pairs or set(),
                )
            )
            is not None
            for pair in [materialized]
        ]
    with analysis_seconds.labels(stage="cluster").time():
        grouped = build_clusters(pairs)
        keys, events, notes = assign_stable_keys(
            [core for core, _members in grouped],
            previous or [],
            now,
        )
        by_id = {doc.id: doc for doc in docs}
        clusters: list[ClusterRecord] = []
        for (core, members), key in zip(grouped, keys, strict=True):
            member_docs = [by_id[member.doc_id] for member in members if member.doc_id in by_id]
            core_docs = [by_id[doc_id] for doc_id in core if doc_id in by_id]
            oldest = oldest_doc(member_docs)
            newest = newest_doc(member_docs)
            origin_doc, origin_time = passage_origin(pairs, {member.doc_id for member in members}, by_id)
            areas = sorted({doc.area for doc in member_docs if doc.area})
            clusters.append(
                ClusterRecord(
                    stable_key=key,
                    label=cluster_label(core_docs or member_docs),
                    size=len(members),
                    member_hash=member_hash([member.doc_id for member in members]),
                    oldest_doc=None if oldest is None else oldest.id,
                    newest_doc=None if newest is None else newest.id,
                    active_copies=active_copy_count(member_docs, now, active_window_days),
                    areas=areas,
                    members=members,
                    passage_origin_doc=origin_doc,
                    passage_origin_time=origin_time,
                    created_at=now,
                )
            )
    stub_groups = _stub_groups(docs)
    other_language = sum(1 for doc in docs if doc.status == "active" and doc.lang not in {None, "en", "und"})
    stubs = sum(1 for doc in docs if doc.is_stub and doc.status == "active")
    relation_counts: dict[str, int] = defaultdict(int)
    for pair in pairs:
        relation_counts[pair.relation] += 1
    pairs_current.set(len(pairs))
    clusters_current.set(len(clusters))
    return AnalysisResult(
        pairs=pairs,
        clusters=clusters,
        events=events,
        stub_groups=stub_groups,
        other_language_docs=other_language,
        stub_docs=stubs,
        stats={
            "docs": len(eligible),
            "pairs": len(pairs),
            "clusters": len(clusters),
            "relations": dict(relation_counts),
            "other_language_docs": other_language,
            "stub_docs": stubs,
            "stub_title_groups": len(stub_groups),
        },
        thresholds=thresholds.to_json(),
        review_notes=notes,
    )


def _eligible(doc: DocumentDraft, ignored_docs: set[UUID], ignored_roots: set[UUID]) -> bool:
    if doc.status != "active" or doc.is_stub or doc.lang != "en":
        return False
    if doc.id in ignored_docs:
        return False
    if doc.root_id is not None and doc.root_id in ignored_roots:
        return False
    return any(chunk.weight > 0 and chunk.vector is not None for chunk in doc.chunks)


def _prepare(docs: list[DocumentDraft]) -> PreparedCorpus:
    by_id = {doc.id: doc for doc in docs}
    ids = [doc.id for doc in docs]
    doc_vectors, doc_index = _doc_matrix(docs)
    neighbors = _doc_neighbors(ids, doc_vectors, doc_index)
    chunk_records = _weighted_chunks(docs)
    raw_pairs: dict[tuple[UUID, UUID], _PairRaw] = {}
    _add_mutual_chunks(chunk_records, raw_pairs, by_id)
    _add_hash_pairs(docs, raw_pairs, by_id)
    _add_doc_neighbor_pairs(neighbors, raw_pairs, by_id, doc_index, doc_vectors)
    return PreparedCorpus(
        docs=by_id,
        pairs=list(raw_pairs.values()),
        doc_neighbors=neighbors,
        eligible_ids=ids,
    )


def _materialize(
    raw: _PairRaw,
    neighbors: dict[UUID, list[tuple[UUID, float]]],
    thresholds: Thresholds,
    ignored_pairs: set[tuple[UUID, UUID]],
) -> PairRecord | None:
    if (raw.doc_a, raw.doc_b) in ignored_pairs:
        return None
    cov_ab = _coverage(raw.spans_a, thresholds.tau_chunk)
    cov_ba = _coverage(raw.spans_b, thresholds.tau_chunk)
    kept = [sim for sim in raw.mutual_sims if sim >= thresholds.tau_chunk]
    median = float(statistics.median(kept)) if kept else 0.0
    evidence = EvidenceView(
        cov_ab=cov_ab,
        cov_ba=cov_ba,
        lex_jaccard=raw.lex_jaccard,
        lex_cont_ab=raw.lex_cont_ab,
        lex_cont_ba=raw.lex_cont_ba,
        doc_sim=raw.doc_sim,
        median_match=median,
        shared_hash=raw.shared_hash,
        weighted_chunks_a=raw.weighted_a,
        weighted_chunks_b=raw.weighted_b,
    )
    doc_hit = _listed(neighbors.get(raw.doc_a, []), raw.doc_b, thresholds.tau_doc) or _listed(
        neighbors.get(raw.doc_b, []), raw.doc_a, thresholds.tau_doc
    )
    chunk_hit = bool(kept)
    if not (doc_hit or chunk_hit or raw.shared_hash):
        return None
    relation = classify(evidence, thresholds)
    if relation is None:
        return None
    score = max(raw.lex_jaccard, (cov_ab + cov_ba) / 2, raw.doc_sim)
    passages = [
        PassageMatch(chunk_a=item.chunk_a, chunk_b=item.chunk_b, sim=item.sim, lex=raw.lex_jaccard)
        for item in raw.passages
        if item.sim >= thresholds.tau_chunk
    ]
    return PairRecord(
        doc_a=raw.doc_a,
        doc_b=raw.doc_b,
        doc_sim=raw.doc_sim,
        cov_ab=cov_ab,
        cov_ba=cov_ba,
        lex_jaccard=raw.lex_jaccard,
        lex_cont_ab=raw.lex_cont_ab,
        lex_cont_ba=raw.lex_cont_ba,
        score=score,
        relation=relation,
        median_match=median,
        passages=passages,
    )


def _coverage(spans: list[_SpanBest], tau: float) -> float:
    total = sum(span.tokens for span in spans)
    if total <= 0:
        return 0.0
    hit = sum(span.tokens for span in spans if span.best_sim >= tau)
    return hit / total


def _listed(neighbors: list[tuple[UUID, float]], other: UUID, tau: float) -> bool:
    kept = 0
    for doc_id, sim in neighbors:
        if sim < tau:
            break
        if kept >= 20:
            break
        if doc_id == other:
            return True
        kept += 1
    return False


def _doc_matrix(docs: list[DocumentDraft]) -> tuple[np.ndarray | None, dict[UUID, int]]:
    vectors: list[np.ndarray] = []
    index: dict[UUID, int] = {}
    for doc in docs:
        vector = doc_vector(doc)
        if vector is None:
            continue
        index[doc.id] = len(vectors)
        vectors.append(vector)
    if not vectors:
        return None, index
    return np.stack(vectors), index


def doc_vector(doc: DocumentDraft) -> np.ndarray | None:
    total: np.ndarray | None = None
    weight = 0.0
    for chunk in doc.chunks:
        if chunk.vector is None or chunk.weight <= 0 or chunk.is_synced:
            continue
        token_weight = chunk.weight * max(chunk.token_count, 1)
        term = np.asarray(chunk.vector, dtype=np.float32) * np.float32(token_weight)
        total = term if total is None else total + term
        weight += token_weight
    if total is None or weight <= 0:
        return None
    return l2_normalize(total / np.float32(weight))


def _attach_doc_vector(doc: DocumentDraft) -> None:
    doc.doc_vector = doc_vector(doc)


def _stored_doc_vector(doc: DocumentDraft) -> np.ndarray | None:
    vector = getattr(doc, "doc_vector", None)
    if isinstance(vector, np.ndarray):
        return vector
    return doc_vector(doc)


def _doc_neighbors(
    ids: list[UUID],
    matrix: np.ndarray | None,
    index: dict[UUID, int],
) -> dict[UUID, list[tuple[UUID, float]]]:
    neighbors: dict[UUID, list[tuple[UUID, float]]] = {doc_id: [] for doc_id in ids}
    if matrix is None or matrix.shape[0] == 0:
        return neighbors
    sims = matrix @ matrix.T
    reverse = {position: doc_id for doc_id, position in index.items()}
    width = sims.shape[0]
    k = min(20, width - 1) if width > 1 else 0
    for position in range(width):
        row = sims[position].copy()
        row[position] = -1.0
        if k <= 0:
            continue
        if k == width - 1:
            chosen = np.argsort(row)[::-1]
        else:
            picked = np.argpartition(row, -k)[-k:]
            chosen = picked[np.argsort(row[picked])[::-1]]
        doc_id = reverse[position]
        neighbors[doc_id] = [(reverse[int(other)], float(row[int(other)])) for other in chosen if row[int(other)] > 0]
    return neighbors


def _weighted_chunks(
    docs: list[DocumentDraft],
) -> list[tuple[UUID, ChunkDraft, np.ndarray]]:
    rows: list[tuple[UUID, ChunkDraft, np.ndarray]] = []
    for doc in docs:
        for chunk in doc.chunks:
            if chunk.weight <= 0 or chunk.is_synced or chunk.vector is None:
                continue
            rows.append((doc.id, chunk, np.asarray(chunk.vector, dtype=np.float32)))
    return rows


def _add_mutual_chunks(
    records: list[tuple[UUID, ChunkDraft, np.ndarray]],
    raw_pairs: dict[tuple[UUID, UUID], _PairRaw],
    docs: dict[UUID, DocumentDraft],
) -> None:
    if len(records) < 2:
        return
    matrix = np.stack([row[2] for row in records])
    owners = [row[0] for row in records]
    best_index = np.full(len(records), -1, dtype=np.int32)
    best_sim = np.full(len(records), -1.0, dtype=np.float32)
    block = 256
    for start in range(0, len(records), block):
        end = min(len(records), start + block)
        sims = matrix[start:end] @ matrix.T
        for offset, row_index in enumerate(range(start, end)):
            same = [owners[col] == owners[row_index] for col in range(len(records))]
            sims[offset, same] = -1.0
        choice = np.argmax(sims, axis=1)
        values = sims[np.arange(end - start), choice]
        best_index[start:end] = choice
        best_sim[start:end] = values
    for left in range(len(records)):
        right = int(best_index[left])
        if right <= left or best_sim[left] < 0.4:
            continue
        if int(best_index[right]) != left:
            continue
        doc_a, doc_b = order_pair(owners[left], owners[right])
        raw = _ensure_raw(raw_pairs, doc_a, doc_b, docs)
        raw.mutual_sims.append(float(best_sim[left]))
        chunk_left = records[left][1]
        chunk_right = records[right][1]
        if owners[left] == doc_a:
            raw.passages.append(_PassageRaw(chunk_left.ord, chunk_right.ord, float(best_sim[left])))
        else:
            raw.passages.append(_PassageRaw(chunk_right.ord, chunk_left.ord, float(best_sim[left])))


def _add_hash_pairs(
    docs: list[DocumentDraft],
    raw_pairs: dict[tuple[UUID, UUID], _PairRaw],
    by_id: dict[UUID, DocumentDraft],
) -> None:
    owners: dict[bytes, set[UUID]] = defaultdict(set)
    for doc in docs:
        for chunk in doc.chunks:
            if chunk.weight > 0 and not chunk.is_boilerplate and not chunk.is_synced:
                owners[chunk.text_hash].add(doc.id)
    for doc_ids in owners.values():
        ordered = sorted(doc_ids, key=lambda item: item.int)
        if len(ordered) > 40:
            continue
        for left in range(len(ordered)):
            for right in range(left + 1, len(ordered)):
                raw = _ensure_raw(raw_pairs, ordered[left], ordered[right], by_id)
                raw.shared_hash = True


def _add_doc_neighbor_pairs(
    neighbors: dict[UUID, list[tuple[UUID, float]]],
    raw_pairs: dict[tuple[UUID, UUID], _PairRaw],
    docs: dict[UUID, DocumentDraft],
    index: dict[UUID, int],
    matrix: np.ndarray | None,
) -> None:
    del index, matrix
    for doc_id, ranked in neighbors.items():
        for other, _sim in ranked[:40]:
            if other == doc_id:
                continue
            doc_a, doc_b = order_pair(doc_id, other)
            _ensure_raw(raw_pairs, doc_a, doc_b, docs)


def _ensure_raw(
    raw_pairs: dict[tuple[UUID, UUID], _PairRaw],
    doc_a: UUID,
    doc_b: UUID,
    docs: dict[UUID, DocumentDraft],
) -> _PairRaw:
    key = (doc_a, doc_b)
    existing = raw_pairs.get(key)
    if existing is not None:
        return existing
    left = docs[doc_a]
    right = docs[doc_b]
    left_vec = _stored_doc_vector(left)
    right_vec = _stored_doc_vector(right)
    if left_vec is None or right_vec is None:
        doc_sim = 0.0
    else:
        doc_sim = float(left_vec @ right_vec)
    shingle_a = word_shingles(left.markdown or _fallback_text(left))
    shingle_b = word_shingles(right.markdown or _fallback_text(right))
    raw = _PairRaw(
        doc_a=doc_a,
        doc_b=doc_b,
        doc_sim=doc_sim,
        spans_a=_span_bests(left, right),
        spans_b=_span_bests(right, left),
        lex_jaccard=jaccard(shingle_a, shingle_b),
        lex_cont_ab=containment(shingle_a, shingle_b),
        lex_cont_ba=containment(shingle_b, shingle_a),
        mutual_sims=[],
        shared_hash=_shared_hash(left, right),
        weighted_a=_weighted_count(left),
        weighted_b=_weighted_count(right),
    )
    raw_pairs[key] = raw
    return raw


def _span_bests(source: DocumentDraft, other: DocumentDraft) -> list[_SpanBest]:
    source_chunks = [chunk for chunk in source.chunks if chunk.weight > 0 and chunk.vector is not None]
    other_chunks = [chunk for chunk in other.chunks if chunk.weight > 0 and chunk.vector is not None]
    if not source_chunks:
        return []
    if not other_chunks:
        return [_SpanBest(tokens=span.tokens, best_sim=0.0) for chunk in source_chunks for span in chunk.spans]
    other_matrix = np.stack([np.asarray(chunk.vector, dtype=np.float32) for chunk in other_chunks])
    best_by_span: dict[str, _SpanBest] = {}
    for chunk in source_chunks:
        vector = np.asarray(chunk.vector, dtype=np.float32)
        sims = other_matrix @ vector
        best = float(sims.max()) if sims.size else 0.0
        for span in chunk.spans:
            current = best_by_span.get(span.span_id)
            if current is None or best > current.best_sim:
                best_by_span[span.span_id] = _SpanBest(tokens=span.tokens, best_sim=best)
    return list(best_by_span.values())


def _shared_hash(left: DocumentDraft, right: DocumentDraft) -> bool:
    left_hashes = {
        chunk.text_hash
        for chunk in left.chunks
        if chunk.weight > 0 and not chunk.is_boilerplate and not chunk.is_synced
    }
    right_hashes = {
        chunk.text_hash
        for chunk in right.chunks
        if chunk.weight > 0 and not chunk.is_boilerplate and not chunk.is_synced
    }
    return bool(left_hashes & right_hashes)


def _weighted_count(doc: DocumentDraft) -> int:
    return sum(1 for chunk in doc.chunks if chunk.weight > 0 and not chunk.is_synced)


def _fallback_text(doc: DocumentDraft) -> str:
    return "\n".join(chunk.text for chunk in doc.chunks)


def _stub_groups(docs: list[DocumentDraft]) -> list[list[UUID]]:
    grouped: dict[bytes, list[UUID]] = defaultdict(list)
    for doc in docs:
        if doc.is_stub and doc.status == "active" and doc.title_hash is not None:
            grouped[doc.title_hash].append(doc.id)
    return [sorted(doc_ids, key=lambda item: item.int) for doc_ids in grouped.values() if len(doc_ids) >= 2]
