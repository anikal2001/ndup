"""JSON and Rich renderings for ``ndup show``."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from ndup.analysis.types import ClusterRecord, PairRecord
from ndup.model import DocumentDraft
from ndup.web.redact import ContentMode, redact_text


def clusters_payload(
    clusters: list[ClusterRecord],
    docs: dict[UUID, DocumentDraft],
    *,
    mode: ContentMode,
    stub_groups: list[list[UUID]] | None = None,
    other_language_docs: int = 0,
    stub_docs: int = 0,
) -> dict[str, Any]:
    return {
        "clusters": [cluster_payload(cluster, docs, mode=mode) for cluster in clusters],
        "other_language_docs": other_language_docs,
        "stub_docs": stub_docs,
        "stub_title_duplicates": [[str(doc_id) for doc_id in group] for group in (stub_groups or [])],
    }


def cluster_payload(
    cluster: ClusterRecord,
    docs: dict[UUID, DocumentDraft],
    *,
    mode: ContentMode,
) -> dict[str, Any]:
    del mode
    return {
        "stable_key": str(cluster.stable_key),
        "label": cluster.label,
        "size": cluster.size,
        "areas": list(cluster.areas),
        "active_copies": cluster.active_copies,
        "oldest": _end_payload(cluster.oldest_doc, docs, "OLDEST"),
        "newest": _end_payload(cluster.newest_doc, docs, "NEWEST"),
        "members": [
            {
                "id": str(member.doc_id),
                "title": None if docs.get(member.doc_id) is None else docs[member.doc_id].title,
                "area": None if docs.get(member.doc_id) is None else docs[member.doc_id].area,
                "url": None if docs.get(member.doc_id) is None else docs[member.doc_id].url,
                "relation": member.relation_to_cluster,
                "is_satellite": member.is_satellite,
            }
            for member in cluster.members
        ],
        "passage_origin": None
        if cluster.passage_origin_doc is None
        else {
            "doc_id": str(cluster.passage_origin_doc),
            "at": _iso(cluster.passage_origin_time),
        },
    }


def pair_payload(
    pair: PairRecord,
    docs: dict[UUID, DocumentDraft],
    *,
    mode: ContentMode,
) -> dict[str, Any]:
    left = docs.get(pair.doc_a)
    right = docs.get(pair.doc_b)
    return {
        "doc_a": str(pair.doc_a),
        "doc_b": str(pair.doc_b),
        "relation": pair.relation,
        "score": pair.score,
        "doc_sim": pair.doc_sim,
        "cov_ab": pair.cov_ab,
        "cov_ba": pair.cov_ba,
        "lex_jaccard": pair.lex_jaccard,
        "passages": [
            {
                "chunk_a": passage.chunk_a,
                "chunk_b": passage.chunk_b,
                "sim": passage.sim,
                "text_a": _chunk_text(left, passage.chunk_a, mode),
                "text_b": _chunk_text(right, passage.chunk_b, mode),
            }
            for passage in pair.passages
        ],
    }


def doc_payload(doc: DocumentDraft, *, mode: ContentMode) -> dict[str, Any]:
    return {
        "id": str(doc.id),
        "title": doc.title,
        "area": doc.area,
        "url": doc.url,
        "kind": doc.kind,
        "lang": doc.lang,
        "status": doc.status,
        "is_stub": doc.is_stub,
        "token_count": doc.token_count,
        "created_time": _iso(doc.created_time),
        "last_edited_time": _iso(doc.last_edited_time),
        "last_content_change": _iso(doc.last_content_change),
        "markdown": redact_text(doc.markdown, mode),
    }


def export_metadata(clusters: list[ClusterRecord], docs: dict[UUID, DocumentDraft]) -> dict[str, Any]:
    payload = clusters_payload(clusters, docs, mode="metadata")
    for cluster in payload["clusters"]:
        for member in cluster["members"]:
            member.pop("title", None)
            # Titles are allowed in metadata export. Keep them.
        cluster["members"] = cluster["members"]
    return {
        "clusters": [
            {
                "stable_key": cluster["stable_key"],
                "label": cluster["label"],
                "areas": cluster["areas"],
                "oldest": cluster["oldest"],
                "newest": cluster["newest"],
                "members": [
                    {
                        "id": member["id"],
                        "title": member["title"],
                        "url": member["url"],
                        "area": member["area"],
                        "relation": member["relation"],
                        "is_satellite": member["is_satellite"],
                    }
                    for member in cluster["members"]
                ],
            }
            for cluster in payload["clusters"]
        ]
    }


def _end_payload(doc_id: UUID | None, docs: dict[UUID, DocumentDraft], badge: str) -> dict[str, Any] | None:
    if doc_id is None:
        return None
    doc = docs.get(doc_id)
    if doc is None:
        return {"id": str(doc_id), "badge": badge}
    moment = doc.created_time if badge == "OLDEST" else (doc.last_content_change or doc.last_edited_time)
    return {
        "id": str(doc.id),
        "title": doc.title,
        "url": doc.url,
        "badge": badge,
        "at": _iso(moment),
        "any_edit": badge == "NEWEST" and doc.last_content_change is None,
    }


def _chunk_text(doc: DocumentDraft | None, ident: int, mode: ContentMode) -> str | None:
    if doc is None or mode == "metadata":
        return None
    for chunk in doc.chunks:
        if chunk.ord == ident or chunk.chunk_id == ident:
            return redact_text(chunk.text, mode)
    return None


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()
