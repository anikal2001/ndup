"""Relation rules from architecture v3 §9.3, including the short-doc rule."""

from __future__ import annotations

from dataclasses import dataclass

from ndup.analysis.thresholds import Thresholds


@dataclass(frozen=True)
class EvidenceView:
    cov_ab: float
    cov_ba: float
    lex_jaccard: float
    lex_cont_ab: float
    lex_cont_ba: float
    doc_sim: float
    median_match: float
    shared_hash: bool
    weighted_chunks_a: int
    weighted_chunks_b: int


def classify(evidence: EvidenceView, thresholds: Thresholds) -> str | None:
    """Return the relation, or None when the pair is not worth storing."""
    near = evidence.lex_jaccard >= 0.8 or (
        evidence.cov_ab >= 0.9 and evidence.cov_ba >= 0.9 and evidence.median_match >= thresholds.tau_chunk + 0.05
    )
    duplicate = evidence.cov_ab >= 0.6 and evidence.cov_ba >= 0.6
    subset = (
        (evidence.cov_ab >= 0.7 and evidence.cov_ba < 0.4)
        or (evidence.cov_ba >= 0.7 and evidence.cov_ab < 0.4)
        or evidence.lex_cont_ab >= 0.7
        or evidence.lex_cont_ba >= 0.7
    )
    if near:
        relation: str | None = "near_copy"
    elif duplicate:
        relation = "duplicate"
    elif subset:
        relation = "subset"
    elif max(evidence.cov_ab, evidence.cov_ba) >= 0.25:
        relation = "partial"
    elif evidence.doc_sim >= thresholds.tau_doc:
        relation = "related"
    else:
        return None

    short = (
        evidence.weighted_chunks_a <= thresholds.short_doc_chunks
        or evidence.weighted_chunks_b <= thresholds.short_doc_chunks
    )
    if relation in {"near_copy", "duplicate"} and short:
        lexical = evidence.lex_cont_ab >= 0.5 or evidence.lex_cont_ba >= 0.5 or evidence.shared_hash
        median_ok = evidence.median_match >= thresholds.tau_chunk + thresholds.short_doc_delta
        if not (median_ok and lexical):
            return "related"
    return relation


def subset_direction(evidence: EvidenceView) -> str | None:
    """Return ``a`` when doc_a is contained in doc_b, ``b`` for the reverse."""
    if evidence.cov_ab >= 0.7 and evidence.cov_ba < 0.4:
        return "a"
    if evidence.cov_ba >= 0.7 and evidence.cov_ab < 0.4:
        return "b"
    if evidence.lex_cont_ab >= 0.7 and evidence.lex_cont_ab > evidence.lex_cont_ba:
        return "a"
    if evidence.lex_cont_ba >= 0.7 and evidence.lex_cont_ba > evidence.lex_cont_ab:
        return "b"
    return None
