"""Fit tau on labeled pairs. The percentile is only a floor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ndup.analysis.relations import EvidenceView, classify
from ndup.analysis.thresholds import Thresholds, apply_floor
from ndup.model import DocumentDraft


@dataclass(frozen=True)
class LabeledExample:
    evidence: EvidenceView
    label: str
    doc_sim: float
    mutual: bool
    shared_hash: bool


def fit_thresholds(
    examples: list[LabeledExample],
    *,
    seed: int = 0,
    subsample: float = 1.0,
    short_doc_chunks: int = 2,
) -> Thresholds:
    """Maximize F1 of duplicate + near_copy. Ties pick the middle of the plateau."""
    chosen = _subsample(examples, seed, subsample)
    chunk_grid = [round(value, 2) for value in _frange(0.50, 0.90, 0.02)]
    doc_grid = [round(value, 2) for value in _frange(0.50, 0.90, 0.02)]
    deltas = (0.03, 0.05, 0.07)
    best_f1 = -1.0
    plateau: list[tuple[float, float, float, float]] = []
    for tau_chunk in chunk_grid:
        for tau_doc in doc_grid:
            for delta in deltas:
                candidate = Thresholds(
                    tau_chunk=tau_chunk,
                    tau_doc=tau_doc,
                    short_doc_delta=delta,
                    short_doc_chunks=short_doc_chunks,
                    source="labeled",
                    n_labels=len(chosen),
                )
                score = _f1(chosen, candidate)
                if score > best_f1 + 1e-9:
                    best_f1 = score
                    plateau = [(tau_chunk, tau_doc, delta, score)]
                elif abs(score - best_f1) <= 1e-9:
                    plateau.append((tau_chunk, tau_doc, delta, score))
    if not plateau:
        return Thresholds(0.72, 0.68, source="labeled", n_labels=len(chosen), f1=0.0)
    midpoint = plateau[len(plateau) // 2]
    return Thresholds(
        tau_chunk=midpoint[0],
        tau_doc=midpoint[1],
        short_doc_delta=midpoint[2],
        short_doc_chunks=short_doc_chunks,
        source="labeled",
        n_labels=len(chosen),
        f1=midpoint[3],
    )


def p999_floor(
    docs: list[DocumentDraft],
    *,
    sample_pairs: int = 5000,
    seed: int = 0,
) -> float:
    pool: list[tuple[object, np.ndarray]] = []
    for doc in docs:
        for chunk in doc.chunks:
            if chunk.vector is not None and chunk.weight > 0 and not chunk.is_synced:
                pool.append((doc.id, np.asarray(chunk.vector, dtype=np.float32)))
    # A handful of chunks makes the percentile the duplicate itself. Wait until
    # there is enough text for a noise floor.
    if len(pool) < 30:
        return 0.0
    rng = np.random.default_rng(seed)
    sims: list[float] = []
    attempts = 0
    while len(sims) < sample_pairs and attempts < sample_pairs * 5:
        attempts += 1
        left, right = rng.integers(0, len(pool), size=2)
        if pool[int(left)][0] == pool[int(right)][0]:
            continue
        sims.append(float(pool[int(left)][1] @ pool[int(right)][1]))
    if not sims:
        return 0.0
    return float(np.percentile(np.asarray(sims, dtype=np.float32), 99.9))


def apply_p999(thresholds: Thresholds, floor: float) -> Thresholds:
    return apply_floor(thresholds, floor)


def _f1(examples: list[LabeledExample], thresholds: Thresholds) -> float:
    positives_true = 0
    positives_pred = 0
    true_positive = 0
    for example in examples:
        truth = example.label in {"duplicate", "near_copy"}
        relation = classify(example.evidence, thresholds) if _candidate(example, thresholds) else None
        predicted = relation in {"duplicate", "near_copy"}
        if truth:
            positives_true += 1
        if predicted:
            positives_pred += 1
        if truth and predicted:
            true_positive += 1
    if positives_pred == 0 or positives_true == 0:
        return 0.0
    precision = true_positive / positives_pred
    recall = true_positive / positives_true
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _candidate(example: LabeledExample, thresholds: Thresholds) -> bool:
    if example.shared_hash:
        return True
    if example.mutual and example.evidence.median_match >= thresholds.tau_chunk:
        return True
    if example.doc_sim >= thresholds.tau_doc:
        return True
    return False


def _subsample(examples: list[LabeledExample], seed: int, subsample: float) -> list[LabeledExample]:
    if subsample >= 1 or len(examples) < 4:
        return list(examples)
    rng = np.random.default_rng(seed)
    size = max(4, int(len(examples) * subsample))
    picked = rng.choice(len(examples), size=size, replace=False)
    return [examples[int(index)] for index in picked]


def _frange(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    current = start
    while current <= stop + 1e-9:
        values.append(current)
        current += step
    return values
