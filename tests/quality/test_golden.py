"""WP1.7 golden-set precision, recall, and the 3-point regression gate."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ndup.analysis.pipeline import analyze_documents
from ndup.analysis.thresholds import FAKE_SHIPPED, thresholds_from_json
from ndup.golden import build_golden_workspace, embed_golden
from ndup.model import order_pair

BASELINE = Path(__file__).resolve().parents[1] / "golden" / "baseline.json"
RELATIONS = ("near_copy", "duplicate", "subset", "partial")


@pytest.mark.quality
def test_golden_f1_gate() -> None:
    docs, labels = build_golden_workspace()
    embed_golden(docs)
    result = analyze_documents(
        docs,
        thresholds_from_json(FAKE_SHIPPED.to_json()),
        now=datetime(2024, 6, 1, tzinfo=UTC),
    )
    predicted = {(pair.doc_a, pair.doc_b): pair.relation for pair in result.pairs}
    truth = {order_pair(left, right): relation for left, right, relation in labels}
    scores = {relation: _f1(truth, predicted, relation) for relation in RELATIONS}
    combined = _f1(truth, predicted, "near_copy", "duplicate")
    assert combined >= 0.80
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    for relation, score in scores.items():
        assert score >= 0.80
        assert score >= float(baseline[relation]) - 0.03
    assert combined >= float(baseline["duplicate_or_near_copy"]) - 0.03


def _f1(
    truth: dict[tuple[object, object], str],
    predicted: dict[tuple[object, object], str],
    *relations: str,
) -> float:
    wanted = set(relations)
    true_pairs = {pair for pair, relation in truth.items() if relation in wanted}
    pred_pairs = {pair for pair, relation in predicted.items() if relation in wanted}
    if not true_pairs:
        return 1.0
    true_positive = len(true_pairs & pred_pairs)
    precision = true_positive / len(pred_pairs) if pred_pairs else 0.0
    recall = true_positive / len(true_pairs)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
