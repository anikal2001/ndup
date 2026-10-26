"""Shipped thresholds.

``fake-bow-v1`` is fitted on the bundled golden workspace. ``bge-base`` keeps a
conservative cosine prior until ``ndup calibrate`` sees labeled pairs from the
real embedder. The golden-set F1 gate runs in the fake-bow space so CI does
not download model weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ndup.embed.fake_provider import FAKE_MODEL
from ndup.embed.openai_provider import OPENAI_MODELS
from ndup.embed.st_provider import BGE_MODEL


@dataclass(frozen=True)
class Thresholds:
    tau_chunk: float
    tau_doc: float
    short_doc_delta: float = 0.05
    short_doc_chunks: int = 2
    source: str = "shipped"
    n_labels: int = 0
    f1: float | None = None
    p999_floor: float | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "tau_chunk": self.tau_chunk,
            "tau_doc": self.tau_doc,
            "short_doc_delta": self.short_doc_delta,
            "short_doc_chunks": self.short_doc_chunks,
            "source": self.source,
            "n_labels": self.n_labels,
            "f1": self.f1,
            "p999_floor": self.p999_floor,
        }


def thresholds_from_json(payload: dict[str, Any] | None) -> Thresholds:
    data = payload or {}
    return Thresholds(
        tau_chunk=float(data.get("tau_chunk", 0.72)),
        tau_doc=float(data.get("tau_doc", 0.68)),
        short_doc_delta=float(data.get("short_doc_delta", 0.05)),
        short_doc_chunks=int(data.get("short_doc_chunks", 2) or 2),
        source=str(data.get("source", "shipped")),
        n_labels=int(data.get("n_labels", 0) or 0),
        f1=None if data.get("f1") is None else float(data["f1"]),
        p999_floor=None if data.get("p999_floor") is None else float(data["p999_floor"]),
    )


def apply_preset(thresholds: Thresholds, preset: str) -> Thresholds:
    shift = {"strict": 0.02, "default": 0.0, "loose": -0.02}[preset]
    return Thresholds(
        tau_chunk=thresholds.tau_chunk + shift,
        tau_doc=thresholds.tau_doc + shift,
        short_doc_delta=thresholds.short_doc_delta,
        short_doc_chunks=thresholds.short_doc_chunks,
        source=thresholds.source,
        n_labels=thresholds.n_labels,
        f1=thresholds.f1,
        p999_floor=thresholds.p999_floor,
    )


def apply_floor(thresholds: Thresholds, floor: float) -> Thresholds:
    tau = thresholds.tau_chunk if floor <= thresholds.tau_chunk else floor
    return Thresholds(
        tau_chunk=tau,
        tau_doc=thresholds.tau_doc,
        short_doc_delta=thresholds.short_doc_delta,
        short_doc_chunks=thresholds.short_doc_chunks,
        source=thresholds.source,
        n_labels=thresholds.n_labels,
        f1=thresholds.f1,
        p999_floor=floor,
    )


# Fitted by tests/unit/test_quality.py on the golden workspace. Updated when
# that fit moves by more than 0.005. See the quality test for the gate.
FAKE_SHIPPED = Thresholds(
    tau_chunk=0.70,
    tau_doc=0.66,
    short_doc_delta=0.05,
    source="shipped",
    n_labels=0,
    f1=None,
)

BGE_SHIPPED = Thresholds(
    tau_chunk=0.82,
    tau_doc=0.78,
    short_doc_delta=0.05,
    source="shipped",
    n_labels=0,
    f1=None,
)


# Provisional prior for text-embedding-3-*; not fitted. Run ``ndup calibrate``.
OPENAI_SHIPPED = Thresholds(
    tau_chunk=0.80,
    tau_doc=0.76,
    short_doc_delta=0.05,
    source="provisional-openai",
    n_labels=0,
    f1=None,
)


def shipped_for(model: str) -> dict[str, Any]:
    if model in {FAKE_MODEL, "fake"}:
        return FAKE_SHIPPED.to_json()
    if model == BGE_MODEL:
        return BGE_SHIPPED.to_json()
    if model in OPENAI_MODELS:
        return OPENAI_SHIPPED.to_json()
    return FAKE_SHIPPED.to_json()
