"""Model registration. HNSW indexes are created immediately, before any insert."""

from __future__ import annotations

from typing import Any

from ndup.analysis.thresholds import shipped_for
from ndup.embed.fake_provider import FAKE_MODEL, FakeEmbeddingProvider
from ndup.embed.st_provider import BGE_MODEL, SentenceTransformerProvider
from ndup.errors import NdUpError


def hnsw_index_sql(*, table: str, model_id: int, dims: int, concurrently: bool = False) -> str:
    if table not in {"embedding", "doc_embedding"}:
        raise NdUpError(f"No vector index for table {table}")
    prefix = "emb" if table == "embedding" else "docemb"
    name = f"{prefix}_hnsw_m{int(model_id)}"
    concurrent = "CONCURRENTLY " if concurrently else ""
    return (
        f"CREATE INDEX {concurrent}IF NOT EXISTS {name} ON {table} "
        f"USING hnsw ((vec::halfvec({int(dims)})) halfvec_cosine_ops) "
        f"WITH (m = 16, ef_construction = 64) WHERE model_id = {int(model_id)}"
    )


def knn_session_sql() -> tuple[str, str]:
    return ("SET hnsw.ef_search = 100", "SET hnsw.iterative_scan = relaxed_order")


def runtime_for(device: str, requested: str | None = None) -> str:
    if requested:
        return requested
    if device == "cuda":
        return "cuda-fp16"
    if device == "mps":
        return "torch-mps"
    return "onnx-int8"


def build_provider(model: str, device: str = "auto", runtime: str = "onnx-int8") -> Any:
    if model in {FAKE_MODEL, "fake"}:
        return FakeEmbeddingProvider()
    if model == BGE_MODEL:
        return SentenceTransformerProvider(device=device, runtime=runtime)
    raise NdUpError(f"Unknown embedding model {model!r}. Use {BGE_MODEL} or the local test model {FAKE_MODEL}.")


def initial_thresholds(model: str) -> dict[str, Any]:
    return shipped_for(model)
