"""OpenAI provider contract. Uses a stub client; no socket is opened."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ndup.analysis.thresholds import shipped_for
from ndup.embed.openai_provider import OPENAI_MODELS, OPENAI_RUNTIME, OpenAIEmbeddingProvider
from ndup.embed.registry import build_provider, runtime_for
from ndup.errors import NdUpError


class StubEmbeddings:
    def __init__(self, dims: int) -> None:
        self.dims = dims
        self.calls: list[list[str]] = []

    def create(self, *, model: str, input: list[str], encoding_format: str) -> SimpleNamespace:
        assert model in OPENAI_MODELS
        assert encoding_format == "float"
        assert all(text for text in input), "OpenAI rejects empty strings"
        self.calls.append(list(input))
        # Return out of order to prove the provider sorts by index.
        data = [
            SimpleNamespace(index=i, embedding=[float(len(text))] + [1.0] * (self.dims - 1))
            for i, text in enumerate(input)
        ]
        return SimpleNamespace(data=list(reversed(data)))


def stub_client(dims: int) -> SimpleNamespace:
    return SimpleNamespace(embeddings=StubEmbeddings(dims))


@pytest.mark.unit
def test_openai_provider_batches_orders_and_normalizes() -> None:
    client = stub_client(1536)
    provider = OpenAIEmbeddingProvider("text-embedding-3-small", client=client, batch_size=4)
    texts = [f"text {'x' * i}" for i in range(10)] + [""]
    vectors = provider.embed(texts)

    assert vectors.shape == (11, 1536)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    assert [len(call) for call in client.embeddings.calls] == [4, 4, 3]
    assert client.embeddings.calls[2][-1] == " "
    # First coordinate encodes len(text), so longer texts rank higher after normalization order is preserved.
    firsts = vectors[:10, 0] / vectors[:10, 1]
    assert np.all(np.diff(firsts) > 0)


@pytest.mark.unit
def test_openai_provider_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("NDUP_OPENAI_API_KEY", raising=False)
    provider = OpenAIEmbeddingProvider("text-embedding-3-large")
    assert provider.dims == 3072
    with pytest.raises(NdUpError, match="NDUP_OPENAI_API_KEY"):
        provider.embed(["hello"])


@pytest.mark.unit
def test_registry_and_thresholds_know_openai_models() -> None:
    assert runtime_for("cuda", None, "text-embedding-3-small") == OPENAI_RUNTIME
    assert runtime_for("cuda", None, "BAAI/bge-base-en-v1.5") == "cuda-fp16"
    provider = build_provider("text-embedding-3-small")
    assert isinstance(provider, OpenAIEmbeddingProvider)
    assert shipped_for("text-embedding-3-small")["source"] == "provisional-openai"
    with pytest.raises(NdUpError):
        OpenAIEmbeddingProvider("text-embedding-ada-002")
