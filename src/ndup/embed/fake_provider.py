"""Deterministic local embedder for tests and offline development.

Each token is a fixed Gaussian vector. The text vector is the L2-normalized
sum, so cosine tracks token overlap. Nothing is downloaded and no socket is
opened.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import cast

import numpy as np

from ndup.text.tokens import simple_token_count

FAKE_MODEL = "fake-bow-v1"


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim == 1:
        norm = float(np.linalg.norm(array))
        if norm == 0.0:
            return array
        return cast(np.ndarray, array / norm)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return cast(np.ndarray, (array / norms).astype(np.float32))


class FakeEmbeddingProvider:
    model = FAKE_MODEL
    dims = 768
    max_tokens = 512

    def __init__(self, dims: int = 768) -> None:
        self.dims = dims
        self._token_vectors: dict[str, np.ndarray] = {}

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dims), dtype=np.float32)
        for index, text in enumerate(texts):
            vectors[index] = self._embed_one(text)
        return vectors

    def count_tokens(self, text: str) -> int:
        return simple_token_count(text)

    def _embed_one(self, text: str) -> np.ndarray:
        tokens = [tok for tok in text.lower().split() if tok]
        if not tokens:
            return np.zeros(self.dims, dtype=np.float32)
        total = np.zeros(self.dims, dtype=np.float32)
        for token in tokens:
            total += self._token_vector(token)
        return l2_normalize(total)

    def _token_vector(self, token: str) -> np.ndarray:
        cached = self._token_vectors.get(token)
        if cached is not None:
            return cached
        digest = hashlib.sha256(token.encode()).digest()
        seed = int.from_bytes(digest[:8], "little")
        rng = np.random.default_rng(seed)
        vector = rng.standard_normal(self.dims).astype(np.float32)
        self._token_vectors[token] = vector
        return vector


def provider_token_counter(provider: FakeEmbeddingProvider) -> Callable[[str], int]:
    return provider.count_tokens
