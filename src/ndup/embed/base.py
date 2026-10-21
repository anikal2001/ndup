"""Embedding provider protocol. Implementations are local only."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class EmbeddingProvider(Protocol):
    model: str
    dims: int
    max_tokens: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return one L2-normalized float32 vector per text."""

    def count_tokens(self, text: str) -> int:
        """Count tokens with this model's tokenizer."""
