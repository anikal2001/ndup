"""OpenAI embeddings provider. The only remote provider; opt-in by model name.

Chunk text is sent to api.openai.com when the configured model is one of
``OPENAI_MODELS``. The local providers and the no-egress guard are unchanged.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from ndup.embed.fake_provider import l2_normalize
from ndup.errors import NdUpError
from ndup.secrets import resolve_secret
from ndup.text.tokens import simple_token_count

OPENAI_MODELS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}
OPENAI_RUNTIME = "openai-api"


class OpenAIEmbeddingProvider:
    max_tokens = 8191

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        client: Any = None,
        batch_size: int = 256,
    ) -> None:
        if model not in OPENAI_MODELS:
            raise NdUpError(f"Unknown OpenAI embedding model {model!r}. Use one of {sorted(OPENAI_MODELS)}.")
        self.model = model
        self.dims = OPENAI_MODELS[model]
        self.batch_size = batch_size
        self._api_key = api_key
        self._client = client

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dims), dtype=np.float32)
        if not texts:
            return vectors
        client = self._load()
        for start in range(0, len(texts), self.batch_size):
            batch = [text if text.strip() else " " for text in texts[start : start + self.batch_size]]
            response = client.embeddings.create(model=self.model, input=batch, encoding_format="float")
            for item in response.data:
                vectors[start + item.index] = np.asarray(item.embedding, dtype=np.float32)
        return l2_normalize(vectors)

    def count_tokens(self, text: str) -> int:
        return simple_token_count(text)

    def _load(self) -> Any:
        if self._client is not None:
            return self._client
        api_key = self._api_key or resolve_secret("openai_api_key") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise NdUpError("OpenAI model configured but no key found. Set NDUP_OPENAI_API_KEY or OPENAI_API_KEY.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise NdUpError("Install the openai extra: uv sync --extra openai") from exc
        self._client = OpenAI(api_key=api_key, max_retries=5, timeout=60.0)
        return self._client
