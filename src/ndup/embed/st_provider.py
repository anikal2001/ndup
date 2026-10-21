"""sentence-transformers provider for BAAI/bge-base-en-v1.5.

The model is loaded from the local Hugging Face cache. ``HF_HUB_OFFLINE`` is
set before the import so a missing snapshot fails instead of downloading
during a sync. CPU uses the ONNX int8 backend; CUDA and native MPS use
PyTorch. This module does not import an HTTP client.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from ndup.embed.fake_provider import l2_normalize
from ndup.errors import NdUpError

BGE_MODEL = "BAAI/bge-base-en-v1.5"


class SentenceTransformerProvider:
    model = BGE_MODEL
    dims = 768
    max_tokens = 512

    def __init__(self, device: str = "auto", runtime: str = "onnx-int8") -> None:
        self.device = device
        self.runtime = runtime
        self._model: Any = None
        self._tokenizer: Any = None

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        os.environ["HF_HUB_OFFLINE"] = "1"
        vectors = model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return l2_normalize(array)

    def count_tokens(self, text: str) -> int:
        tokenizer = self._load_tokenizer()
        encoded = tokenizer.encode(text, add_special_tokens=False)
        return len(encoded)

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise NdUpError("BAAI/bge-base-en-v1.5 needs the local extra: uv tool install 'ndup[local]'") from exc
        kwargs: dict[str, Any] = {}
        if self.runtime.startswith("onnx"):
            kwargs["backend"] = "onnx"
            kwargs["model_kwargs"] = {"file_name": "onnx/model_qint8_avx512.onnx"}
        device = _resolve_device(self.device, self.runtime)
        if device is not None:
            kwargs["device"] = device
        self._model = SentenceTransformer(self.model, **kwargs)
        return self._model

    def _load_tokenizer(self) -> Any:
        if self._tokenizer is not None:
            return self._tokenizer
        model = self._load()
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            raise NdUpError("The local embedding model did not expose a tokenizer")
        self._tokenizer = tokenizer
        return tokenizer


def _resolve_device(device: str, runtime: str) -> str | None:
    if device == "cpu" or runtime.startswith("onnx"):
        return "cpu"
    if device == "cuda":
        return "cuda"
    if device == "mps":
        return "mps"
    if device == "auto":
        return None
    return device
