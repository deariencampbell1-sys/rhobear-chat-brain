from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]:
        ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device="cpu")

    def embed(self, text: str) -> list[float]:
        vector = self._model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.astype(np.float32).tolist()


class DeterministicEmbedder:
    """Lightweight embedder for tests: stable vectors from text hash."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big") % (2**32)
        rng = np.random.default_rng(seed)
        vector = rng.standard_normal(self.dim).astype(np.float32)
        vector /= np.linalg.norm(vector)
        return vector.tolist()


_embedder: Embedder | None = None


def get_embedder(model_name: str, *, test_mode: bool = False, dim: int = 384) -> Embedder:
    global _embedder
    if _embedder is None:
        if test_mode:
            _embedder = DeterministicEmbedder(dim=dim)
        else:
            _embedder = SentenceTransformerEmbedder(model_name)
    return _embedder


def reset_embedder() -> None:
    global _embedder
    _embedder = None