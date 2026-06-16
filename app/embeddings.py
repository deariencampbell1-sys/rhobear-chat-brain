from __future__ import annotations

import hashlib
import os
from pathlib import Path
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


class OnnxEmbedder:
    """CPU ONNX runtime embedder used in the slim Docker image."""

    def __init__(self, model_dir: Path) -> None:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model_path = model_dir / "model.onnx"
        self._session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )

    def embed(self, text: str) -> list[float]:
        encoded = self._tokenizer(
            text,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {inp.name: encoded[inp.name] for inp in self._session.get_inputs()}
        outputs = self._session.run(None, inputs)
        token_embeddings = outputs[0]
        attention_mask = encoded["attention_mask"]
        mask = np.expand_dims(attention_mask, -1).astype(np.float32)
        summed = (token_embeddings * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        vector = summed / counts
        vector = vector / np.linalg.norm(vector, axis=1, keepdims=True)
        return vector[0].astype(np.float32).tolist()


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


def get_embedder(
    model_name: str,
    *,
    test_mode: bool = False,
    dim: int = 384,
    onnx_model_dir: Path | None = None,
) -> Embedder:
    global _embedder
    if _embedder is None:
        if test_mode:
            _embedder = DeterministicEmbedder(dim=dim)
        elif onnx_model_dir and (onnx_model_dir / "model.onnx").exists():
            _embedder = OnnxEmbedder(onnx_model_dir)
        else:
            _embedder = SentenceTransformerEmbedder(model_name)
    return _embedder


def reset_embedder() -> None:
    global _embedder
    _embedder = None