"""Transformer models behind small protocols, so the engine and tests do not depend on them.

- Encoder: text → L2-normalized vector (bi-encoder, BAAI/bge-m3).
- Scorer:  (query, candidate) → relevance in [0, 1] (cross-encoder, BAAI/bge-reranker-v2-m3).

Models are loaded once per process (cached by name, revision, device) because loading takes
seconds and the engine is rebuilt on /reload and twice per evaluation run.
"""

import logging
import threading
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class Scorer(Protocol):
    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray: ...


_cache: dict[tuple, object] = {}
_cache_lock = threading.Lock()


def _dtype_kwargs(device: str) -> dict:
    import torch

    return {"torch_dtype": torch.float16} if device == "cuda" else {}


def _is_oom(error: Exception) -> bool:
    return "out of memory" in str(error).lower()


class SentenceTransformerEncoder:
    def __init__(
        self,
        model: str,
        revision: str | None,
        device: str,
        max_length: int,
        batch_size: int,
        cache_dir: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        key = ("bi", model, revision, device)
        with _cache_lock:
            if key not in _cache:
                log.info("Loading %s@%s on %s", model, (revision or "latest")[:8], device)
                _cache[key] = SentenceTransformer(
                    model,
                    revision=revision,
                    device=device,
                    cache_folder=cache_dir,
                    model_kwargs=_dtype_kwargs(device),
                )
        self.model = _cache[key]
        self.model.max_seq_length = max_length
        self.batch_size = batch_size

    def encode(self, texts: list[str]) -> np.ndarray:
        batch = self.batch_size
        while True:
            try:
                vectors = self.model.encode(
                    texts,
                    batch_size=batch,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                return np.asarray(vectors, dtype=np.float32)
            except RuntimeError as e:  # CUDA OOM on small GPUs: retry with smaller batches
                if not _is_oom(e) or batch == 1:
                    raise
                batch = max(1, batch // 2)
                log.warning("Out of GPU memory while encoding; retrying with batch %d", batch)
                _empty_cuda_cache()


class CrossEncoderScorer:
    def __init__(
        self,
        model: str,
        revision: str | None,
        device: str,
        max_length: int,
        batch_size: int,
        cache_dir: str | None = None,
    ) -> None:
        from sentence_transformers import CrossEncoder

        key = ("cross", model, revision, device)
        with _cache_lock:
            if key not in _cache:
                log.info("Loading %s@%s on %s", model, (revision or "latest")[:8], device)
                _cache[key] = CrossEncoder(
                    model,
                    revision=revision,
                    device=device,
                    cache_folder=cache_dir,
                    model_kwargs=_dtype_kwargs(device),
                )
        self.model = _cache[key]
        self.max_length = max_length
        self.batch_size = batch_size

    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        scores = self.model.predict(
            [list(p) for p in pairs],
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )  # sigmoid is applied by the model config → [0, 1]
        return np.asarray(scores, dtype=np.float32).reshape(-1)


def _empty_cuda_cache() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --- Deterministic fakes for unit tests (no download, no torch) --------------------------------


class HashingEncoder:
    """Character-trigram hashing vectors: similar strings get similar vectors."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            t = f"  {text.lower()}  "
            for j in range(len(t) - 2):
                out[i, hash(t[j : j + 3]) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


class OverlapScorer:
    """Word-overlap relevance in [0, 1] (Jaccard)."""

    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        scores = []
        for query, doc in pairs:
            a, b = set(query.lower().split()), set(doc.lower().split())
            scores.append(len(a & b) / len(a | b) if a | b else 0.0)
        return np.asarray(scores, dtype=np.float32)
