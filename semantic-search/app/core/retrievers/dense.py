"""Dense (bi-encoder) vectors for a fixed list of texts, with an on-disk cache.

Encoding the whole corpus takes seconds on a GPU but minutes on a CPU, so document vectors are
cached under EMBEDDING_CACHE_DIR, keyed by the model id and the exact texts. Any change to the
model, its revision or the data gives a new key, so a stale cache is never used.
"""

import hashlib
import logging
import os
from pathlib import Path

import numpy as np

from app.core.encoders import Encoder

log = logging.getLogger(__name__)


def _cache_key(model_id: str, texts: list[str]) -> str:
    digest = hashlib.sha256(model_id.encode())
    for text in texts:
        digest.update(b"\0" + text.encode())
    return digest.hexdigest()[:32]


def encode_cached(encoder: Encoder, texts: list[str], cache_dir: Path | None) -> np.ndarray:
    """encoder.encode(texts), reusing a cached result when the same model saw the same texts."""
    model_id = getattr(encoder, "cache_id", None)
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    if cache_dir is None or model_id is None:
        return encoder.encode(texts)
    path = Path(cache_dir) / f"{_cache_key(model_id, texts)}.npy"
    if path.is_file():
        vectors = np.load(path)
        if vectors.shape[0] == len(texts):
            log.info("Loaded %d cached vectors from %s", len(texts), path.name)
            return vectors
    vectors = encoder.encode(texts)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, vectors)
        os.replace(tmp, path)  # atomic: concurrent builds never see a half-written file
    except OSError as e:  # a read-only cache is not an error, only slower
        log.warning("Could not write embedding cache %s: %s", path, e)
    return vectors


def best_per_group(
    scores: np.ndarray, groups: np.ndarray, n_groups: int
) -> tuple[np.ndarray, np.ndarray]:
    """Max score and the index of the row that reached it, per group (groups = 0..n_groups-1)."""
    best = np.full(n_groups, -np.inf, dtype=np.float32)
    np.maximum.at(best, groups, scores)
    arg = np.full(n_groups, -1, dtype=np.int64)
    hits = np.flatnonzero(scores >= best[groups])
    # first row reaching the max wins (reverse so earlier rows overwrite later ones)
    arg[groups[hits[::-1]]] = hits[::-1]
    return best, arg
