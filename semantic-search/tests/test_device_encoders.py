import numpy as np
import pytest

from app.core import device as dev
from app.core.encoders import HashingEncoder, OverlapScorer


def test_resolve_device(monkeypatch):
    monkeypatch.setattr(dev, "cuda_available", lambda: True)
    assert dev.resolve_device("auto") == "cuda"
    monkeypatch.setattr(dev, "cuda_available", lambda: False)
    assert dev.resolve_device("auto") == "cpu"
    assert dev.resolve_device("CPU") == "cpu"
    with pytest.raises(RuntimeError, match="no CUDA"):
        dev.resolve_device("cuda")
    with pytest.raises(ValueError):
        dev.resolve_device("tpu")


def test_hashing_encoder_is_normalized_and_similarity_aware():
    v = HashingEncoder().encode(["pasport olish", "pasport olish tartibi", "soliq qarzi"])
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
    assert v[0] @ v[1] > v[0] @ v[2]


def test_overlap_scorer_range():
    s = OverlapScorer().score([("a b", "a b"), ("a b", "c d"), ("", "")])
    assert s.tolist() == [1.0, 0.0, 0.0]
