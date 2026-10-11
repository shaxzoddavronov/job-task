"""Measure per-question latency and GPU memory of the engine on this machine.

Usage: python -m scripts.latency [--n 100]   (or `make latency`)
Builds the engine with the current settings (DEVICE / RERANKER from .env), times `n` realistic
questions after a warm-up, and prints p50 / p95 latency, the mode and peak GPU memory.
"""

import argparse
import time

import numpy as np
import pandas as pd

from app.core.config import get_settings
from app.core.data import load_corpus
from app.core.device import describe_device
from app.core.engine import QAEngine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    settings = get_settings()
    engine = QAEngine(load_corpus(settings), settings)
    queries = pd.read_csv(settings.eval_realistic_dev_path)["query"].tolist()
    queries = (queries * (args.n // len(queries) + 1))[: args.n]
    for q in queries[:5]:  # warm-up
        engine.ask(q)
    if engine.device == "cuda":
        import torch

        torch.cuda.reset_peak_memory_stats()
    times = []
    for q in queries:
        start = time.perf_counter()
        engine.ask(q)
        times.append((time.perf_counter() - start) * 1000)
    print(f"device/mode: {describe_device(engine.device)} · {engine.mode}")
    print(
        f"latency over {len(times)} questions: p50 {np.percentile(times, 50):.1f} ms, "
        f"p95 {np.percentile(times, 95):.1f} ms"
    )
    if engine.device == "cuda":
        print(f"peak GPU memory (PyTorch): {torch.cuda.max_memory_allocated() / 2**20:.0f} MB")


if __name__ == "__main__":
    main()
