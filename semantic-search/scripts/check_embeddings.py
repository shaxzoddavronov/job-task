"""Sanity-check a static (Model2Vec) embedding model on the corpus vocabulary.

Usage: python -m scripts.check_embeddings models/potion-multilingual-128M [more model dirs...]

Reports, per model: size, tokenizer coverage of the corpus words (unknown-token rate, pieces
per word, share of words kept as a single token) and the nearest corpus words for a fixed list
of Uzbek probe words. Run on several models to compare them side by side.
"""

import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from model2vec import StaticModel

from app.core.config import get_settings
from app.core.data import FileSource, build_corpus
from app.core.normalize import normalize

PROBES = [
    "pasport",
    "ariza",
    "sud",
    "vazirlik",
    "soliq",
    "pensiya",
    "nikoh",
    "meros",
    "guvohnoma",
    "notarius",
]
MIN_FREQ = 2
NEIGHBOURS = 6


def corpus_words() -> Counter:
    settings = get_settings()
    corpus = build_corpus(FileSource(settings.answers_path, settings.questions_path).load_raw())
    texts = [q.text for q in corpus.questions] + [a.text for a in corpus.answers.values()]
    return Counter(w for t in texts for w in re.findall(r"[a-z']+", normalize(t)) if len(w) > 2)


def check(model_dir: Path, words: Counter) -> None:
    start = time.perf_counter()
    model = StaticModel.from_pretrained(str(model_dir))
    load_s = time.perf_counter() - start
    size_mb = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file()) / 1e6
    tokenizer = model.tokenizer
    unk = tokenizer.token_to_id("[UNK]") if tokenizer.token_to_id("[UNK]") is not None else -1

    vocab = [w for w, n in words.items() if n >= MIN_FREQ]
    pieces = [tokenizer.encode(w, add_special_tokens=False).ids for w in vocab]
    n_pieces = np.array([len(p) for p in pieces])
    unk_rate = np.mean([unk in p for p in pieces])
    print(f"\n=== {model_dir.name}: {size_mb:.0f} MB on disk, dim {model.dim}, load {load_s:.1f}s")
    print(
        f"corpus words (freq ≥ {MIN_FREQ}): {len(vocab)}"
        f" · single-token {np.mean(n_pieces == 1):.1%}"
        f" · pieces/word {n_pieces.mean():.2f} · words with [UNK] {unk_rate:.1%}"
    )

    emb = model.encode(vocab)
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
    probe_emb = model.encode(PROBES)
    probe_emb = probe_emb / np.maximum(np.linalg.norm(probe_emb, axis=1, keepdims=True), 1e-9)
    sims = probe_emb @ emb.T
    for probe, row in zip(PROBES, sims, strict=True):
        order = [i for i in np.argsort(-row) if vocab[i] != probe][:NEIGHBOURS]
        print(f"  {probe:10} → " + ", ".join(f"{vocab[i]} ({row[i]:.2f})" for i in order))


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    words = corpus_words()
    for path in sys.argv[1:]:
        check(Path(path), words)


if __name__ == "__main__":
    main()
