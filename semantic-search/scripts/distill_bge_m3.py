"""Offline: distill BAAI/bge-m3 into a static Model2Vec table (numpy only at inference).

Runs in the bench environment (`make install-bench`, then `make distill`). BGE-M3 is loaded
only here; the API and the bot load the saved table from models/ and never run a transformer.

Usage: python -m scripts.distill_bge_m3 [--pca-dims 128 256] [--device cuda]
Writes models/bge-m3-static-<dims>/ (float16) and prints time and size per variant.
"""

import argparse
import time
from pathlib import Path

from model2vec.distill import distill

MODEL = "BAAI/bge-m3"
OUT_DIR = Path(__file__).resolve().parents[1] / "models"


def dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pca-dims", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    args = parser.parse_args()

    for dims in args.pca_dims:
        start = time.perf_counter()
        model = distill(model_name=MODEL, pca_dims=dims, device=args.device, quantize_to="float16")
        out = OUT_DIR / f"bge-m3-static-{dims}"
        model.save_pretrained(str(out))
        print(
            f"{out.name}: {time.perf_counter() - start:.0f}s, {dir_size_mb(out):.0f} MB, "
            f"dim {model.dim}, vocab {len(model.tokens)}"
        )


if __name__ == "__main__":
    main()
