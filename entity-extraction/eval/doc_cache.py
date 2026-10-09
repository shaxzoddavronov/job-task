"""Cache extracted documents so evaluation runs do not re-OCR the corpus.

OCR of the scanned variants dominates evaluation time — about ten seconds per
page against well under a second for the LLM calls. The cache is keyed by path
and modification time, so regenerating a PDF invalidates its entry.
"""

from __future__ import annotations

import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "eval" / ".cache"


def cached_extract(pdf: Path, use_cache: bool = True):
    from app.extract_text import extract

    CACHE_DIR.mkdir(exist_ok=True)
    key = CACHE_DIR / f"doc-{pdf.stem}-{int(pdf.stat().st_mtime)}.pickle"
    if use_cache and key.exists():
        try:
            return pickle.loads(key.read_bytes())
        except Exception:
            pass  # a stale or truncated entry is just re-extracted

    document = extract(str(pdf))
    try:
        key.write_bytes(pickle.dumps(document))
    except Exception:
        pass
    return document
