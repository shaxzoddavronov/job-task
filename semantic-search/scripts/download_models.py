"""Download the transformer models (pinned revisions) into the Hugging Face cache.

Usage: python -m scripts.download_models   (or `make models`)
Docker runs this on first start, before the API loads the models, so the ~4.5 GB download
shows progress in the logs and later starts reuse the cached files. Only the files the models
need are fetched (safetensors weights, configs, tokenizer).
"""

import logging

from huggingface_hub import snapshot_download

from app.core.config import get_settings

PATTERNS = ["*.json", "*.safetensors", "sentencepiece.bpe.model", "1_Pooling/*"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    s = get_settings()
    models = [(s.bi_encoder_model, s.bi_encoder_revision), (s.reranker_model, s.reranker_revision)]
    for repo, revision in models:
        path = snapshot_download(
            repo, revision=revision, allow_patterns=PATTERNS, cache_dir=s.hf_cache_dir
        )
        print(f"{repo}@{(revision or 'latest')[:8]} ready ({path})")


if __name__ == "__main__":
    main()
