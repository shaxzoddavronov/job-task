"""Download the static embedding model used by the engine into models/.

Usage: python -m scripts.download_models   (or `make models`)
The pretrained Model2Vec table (~530 MB) is too large to commit; this fetches only the files
StaticModel needs.
"""

from huggingface_hub import snapshot_download

from app.core.config import get_settings

REPO = "minishlab/potion-multilingual-128M"


def main() -> None:
    target = get_settings().static_model_path
    snapshot_download(
        REPO,
        local_dir=target,
        allow_patterns=["*.json", "*.safetensors", "*.md"],
        ignore_patterns=["onnx/*"],
    )
    print(f"{REPO} → {target}")


if __name__ == "__main__":
    main()
