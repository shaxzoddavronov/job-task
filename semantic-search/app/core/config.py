"""Central configuration. Every path, threshold and weight lives here or in `.env`."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_ignore_empty=True,
        extra="ignore",
    )

    # Paths
    data_dir: Path = PROJECT_ROOT / "data"
    answers_path: Path = PROJECT_ROOT / "data" / "answers.csv"
    questions_path: Path = PROJECT_ROOT / "data" / "questions.csv"
    gazetteer_path: Path = PROJECT_ROOT / "data" / "gazetteer.yaml"
    synonyms_path: Path = PROJECT_ROOT / "data" / "synonyms.yaml"
    ood_questions_path: Path = PROJECT_ROOT / "data" / "ood_questions.txt"
    ood_dev_path: Path = PROJECT_ROOT / "data" / "ood_dev.txt"  # threshold calibration only
    models_dir: Path = PROJECT_ROOT / "models"
    reports_dir: Path = PROJECT_ROOT / "reports"

    # Normalization
    normalize_strip_apostrophes: bool = False

    # Transformer models (pinned Hugging Face revisions for reproducible results)
    model_backend: Literal["transformers", "fake"] = "transformers"  # fake = tests only
    bi_encoder_model: str = "BAAI/bge-m3"
    bi_encoder_revision: str | None = "5617a9f61b028005a4858fdac845db406aefb181"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_revision: str | None = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    hf_cache_dir: str | None = None  # None = HF_HOME / default Hugging Face cache

    # Compute: DEVICE=auto picks CUDA when available. The reranker runs on GPU only by
    # default (RERANKER=auto); on CPU the engine uses dense retrieval alone.
    device: str = "auto"
    reranker: Literal["auto", "on", "off"] = "auto"
    cpu_threads: int = 0  # 0 = torch default
    max_seq_length: int = 512
    encode_batch_size: int = 32
    rerank_batch_size: int = 32
    embedding_cache_dir: Path | None = PROJECT_ROOT / ".cache" / "embeddings"

    # Retrieval: question vectors plus answer-text vectors scaled by answer_index_weight
    use_answer_index: bool = True
    answer_index_weight: float = 0.3
    rerank_top_n: int = 10  # dev sweep: 10 ≥ 20 ≈ 30, and the fastest
    # Text paired with the query for the cross-encoder: best-matching question, answer, or both
    # (dev sweep: question+answer 91.6 > question 82.2 > answer 64.5 Top-1)
    rerank_pair: Literal["question", "answer", "question+answer"] = "question+answer"

    # Query expansion with data/synonyms.yaml
    use_synonyms: bool = True  # dev sweep: +3.7 Top-1 in dense mode, +1.9 with the reranker

    # Final linear score per mode (weighted mean of features in [0, 1] minus a law-conflict
    # penalty; see ranker.py). "rerank" = GPU with cross-encoder, "dense" = CPU / no reranker.
    # Tuned on validation + realistic-dev: reranker 1 : dense 2 keeps the reranker's dev gain
    # (91.6 vs 89.7 Top-1) without its validation loss (99.2 vs 97.0). Entity features did
    # not help in either mode; the law-conflict penalty is a safety rule (no laws in dev).
    weights_rerank: dict[str, float] = {
        "rerank": 1.0,
        "dense": 2.0,
        "sim_a": 0.0,
        "entity_match": 0.0,
        "entity_match_q": 0.0,
        "category": 0.0,
        "entity_conflict": 3.0,
    }
    weights_dense: dict[str, float] = {
        "rerank": 0.0,
        "dense": 1.0,
        "sim_a": 0.0,
        "entity_match": 0.0,
        "entity_match_q": 0.0,
        "category": 0.0,
        "entity_conflict": 3.0,
    }

    # Decision thresholds per mode, calibrated on validation + realistic-dev + OOD-dev
    # (`scripts.evaluate --calibrate`, reports/thresholds_<mode>.md): ≥ 95% precision when
    # answering on realistic-dev, ≤ 5% of off-topic questions answered, ≥ 80% not_found.
    answer_threshold_rerank: float = 0.56
    suggest_threshold_rerank: float = 0.45
    answer_min_margin_rerank: float = 0.02
    answer_threshold_dense: float = 0.65
    suggest_threshold_dense: float = 0.62
    answer_min_margin_dense: float = 0.02

    # API
    max_question_chars: int = 500
    preview_chars: int = 160
    logs_dir: Path = PROJECT_ROOT / "data" / "logs"  # query logs + feedback until Postgres (9b)

    # Data source
    data_source: Literal["postgres", "file"] = "file"
    # Local default = the compose `db` service on host port 5440 (Docker uses db:5432)
    database_url: str = "postgresql+psycopg://qa:qa@localhost:5440/qa"

    # Evaluation
    eval_seed: int = 42
    eval_top_k: int = 10
    eval_realistic_path: Path = PROJECT_ROOT / "data" / "eval_realistic.csv"
    eval_realistic_dev_path: Path = PROJECT_ROOT / "data" / "eval_realistic_dev.csv"

    # Bot
    api_url: str = "http://localhost:8000"
    bot_api_timeout: float = 10.0
    bot_token: str | None = None
    bot_token_file: Path = PROJECT_ROOT / "bot_token.txt"

    def resolve_bot_token(self) -> str | None:
        """Return BOT_TOKEN from the environment, falling back to the token file."""
        if self.bot_token:
            return self.bot_token.strip()
        path = self.bot_token_file
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.is_file():
            return path.read_text(encoding="utf-8").strip() or None
        return None


@lru_cache
def get_settings() -> Settings:
    return Settings()
