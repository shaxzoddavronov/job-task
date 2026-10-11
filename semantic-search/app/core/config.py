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
    models_dir: Path = PROJECT_ROOT / "models"
    reports_dir: Path = PROJECT_ROOT / "reports"

    # Normalization (stripping apostrophes is measured as a Phase 4 ablation)
    normalize_strip_apostrophes: bool = False

    # Template filler phrases from the generated paraphrases (Phase 1 audit, ablation E4)
    strip_template_fillers: bool = False
    template_fillers: list[str] = [
        "haqida ma'lumot bering",
        "bo'yicha ma'lumot kerak",
        "haqida ayting",
        "iltimos ayting",
        "menga",
    ]

    # Retrieval
    use_tfidf: bool = True
    use_bm25: bool = False  # hurt realistic-dev Top-1 by ~6 points (Phase 4 sweep)
    use_answer_index: bool = True
    tfidf_ngram_min: int = 3
    tfidf_ngram_max: int = 5
    tfidf_entity_weight: float = 0.25
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    candidate_pool: int = 100

    # Fusion (weighted RRF); answer-text lists are scaled by answer_index_weight.
    # Small k and a small answer weight keep the answer index a tie-breaker (Phase 4 sweep).
    rrf_k: int = 5
    weight_tfidf: float = 1.0
    weight_bm25: float = 1.0
    answer_index_weight: float = 0.1

    # Data source
    data_source: Literal["postgres", "file"] = "file"
    database_url: str = "postgresql+psycopg://qa:qa@localhost:5432/qa"

    # Evaluation
    eval_seed: int = 42
    eval_top_k: int = 10
    eval_realistic_path: Path = PROJECT_ROOT / "data" / "eval_realistic.csv"
    eval_realistic_dev_path: Path = PROJECT_ROOT / "data" / "eval_realistic_dev.csv"

    # Decision thresholds (placeholders until Phase 7 calibration)
    answer_threshold: float = 0.5
    suggest_threshold: float = 0.2
    answer_min_margin: float = 0.0

    # Bot
    api_url: str = "http://localhost:8000"
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
