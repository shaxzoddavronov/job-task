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
    use_static_emb: bool = True  # Model2Vec channel: +3.8 realistic-dev Top-1 (Phase 6 sweep)
    static_model_path: Path = PROJECT_ROOT / "models" / "potion-multilingual-128M"
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
    weight_static: float = 1.0
    answer_index_weight: float = 0.1

    # Query expansion with data/synonyms.yaml (all retrieval channels)
    use_synonyms: bool = True  # +4.7 realistic-dev Top-1 with the re-ranker (Phase 5 sweep)

    # Entity-aware linear re-ranker over the top fused candidates (Phase 5, tuned on
    # validation + realistic-dev with `scripts.evaluate --tune-reranker`; re-tuned in Phase 6).
    use_reranker: bool = True
    rerank_top_n: int = 20
    rerank_w_rrf: float = 2.0
    rerank_w_sim_q: float = 0.5
    rerank_w_sim_a: float = 0.0
    rerank_w_sim_e: float = 0.0
    rerank_w_fuzzy: float = 2.0
    rerank_w_entity_match: float = 0.5
    # Phase 6: entity_match / entity_match_q at 0.5 each tie with either alone on the decision
    # sets; both kept so an entity found only in an answer still lifts it (plan requirement).
    rerank_w_entity_match_q: float = 0.5
    rerank_w_category: float = 0.5
    # No law numbers in the decision sets, so conflict is a safety setting, not a tuned one.
    rerank_w_entity_conflict: float = 3.0

    # API
    max_question_chars: int = 500
    preview_chars: int = 160
    logs_dir: Path = PROJECT_ROOT / "data" / "logs"  # query logs + feedback until Postgres (9b)

    # Data source
    data_source: Literal["postgres", "file"] = "file"
    database_url: str = "postgresql+psycopg://qa:qa@localhost:5432/qa"

    # Evaluation
    eval_seed: int = 42
    eval_top_k: int = 10
    eval_realistic_path: Path = PROJECT_ROOT / "data" / "eval_realistic.csv"
    eval_realistic_dev_path: Path = PROJECT_ROOT / "data" / "eval_realistic_dev.csv"

    # Decision thresholds, calibrated on validation + realistic-dev + OOD-dev
    # (`scripts.evaluate --calibrate`, reports/thresholds.md): 95% precision when answering.
    answer_threshold: float = 0.68
    suggest_threshold: float = 0.61
    answer_min_margin: float = 0.05

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
