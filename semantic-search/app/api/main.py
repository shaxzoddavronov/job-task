"""FastAPI service. The engine is built once at startup and kept in RAM; /reload swaps it.

Run: `make api` → http://localhost:8000/docs
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from app.api.schemas import (
    AnswerOut,
    AskRequest,
    AskResponse,
    CandidateOut,
    EntityOut,
    FeedbackIn,
    FeedbackOut,
    HealthOut,
    ReloadOut,
    StatsOut,
)
from app.core.config import Settings, get_settings
from app.core.data import load_corpus
from app.core.engine import QAEngine
from app.core.normalize import normalize
from app.db.repository import Feedback, QueryLog, Repository, build_repository

log = logging.getLogger("app.api")


@dataclass
class EngineState:
    """Engine plus the per-answer lookups the API needs; swapped as one object on reload."""

    engine: QAEngine
    first_question: dict[int, str]
    questions_by_answer: dict[int, list[str]]


def build_state(settings: Settings) -> EngineState:
    corpus = load_corpus(settings)
    questions: dict[int, list[str]] = {a: [] for a in corpus.answers}
    for q in sorted(corpus.questions, key=lambda q: q.id):
        questions[q.answer_id].append(q.text)
    first = {a: (qs[0] if qs else "") for a, qs in questions.items()}
    return EngineState(QAEngine(corpus, settings), first, questions)


def create_app(settings: Settings | None = None, repo: Repository | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.engine_state = await run_in_threadpool(build_state, settings)
        app.state.repo = repo or build_repository(settings)
        app.state.reload_lock = asyncio.Lock()
        s = app.state.engine_state
        log.info(
            "Engine ready: %d answers, %d questions, built in %.2fs",
            len(s.engine.corpus.answers),
            len(s.engine.corpus.questions),
            s.engine.build_seconds,
        )
        yield

    app = FastAPI(
        title="Uzbek Legal QA",
        description="Semantic search over a fixed answer bank (no LLMs, no transformers at "
        "inference). Ask in Uzbek, Latin or Cyrillic.",
        version="1.0.0",
        lifespan=lifespan,
    )

    def state(request: Request) -> EngineState:
        return request.app.state.engine_state

    def preview(text: str) -> str:
        n = settings.preview_chars
        return text if len(text) <= n else text[: n - 1].rstrip() + "…"

    def write_log(repository: Repository, entry: QueryLog) -> None:
        try:
            repository.log_query(entry)
        except Exception:  # logging must never break a request
            log.exception("Failed to write query log")

    @app.post("/ask", response_model=AskResponse, tags=["search"])
    def ask(body: AskRequest, request: Request, background: BackgroundTasks) -> AskResponse:
        st = state(request)
        result = st.engine.ask(body.question, k=body.top_k, debug=body.debug)
        answers = st.engine.corpus.answers
        top = result.top if result.status == "answer" else None
        log.info(
            "ask status=%s answer_id=%s confidence=%.3f latency=%.1fms source=%s",
            result.status,
            top.answer_id if top else None,
            result.confidence,
            result.latency_ms,
            body.source,
        )
        background.add_task(
            write_log,
            request.app.state.repo,
            QueryLog(
                query=body.question,
                normalized_query=normalize(body.question),
                status=result.status,
                answer_id=top.answer_id if top else None,
                confidence=result.confidence,
                latency_ms=result.latency_ms,
                source=body.source,
            ),
        )
        return AskResponse(
            status=result.status,
            answer_id=top.answer_id if top else None,
            answer=answers[top.answer_id].text if top else None,
            confidence=result.confidence,
            reason=result.reason,
            candidates=[
                CandidateOut(
                    answer_id=c.answer_id,
                    score=c.score,
                    question=st.first_question.get(c.answer_id, ""),
                    preview=preview(answers[c.answer_id].text),
                )
                for c in result.candidates
            ],
            entities=[EntityOut(**e.__dict__) for e in result.entities],
            debug=result.debug,
            latency_ms=result.latency_ms,
        )

    @app.get("/answers/{answer_id}", response_model=AnswerOut, tags=["search"])
    def get_answer(answer_id: int, request: Request) -> AnswerOut:
        st = state(request)
        answer = st.engine.corpus.answers.get(answer_id)
        if answer is None:
            raise HTTPException(status_code=404, detail=f"Answer {answer_id} not found")
        return AnswerOut(
            answer_id=answer.id,
            answer=answer.text,
            category_id=answer.category_id,
            topic_id=answer.topic_id,
            questions=st.questions_by_answer.get(answer.id, []),
        )

    @app.get("/health", response_model=HealthOut, tags=["service"])
    def health(request: Request) -> HealthOut:
        engine = state(request).engine
        return HealthOut(
            status="ok",
            answers=len(engine.corpus.answers),
            questions=len(engine.corpus.questions),
            build_seconds=round(engine.build_seconds, 3),
            data_source=settings.data_source,
            static_embeddings=settings.use_static_emb,
        )

    @app.get("/stats", response_model=StatsOut, tags=["service"])
    def stats(request: Request) -> StatsOut:
        engine = state(request).engine
        s = engine.settings
        return StatsOut(
            data={"answers": len(engine.corpus.answers), "questions": len(engine.corpus.questions)},
            config={
                "answer_threshold": s.answer_threshold,
                "suggest_threshold": s.suggest_threshold,
                "answer_min_margin": s.answer_min_margin,
                "retrievers": [ch.name for ch in engine.channels],
                "reranker": s.use_reranker,
                "synonyms": s.use_synonyms,
                "data_source": s.data_source,
            },
            eval=_eval_summary(settings),
            usage=request.app.state.repo.usage_stats(),
        )

    @app.post("/feedback", response_model=FeedbackOut, status_code=201, tags=["service"])
    def feedback(body: FeedbackIn, request: Request) -> FeedbackOut:
        if (
            body.answer_id is not None
            and body.answer_id not in state(request).engine.corpus.answers
        ):
            raise HTTPException(status_code=404, detail=f"Answer {body.answer_id} not found")
        new_id = request.app.state.repo.save_feedback(
            Feedback(
                query=body.query, answer_id=body.answer_id, helpful=body.helpful, source=body.source
            )
        )
        return FeedbackOut(id=new_id)

    @app.post("/reload", response_model=ReloadOut, tags=["service"])
    async def reload(request: Request) -> ReloadOut:
        async with request.app.state.reload_lock:
            try:
                new_state = await run_in_threadpool(build_state, settings)
            except Exception as e:  # keep serving with the old engine
                log.exception("Reload failed")
                raise HTTPException(status_code=500, detail=f"Reload failed: {e}") from e
            request.app.state.engine_state = new_state  # atomic swap
        engine = new_state.engine
        return ReloadOut(
            status="reloaded",
            answers=len(engine.corpus.answers),
            questions=len(engine.corpus.questions),
            build_seconds=round(engine.build_seconds, 3),
        )

    return app


def _eval_summary(settings: Settings) -> dict | None:
    """Headline metrics of the production engine from the last `make eval` (+ BGE-M3 if run)."""
    path = settings.reports_dir / "eval_results.json"
    if not path.is_file():
        return None
    results = json.loads(path.read_text(encoding="utf-8"))
    engine = results.get("systems", {}).get("engine", {})
    keys = (
        "validation",
        "realistic_dev",
        "test",
        "test_entity",
        "realistic",
        "ood",
        "law_injection",
    )
    summary = {"generated_at": results.get("generated_at"), "dataset": results.get("dataset")}
    summary |= {k: engine[k] for k in keys if k in engine}
    bench = settings.reports_dir / "benchmark_bge_m3.json"
    if bench.is_file():
        systems = json.loads(bench.read_text(encoding="utf-8")).get("systems", {})
        summary["bge_m3_benchmark"] = {
            name: {k: r["metrics"].get(k, {}).get("top1") for k in ("test", "realistic")}
            for name, r in systems.items()
        }
    return summary


app = create_app()
