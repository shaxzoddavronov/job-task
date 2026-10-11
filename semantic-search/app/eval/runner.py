"""Evaluation runner: builds each system on the index split and scores every query set.

Shared by scripts/evaluate.py and the offline BGE-M3 benchmark so both use the exact same
split, query sets and metrics.

Query sets
- validation / test: held-out paraphrases (random split, seed from settings). Optimistic,
  because most paraphrases are template variants of one original question (Phase 1 audit).
- test_entity: test questions that contain an entity (law, article, number, ORG, TERM).
- law_injection: test questions with an invented law number appended. A law that exists
  nowhere in the data must not be answered: the expected status is not_found.
- law_swap: test questions whose real law number is replaced by another existing one.
- org_swap: test questions whose ORG/TERM entity is replaced by a different entry of the
  same type. Diagnostic only: we report how often the original answer is still given.
- ood: out-of-domain questions (data/ood_questions.txt); expected status not_found.
- realistic: hand-written queries (data/eval_realistic.csv), run against a system built on
  the full corpus because none of these queries come from the data.
"""

import logging
import random
import time
from dataclasses import dataclass, field

import pandas as pd

from app.core.config import Settings
from app.core.data import Corpus, FileSource, Question, build_corpus
from app.core.engine import QASystem
from app.core.entities import EntityExtractor
from app.core.normalize import normalize_script
from app.eval.metrics import QueryRecord, status_rates, summarize
from app.eval.split import Split, split_questions
from app.eval.systems import SYSTEMS, SystemFactory

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalQuery:
    text: str
    gold: int | None
    kind: str = ""


@dataclass
class SystemReport:
    name: str
    build_seconds: float
    metrics: dict[str, dict]
    records: dict[str, list[QueryRecord]] = field(repr=False)


@dataclass
class EvalContext:
    settings: Settings
    corpus: Corpus
    split: Split
    extractor: EntityExtractor
    query_sets: dict[str, list[EvalQuery]]
    realistic: list[EvalQuery]

    @property
    def index_corpus(self) -> Corpus:
        return self.corpus.with_questions(self.split.index)


def run_queries(
    system: QASystem, queries: list[EvalQuery], extractor: EntityExtractor, k: int
) -> list[QueryRecord]:
    records = []
    for q in queries:
        start = time.perf_counter()
        result = system.ask(q.text, k=k)
        latency_ms = (time.perf_counter() - start) * 1000
        records.append(
            QueryRecord(
                query=q.text,
                gold=q.gold,
                status=result.status,
                ranked=[(c.answer_id, c.score) for c in result.candidates],
                latency_ms=latency_ms,
                entity_tokens=sorted(extractor.process(q.text).tokens),
                kind=q.kind,
            )
        )
    return records


# --- Query set construction --------------------------------------------------------------


def _as_queries(questions: list[Question]) -> list[EvalQuery]:
    return [EvalQuery(q.text, q.answer_id) for q in questions]


def _known_laws(corpus: Corpus, extractor: EntityExtractor) -> set[str]:
    texts = [a.text for a in corpus.answers.values()] + [q.text for q in corpus.questions]
    return {e.value for t in texts for e in extractor.process(t).entities if e.type == "LAW"}


def law_injection_queries(test: list[Question], known_laws: set[str], seed: int) -> list[EvalQuery]:
    rng = random.Random(seed)
    queries = []
    for q in test:
        while True:
            kind, number = rng.choice(["pq", "pf"]), rng.randint(1000, 9999)
            if f"{kind}_{number}" not in known_laws:
                break
        queries.append(EvalQuery(f"{q.text} ({kind.upper()}-{number} bo'yicha)", q.answer_id))
    return queries


def law_swap_queries(
    test: list[Question], known_laws: set[str], extractor: EntityExtractor, seed: int
) -> list[EvalQuery]:
    rng = random.Random(seed)
    queries = []
    for q in test:
        script = normalize_script(q.text)
        laws = [e for e in extractor.process(q.text).entities if e.type == "LAW"]
        for law in laws[:1]:
            kind = law.value.split("_")[0]
            others = sorted(v for v in known_laws if v.startswith(f"{kind}_") and v != law.value)
            if others:
                number = rng.choice(others).split("_")[1]
                swapped = script.replace(law.surface, f"{kind.upper()}-{number}", 1)
                queries.append(EvalQuery(swapped, q.answer_id))
    return queries


def org_swap_queries(
    test: list[Question], extractor: EntityExtractor, seed: int
) -> list[EvalQuery]:
    rng = random.Random(seed)
    entries = extractor.gazetteer.entries
    queries = []
    for q in test:
        script = normalize_script(q.text)
        named = [e for e in extractor.process(q.text).entities if e.type in ("ORG", "TERM")]
        for ent in named[:1]:
            others = [g for g in entries if g.type == ent.type and g.id != ent.value]
            if others and ent.surface in script:
                replacement = rng.choice(others).aliases[0]
                queries.append(EvalQuery(script.replace(ent.surface, replacement, 1), q.answer_id))
    return queries


def load_ood(settings: Settings) -> list[EvalQuery]:
    path = settings.ood_questions_path
    if not path.is_file():
        log.warning("%s not found; skipping the OOD check", path)
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [EvalQuery(t.strip(), None) for t in lines if t.strip() and not t.startswith("#")]


def load_realistic(settings: Settings, corpus: Corpus) -> list[EvalQuery]:
    path = settings.eval_realistic_path
    if not path.is_file():
        log.warning("%s not found; skipping the realistic set", path)
        return []
    df = pd.read_csv(path, encoding="utf-8")
    unknown = sorted(set(df["answer_id"]) - set(corpus.answers))
    if unknown:
        raise ValueError(f"{path.name} references unknown answer ids: {unknown}")
    return [
        EvalQuery(str(r.query), int(r.answer_id), str(getattr(r, "kind", "") or ""))
        for r in df.itertuples()
    ]


def build_context(settings: Settings) -> EvalContext:
    raw = FileSource(settings.answers_path, settings.questions_path).load_raw()
    corpus = build_corpus(raw)
    split = split_questions(corpus.questions, settings.eval_seed)
    extractor = EntityExtractor.from_settings(settings)
    known_laws = _known_laws(corpus, extractor)
    seed = settings.eval_seed
    query_sets = {
        "validation": _as_queries(split.validation),
        "test": _as_queries(split.test),
        "law_injection": law_injection_queries(split.test, known_laws, seed),
        "law_swap": law_swap_queries(split.test, known_laws, extractor, seed),
        "org_swap": org_swap_queries(split.test, extractor, seed),
        "ood": load_ood(settings),
    }
    return EvalContext(
        settings, corpus, split, extractor, query_sets, load_realistic(settings, corpus)
    )


# --- Running a system ----------------------------------------------------------------------


def _probe_metrics(records: list[QueryRecord]) -> dict:
    """For perturbed queries: how often the original (now wrong) answer is given confidently."""
    out = status_rates(records)
    if records:
        out["kept_original_answer_rate"] = sum(
            r.status == "answer" and r.top1 == r.gold for r in records
        ) / len(records)
    return out


def evaluate_system(ctx: EvalContext, name: str, factory: SystemFactory) -> SystemReport:
    k = ctx.settings.eval_top_k
    start = time.perf_counter()
    system = factory(ctx.index_corpus, ctx.settings)
    build_seconds = time.perf_counter() - start

    records = {
        set_name: run_queries(system, queries, ctx.extractor, k)
        for set_name, queries in ctx.query_sets.items()
    }
    metrics = {
        "validation": summarize(records["validation"]),
        "test": summarize(records["test"]),
        "test_entity": summarize([r for r in records["test"] if r.entity_tokens]),
        "law_injection": _probe_metrics(records["law_injection"]),
        "law_swap": _probe_metrics(records["law_swap"]),
        "org_swap": _probe_metrics(records["org_swap"]),
        "ood": status_rates(records["ood"]),
    }
    if ctx.realistic:
        full_system = factory(ctx.corpus, ctx.settings)
        records["realistic"] = run_queries(full_system, ctx.realistic, ctx.extractor, k)
        metrics["realistic"] = summarize(records["realistic"])
        kinds = sorted({r.kind for r in records["realistic"] if r.kind})
        metrics["realistic_by_kind"] = {
            kind: summarize([r for r in records["realistic"] if r.kind == kind]) for kind in kinds
        }
    return SystemReport(name, build_seconds, metrics, records)


def evaluate(settings: Settings, names: list[str] | None = None) -> tuple[EvalContext, list]:
    ctx = build_context(settings)
    names = names or list(SYSTEMS)
    unknown = [n for n in names if n not in SYSTEMS]
    if unknown:
        raise ValueError(f"Unknown systems {unknown}; available: {list(SYSTEMS)}")
    return ctx, [evaluate_system(ctx, n, SYSTEMS[n]) for n in names]
