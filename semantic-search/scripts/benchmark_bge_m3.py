"""Offline reference benchmark: full BAAI/bge-m3 dense retrieval. NOT part of the delivered system.

Runs in the bench environment (`make install-bench`, then `make benchmark`). Uses the exact same
split, query sets, answer aggregation and metrics as scripts/evaluate.py (app.eval.runner), so
the numbers are directly comparable. Writes reports/benchmark_bge_m3.{md,json}; `make eval`
merges the comparison table into reports/eval_results.md when the json exists.

Rows
- bge-m3 dense: questions + answer texts encoded with BGE-M3 (fp16), cosine search, the same
  max-aggregation to answers and weighted RRF over the two indexes as our engine.
- bge-m3 + our engine (hybrid): our full engine with BGE-M3 as an extra fusion channel, current
  settings, no re-tuning (we do not tune anything to beat BGE-M3).
- ours (phase 5 / final): our CPU systems on the same run, for the side-by-side table.
"""

import argparse
import json
import platform
import time
from datetime import UTC, datetime

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from app.core.config import Settings, get_settings
from app.core.data import Corpus
from app.core.engine import Candidate, QAEngine, QAResult, decide_status
from app.core.fusion import weighted_rrf
from app.core.normalize import normalize_script
from app.core.ranker import aggregate_max
from app.core.retrievers.base import top_k
from app.eval.runner import build_context, evaluate_system
from app.eval.systems import SYSTEMS

MODEL_NAME = "BAAI/bge-m3"
LABEL = "benchmark only, not used in the delivered system"


class BGEEncoder:
    """Loads BGE-M3 once per device and encodes script-normalized text (Latin, case kept)."""

    _models: dict[str, SentenceTransformer] = {}

    def __init__(self, device: str) -> None:
        if device not in self._models:
            kwargs = {"torch_dtype": torch.float16} if device.startswith("cuda") else {}
            self._models[device] = SentenceTransformer(
                MODEL_NAME, device=device, model_kwargs=kwargs
            )
        self.model = self._models[device]

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return self.model.encode(
            [normalize_script(t) for t in texts],
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)


class BGERetriever:
    """Retriever-protocol wrapper so BGE-M3 can join our engine as a fusion channel."""

    def __init__(self, encoder: BGEEncoder) -> None:
        self.encoder = encoder
        self.matrix: np.ndarray | None = None

    def fit(self, docs: list[str]) -> None:
        self.matrix = self.encoder.encode(docs)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        if not query.strip():
            return []
        return top_k(self.matrix @ self.encoder.encode([query])[0], k)


class BGEDenseSystem:
    """BGE-M3 dense only, with the engine's answer aggregation and question/answer RRF."""

    def __init__(self, corpus: Corpus, settings: Settings, device: str) -> None:
        self.settings = settings
        self.encoder = BGEEncoder(device)
        answer_ids = sorted(corpus.answers)
        self.lists = [
            (
                1.0,
                self.encoder.encode([q.text for q in corpus.questions]),
                [q.answer_id for q in corpus.questions],
            ),
            (
                settings.answer_index_weight,
                self.encoder.encode([corpus.answers[a].text for a in answer_ids]),
                answer_ids,
            ),
        ]

    def ask(self, query: str, k: int = 10) -> QAResult:
        s = self.settings
        q = self.encoder.encode([query])[0]
        ranked_lists, best_cos = [], {}
        for weight, matrix, doc_answer_ids in self.lists:
            ranked = aggregate_max(top_k(matrix @ q, s.candidate_pool), doc_answer_ids)
            ranked_lists.append((weight, [a for a, _ in ranked]))
            for a, cos in ranked:
                best_cos[a] = max(best_cos.get(a, 0.0), cos)
        fused = weighted_rrf(ranked_lists, s.rrf_k)
        order = sorted(fused, key=lambda a: (-fused[a], a))[:k]
        # Confidence = cosine of the best question/answer match (absolute, comparable to ours).
        candidates = [Candidate(a, float(best_cos[a])) for a in order]
        status = decide_status(
            candidates, s.answer_threshold, s.suggest_threshold, s.answer_min_margin
        )
        return QAResult(status=status, candidates=candidates)


class HybridEngine(QAEngine):
    """Our full engine plus a BGE-M3 fusion channel (current settings, untuned)."""

    device = "cuda"

    def _retriever_specs(self):
        yield from super()._retriever_specs()
        yield "bge", 1.0, lambda: BGERetriever(BGEEncoder(self.device))


def cpu_latency(corpus: Corpus, settings: Settings, queries: list[str]) -> dict:
    system = BGEDenseSystem(corpus, settings, "cpu")
    times = []
    for q in queries:
        start = time.perf_counter()
        system.ask(q)
        times.append((time.perf_counter() - start) * 1000)
    return {
        "n": len(times),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
    }


def fmt(v) -> str:
    return "–" if v is None else f"{100 * v:.1f}"


def render(results: dict) -> str:
    header = [
        "System",
        "Test Top-1",
        "Test Top-3",
        "Test MRR@10",
        "Entity Top-1",
        "Realistic-dev Top-1",
        "Realistic Top-1",
        "Realistic Top-3",
        "Unknown law → not_found",
        "Latency p50 / p95",
        "Runs on",
    ]
    rows = []
    for name, r in results["systems"].items():
        m = r["metrics"]
        dev, real, test = m.get("realistic_dev", {}), m.get("realistic", {}), m["test"]
        cells = [
            name,
            fmt(test["top1"]),
            fmt(test["top3"]),
            fmt(test["mrr10"]),
            fmt(m["test_entity"]["top1"]),
            fmt(dev.get("top1")),
            fmt(real.get("top1")),
            fmt(real.get("top3")),
            fmt(m["law_injection"].get("not_found_rate")),
            f"{test['latency_p50_ms']:.1f} / {test['latency_p95_ms']:.1f}",
            r["runs_on"],
        ]
        rows.append("| " + " | ".join(cells) + " |")
    hw = results["hardware"]
    lines = [
        f"# BGE-M3 reference benchmark ({LABEL})",
        "",
        f"Generated {results['generated_at']} · model `{MODEL_NAME}` "
        f"({results['model_size_gb']:.2f} GB fp32 weights, run in fp16 on GPU) · "
        f"GPU: {hw['gpu']} · CPU: {hw['cpu']}",
        "",
        "Same split, query sets, aggregation and metrics as `make eval`. Accuracy in %, latency "
        "per query in ms. Status-based numbers use our thresholds, which were not calibrated "
        "for BGE-M3 cosines.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *rows,
        "",
    ]
    cpu = results.get("bge_cpu_latency")
    if cpu:
        lines += [
            f"BGE-M3 dense on CPU ({cpu['n']} queries): p50 {cpu['p50_ms']:.0f} ms, "
            f"p95 {cpu['p95_ms']:.0f} ms per query.",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--cpu-sample", type=int, default=50, help="queries for CPU latency (0=skip)"
    )
    parser.add_argument("--no-hybrid", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    ctx = build_context(settings)
    HybridEngine.device = args.device
    systems = {
        "ours: phase 5 (TF-IDF + entities + re-ranker, no embeddings)": (
            SYSTEMS["6 +rerank +synonyms"],
            "CPU",
        ),
        "ours: final engine (+ Model2Vec)": (SYSTEMS["engine"], "CPU"),
        "BGE-M3 dense (benchmark only)": (
            lambda c, s: BGEDenseSystem(c, s, args.device),
            args.device.upper(),
        ),
    }
    if not args.no_hybrid:
        systems["BGE-M3 + our engine, hybrid (benchmark only)"] = (
            HybridEngine,
            args.device.upper(),
        )

    results = {
        "label": LABEL,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": MODEL_NAME,
        "model_size_gb": sum(p.numel() for p in BGEEncoder(args.device).model.parameters())
        * 4
        / 1e9,
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
            "cpu": platform.processor() or platform.machine(),
        },
        "systems": {},
    }
    for name, (factory, runs_on) in systems.items():
        report = evaluate_system(ctx, name, factory)
        results["systems"][name] = {
            "runs_on": runs_on,
            "build_seconds": report.build_seconds,
            "metrics": report.metrics,
        }
        print(
            f"{name}: test Top-1 {report.metrics['test']['top1']:.3f}, realistic Top-1 "
            f"{report.metrics.get('realistic', {}).get('top1', 0):.3f}"
        )
    if args.cpu_sample:
        sample = [q.text for q in ctx.query_sets["test"][: args.cpu_sample]]
        results["bge_cpu_latency"] = cpu_latency(ctx.index_corpus, settings, sample)

    out = settings.reports_dir
    (out / "benchmark_bge_m3.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    (out / "benchmark_bge_m3.md").write_text(render(results), encoding="utf-8")
    print(f"\nReport: {out / 'benchmark_bge_m3.md'}")


if __name__ == "__main__":
    main()
