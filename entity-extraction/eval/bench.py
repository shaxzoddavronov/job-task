"""Latency and throughput benchmark: baseline against optimized.

Plan sections 9 and 11 (Phase 10).

Two configurations are compared on the same documents:

* **baseline** — the obvious implementation. One call carrying the whole
  document, calls issued sequentially, no output cap, Ollama's default context,
  and a server with flash attention off and an f16 KV cache.
* **optimized** — what the app actually does. Per-chunk calls fanned out
  concurrently, a byte-stable prompt prefix so the cache is reused, a small
  context, capped output, constrained decoding, and a server with flash
  attention on and a q8_0 KV cache.

Flash attention and the KV cache type are server settings rather than request
settings, so the two configurations point at two Ollama instances. Start the
baseline one with::

    docker run -d --gpus all -v ollama:/root/.ollama -p 11435:11434 \\
      -e OLLAMA_FLASH_ATTENTION=0 -e OLLAMA_KV_CACHE_TYPE=f16 \\
      -e OLLAMA_NUM_PARALLEL=1 --name ollama-baseline ollama/ollama

Two traps this benchmark had to be fixed for, both of which made the
optimizations look better than they are:

* a cold model load landing inside the timed window, and
* Ollama reloading the model whenever ``num_ctx`` or the server changes, which
  charged every "without" row for a reload and made flash attention appear 4.8x
  faster than it is.

Every configuration is therefore warmed before it is timed.

Usage::

    python eval/bench.py --model qwen3:4b
    python eval/bench.py --model qwen3:4b --cpu

The interpretation section at the end of ``eval/results/bench.md`` is written
by hand after a run; regenerating the tables does not replace it.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import httpx

from app import prompts
from app.chunker import chunk
from app.llm import Client, Usage
from app.script_utils import to_latin_with_offsets
from doc_cache import cached_extract

RESULTS = ROOT / "eval" / "results" / "bench.md"

TUNED_URL = "http://localhost:11434"
BASELINE_URL = "http://localhost:11435"

#: A spread of shapes: short Latin letter, Cyrillic action-plan table, scanned
#: Cyrillic order, and a real two-page act.
DOCUMENTS = [
    ROOT / "eval/data/synthetic/doc_01_lat.pdf",
    ROOT / "eval/data/synthetic/doc_03_cyr.pdf",
    ROOT / "eval/data/synthetic/doc_02_cyr_scan.pdf",
    ROOT / "eval/data/lexuz/lexuz_8519155_lat.pdf",
]

#: Baseline asks one question about the whole document, so it needs room for
#: the document in and a list of tasks out.
BASELINE_NUM_CTX = 8192
BASELINE_NUM_PREDICT = 2048

BASELINE_SYSTEM = """You read an official Uzbek document and list every task it assigns.

Return JSON with a "tasks" array. One entry per task, each with:
- task: the assigned work, copied from the document.
- deadline: the time limit exactly as written, or null.
- executors: the organisations or officials responsible, exactly as written.

Copy values exactly as written. If a value is not present, return null or an empty list."""


def baseline_messages(document_text: str) -> list[dict]:
    """One call, whole document, no few-shot prefix to cache."""
    return [
        {"role": "system", "content": BASELINE_SYSTEM},
        {"role": "user", "content": f"Document:\n{document_text}"},
    ]


async def run_baseline(pdf: Path, model: str, num_gpu: int | None) -> dict:
    document = cached_extract(pdf)
    latin, _ = to_latin_with_offsets(document.text)

    client = Client(
        model=model, url=BASELINE_URL, num_ctx=BASELINE_NUM_CTX,
        concurrency=1, num_gpu=num_gpu,
    )
    started = time.time()
    async with httpx.AsyncClient() as http:
        try:
            answer = await client.chat(
                http, baseline_messages(latin),
                prompts.PARAGRAPH_SCHEMA, BASELINE_NUM_PREDICT,
            )
            tasks = len(answer.get("tasks") or [])
        except Exception as exc:
            return {"error": f"{type(exc).__name__}", "detail": str(exc)[:80]}
    return _summary(time.time() - started, client.usage, tasks)


async def run_optimized(pdf: Path, model: str, num_gpu: int | None) -> dict:
    document = cached_extract(pdf)
    pieces = chunk(document)

    client = Client(model=model, url=TUNED_URL, concurrency=4, num_gpu=num_gpu)
    limit = asyncio.Semaphore(client.concurrency)
    started = time.time()

    async with httpx.AsyncClient() as http:
        async def one(piece):
            async with limit:
                text, _ = to_latin_with_offsets(piece.text)
                context, _ = to_latin_with_offsets(piece.context)
                try:
                    if piece.kind == "paragraph":
                        return await client.paragraph(http, text)
                    return await client.clause(http, text, context)
                except Exception:
                    return None

        answers = await asyncio.gather(*(one(p) for p in pieces))

    tasks = 0
    for piece, answer in zip(pieces, answers):
        if not answer:
            continue
        if piece.kind == "paragraph":
            tasks += len(answer.get("tasks") or [])
        elif answer.get("is_task"):
            tasks += 1
    return _summary(time.time() - started, client.usage, tasks)


def _summary(elapsed: float, usage: Usage, tasks: int) -> dict:
    prompt_seconds = usage.prompt_eval_duration / 1e9 if usage.prompt_eval_duration else 0
    return {
        "seconds": round(elapsed, 2),
        "calls": usage.calls,
        "prompt_tokens": usage.prompt_eval_count,
        "output_tokens": usage.eval_count,
        "output_tps": round(usage.tokens_per_second, 1),
        "prompt_tps": round(usage.prompt_eval_count / prompt_seconds, 1) if prompt_seconds else 0,
        "tasks": tasks,
    }


async def warm(model: str, url: str, num_gpu: int | None) -> None:
    client = Client(model=model, url=url, num_gpu=num_gpu)
    async with httpx.AsyncClient() as http:
        with __import__("contextlib").suppress(Exception):
            await client.clause(http, "1. hisobot taqdim etilsin.")


def table(rows: list[dict], device: str) -> list[str]:
    out = [
        f"### {device}",
        "",
        "| Document | Config | Wall s | LLM calls | Prompt tok | Output tok "
        "| Output tok/s | Raw tasks |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        for config in ("baseline", "optimized"):
            data = row[config]
            if "error" in data:
                out.append(
                    f"| `{row['name']}` | {config} | — | — | — | — | — | "
                    f"{data['error']} |"
                )
                continue
            out.append(
                f"| `{row['name']}` | {config} | {data['seconds']} | {data['calls']} "
                f"| {data['prompt_tokens']} | {data['output_tokens']} "
                f"| {data['output_tps']} | {data['tasks']} |"
            )
    totals = {}
    for config in ("baseline", "optimized"):
        good = [r[config] for r in rows if "error" not in r[config]]
        totals[config] = {
            "seconds": round(sum(d["seconds"] for d in good), 2),
            "output": sum(d["output_tokens"] for d in good),
            "calls": sum(d["calls"] for d in good),
        }
    speedup = (
        totals["baseline"]["seconds"] / totals["optimized"]["seconds"]
        if totals["optimized"]["seconds"]
        else 0
    )
    out += [
        "",
        f"**Totals** — baseline {totals['baseline']['seconds']}s in "
        f"{totals['baseline']['calls']} calls producing "
        f"{totals['baseline']['output']} output tokens; optimized "
        f"{totals['optimized']['seconds']}s in {totals['optimized']['calls']} calls "
        f"producing {totals['optimized']['output']} output tokens. "
        f"**{speedup:.1f}x faster**, "
        f"{totals['baseline']['output'] / max(totals['optimized']['output'], 1):.1f}x "
        "fewer output tokens.",
        "",
    ]
    return out


# --- individual techniques (plan section 9 asks for each to be measured) ---


async def _chunk_workload(pdf: Path):
    document = cached_extract(pdf)
    pieces = chunk(document)
    items = []
    for piece in pieces:
        text, _ = to_latin_with_offsets(piece.text)
        context, _ = to_latin_with_offsets(piece.context)
        items.append((piece.kind, text, context))
    return items


async def _run_clauses(client: Client, items, concurrency: int, mangle_prefix=False):
    """Run a chunk workload, optionally defeating the prompt cache.

    One untimed call runs first. Ollama reloads the model whenever `num_ctx`
    or the server changes, and without this the reload lands inside the timed
    window — which made flash attention look 4.8x faster than it is.
    """
    async with httpx.AsyncClient() as http:
        with __import__("contextlib").suppress(Exception):
            await client.chat(
                http, prompts.clause_messages("1. hisobot taqdim etilsin."),
                prompts.CLAUSE_SCHEMA, 192,
            )
    client.usage = Usage()

    limit = asyncio.Semaphore(concurrency)
    started = time.time()
    async with httpx.AsyncClient() as http:
        async def one(index, item):
            kind, text, context = item
            async with limit:
                messages = prompts.clause_messages(text, context)
                if mangle_prefix:
                    # A unique system prompt per call, so no prefix is shared.
                    messages = [dict(messages[0]), *messages[1:]]
                    messages[0]["content"] += f"\n(request {index})"
                with __import__("contextlib").suppress(Exception):
                    return await client.chat(
                        http, messages, prompts.CLAUSE_SCHEMA, 192
                    )
        await asyncio.gather(*(one(i, item) for i, item in enumerate(items)))
    return time.time() - started


async def techniques(model: str, num_gpu: int | None) -> list[str]:
    """Measure the section 9 techniques one at a time, on one workload."""
    pdf = ROOT / "eval/data/lexuz/lexuz_8519155_lat.pdf"
    if not pdf.exists():
        return []
    items = [i for i in await _chunk_workload(pdf) if i[0] != "paragraph"]
    rows = []

    def client(url=TUNED_URL, **kwargs):
        return Client(model=model, url=url, num_gpu=num_gpu, **kwargs)

    # Concurrency.
    c = client()
    serial = await _run_clauses(c, items, 1)
    c = client()
    parallel = await _run_clauses(c, items, 4)
    rows.append(("Concurrency (4 vs 1)", f"{serial:.2f}s", f"{parallel:.2f}s",
                 f"{serial / parallel:.1f}x" if parallel else "—"))

    # Prefix caching: identical prefix versus a unique one per call.
    c = client()
    cached = await _run_clauses(c, items, 1)
    cached_prompt_tps = c.usage.prompt_eval_count / (c.usage.prompt_eval_duration / 1e9 or 1)
    c = client()
    uncached = await _run_clauses(c, items, 1, mangle_prefix=True)
    uncached_prompt_tps = c.usage.prompt_eval_count / (c.usage.prompt_eval_duration / 1e9 or 1)
    rows.append((
        "Prefix caching (stable vs unique prefix)",
        f"{uncached:.2f}s ({uncached_prompt_tps:.0f} prompt tok/s)",
        f"{cached:.2f}s ({cached_prompt_tps:.0f} prompt tok/s)",
        f"{uncached / cached:.1f}x" if cached else "—",
    ))

    # Flash attention + quantized KV cache (server settings).
    c = client(url=BASELINE_URL)
    plain = await _run_clauses(c, items, 1)
    c = client(url=TUNED_URL)
    tuned = await _run_clauses(c, items, 1)
    rows.append(("Flash attention + q8_0 KV (server)", f"{plain:.2f}s",
                 f"{tuned:.2f}s", f"{plain / tuned:.1f}x" if tuned else "—"))

    # Small context.
    c = client(num_ctx=8192)
    wide = await _run_clauses(c, items, 1)
    c = client(num_ctx=4096)
    narrow = await _run_clauses(c, items, 1)
    rows.append(("Context 4096 vs 8192", f"{wide:.2f}s", f"{narrow:.2f}s",
                 f"{wide / narrow:.1f}x" if narrow else "—"))

    out = [
        "### Techniques measured individually",
        "",
        "One workload: the clause chunks of `lexuz_8519155_lat.pdf`, run "
        "sequentially unless the row says otherwise.",
        "",
        "| Technique | Without | With | Gain |",
        "|---|---|---|---:|",
    ]
    out += [f"| {name} | {without} | {with_} | {gain} |"
            for name, without, with_, gain in rows]
    out.append("")
    return out


async def main_async(model: str, cpu: bool) -> int:
    num_gpu = 0 if cpu else None
    device = "CPU (`options.num_gpu = 0`)" if cpu else "GPU"

    print(f"warming up on {device} ...", flush=True)
    for _ in range(2):  # load, then settle
        await warm(model, TUNED_URL, num_gpu)
        await warm(model, BASELINE_URL, num_gpu)
    # A whole-document call is the baseline's real shape; warm that too, so the
    # first document measured is not charged for a context resize.
    if DOCUMENTS and DOCUMENTS[0].exists():
        await run_baseline(DOCUMENTS[0], model, num_gpu)

    rows = []
    for pdf in DOCUMENTS:
        if not pdf.exists():
            continue
        print(f"  {pdf.name}", flush=True)
        base = await run_baseline(pdf, model, num_gpu)
        opt = await run_optimized(pdf, model, num_gpu)
        print(f"    baseline  {base}")
        print(f"    optimized {opt}")
        rows.append({"name": pdf.name, "baseline": base, "optimized": opt})

    body = [
        "# Inference benchmark (plan sections 9 and 11)",
        "",
        f"Model `{model}`. Extraction is cached and excluded: this measures the "
        "LLM stage only, which is the part the optimizations touch.",
        "",
        "`Raw tasks` is what each configuration returned before grounding and "
        "deadline parsing. It is here to show the two are doing comparable "
        "work, not as an accuracy measure — accuracy is in "
        "`eval/results/eval.md`.",
        "",
        "**baseline** — one call with the whole document, sequential, no output "
        f"cap, `num_ctx {BASELINE_NUM_CTX}`, server with flash attention off and "
        "an f16 KV cache.",
        "",
        "**optimized** — per-chunk calls fanned out 4 at a time, byte-stable "
        "prompt prefix, `num_ctx 4096`, capped output, constrained decoding, "
        "server with flash attention on and a q8_0 KV cache.",
        "",
        *table(rows, device),
        *(await techniques(model, num_gpu)),
    ]
    text = "\n".join(body) + "\n"

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    previous = RESULTS.read_text(encoding="utf-8") if RESULTS.exists() else ""
    if previous and device.split()[0] not in previous:
        text = previous.rstrip() + "\n\n" + "\n".join(table(rows, device)) + "\n"
    RESULTS.write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"written to {RESULTS}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--cpu", action="store_true", help="force CPU with num_gpu=0")
    args = parser.parse_args()
    return asyncio.run(main_async(args.model, args.cpu))


if __name__ == "__main__":
    raise SystemExit(main())
