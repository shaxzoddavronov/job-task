"""Hardware tier -> model, pull, warm-up, /api/ps verification.

Plan section 8.

The user never configures this: the launcher reads free VRAM from the host with
nvidia-smi and passes it in as HOST_GPU_FREE_MB, and everything else follows
from the tier table below.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

import httpx

logger = logging.getLogger("uzdoc.model")

#: Tier thresholds in MB of *free* VRAM. Each tier leaves room for the KV
#: cache on top of the weights (num_ctx x OLLAMA_NUM_PARALLEL).
#:
#: Models chosen in Phase 4 from `eval/results/model_comparison.md`, scored on
#: 80 labelled clauses as is_task / deadline / executors:
#:
#:   qwen3:4b       Q4_K_M  2.5 GB   96 /  90 / 85   172 tok/s
#:   qwen3:8b       Q4_K_M  5.2 GB   99 / 100 / 85   119 tok/s
#:   qwen3:8b-q8_0          8.9 GB   99 / 100 / 87    78 tok/s
#:
#: Two candidates named in section 8.2 were tested and rejected on the numbers:
#:
#: * `qwen3:30b-a3b` (18 GB) scored 92 / 80 / 90 — *below* the 4B on two of
#:   three metrics for seven times the weights. Bigger is not better for this
#:   task, so T3 spends its headroom on precision (Q8_0) instead.
#: * The Uzbek-tuned models scored 90 / 58 / 58 (`uzbek-llama-3.1-8B-Uz`) and
#:   86 / 47 / 43 (`alloma-8b-q4`). They are Llama-3.1 fine-tunes for general
#:   Uzbek chat and follow constrained-extraction instructions poorly.
#:   `uzlm/alloma-3B/8B-Instruct` have no GGUF at all and cannot be loaded.
TIERS = {
    "T0": {
        "min_free_vram_mb": 0,
        "model": "qwen3:4b",
        "note": "CPU only",
    },
    "T1": {
        "min_free_vram_mb": 4096,
        "model": "qwen3:4b",
        "note": "2.5 GB of weights leaves KV headroom on a 6 GB card",
    },
    "T2": {
        "min_free_vram_mb": 10240,
        "model": "qwen3:8b",
        "note": "best accuracy per GB",
    },
    "T3": {
        "min_free_vram_mb": 22528,
        "model": "qwen3:8b-q8_0",
        "note": "same 8B at higher precision rather than a larger model",
    },
}


def select_tier(free_vram_mb: int) -> str:
    """The highest tier the hardware can hold. No GPU (0 MB) gives T0."""
    best = "T0"
    for name, tier in TIERS.items():
        if free_vram_mb >= tier["min_free_vram_mb"]:
            best = max(best, name)
    return best


def model_for(tier: str) -> str:
    return TIERS[tier]["model"]


#: Order from smallest to largest, so stepping down is just an index walk.
TIER_ORDER = ["T0", "T1", "T2", "T3"]

#: Section 8.3 step 4: below this share of the weights resident in VRAM, the
#: model has spilled to system memory and the tier is too ambitious.
VRAM_RESIDENT_RATIO = 0.95

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


def host_gpu_free_mb() -> int:
    """Free VRAM as the launcher measured it on the *host* (section 8.1)."""
    try:
        return int(os.environ.get("HOST_GPU_FREE_MB", "0") or 0)
    except ValueError:
        return 0


async def wait_for_ollama(url: str = OLLAMA_URL, attempts: int = 120) -> bool:
    """Step 1: wait until Ollama answers."""
    async with httpx.AsyncClient() as client:
        for _ in range(attempts):
            try:
                if (await client.get(f"{url}/api/tags", timeout=5)).status_code == 200:
                    return True
            except Exception:
                pass
            await asyncio.sleep(1)
    return False


async def installed_models(url: str = OLLAMA_URL) -> set[str]:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{url}/api/tags", timeout=30)
        response.raise_for_status()
        return {m["name"] for m in response.json().get("models", [])}


async def pull(model: str, url: str = OLLAMA_URL) -> None:
    """Step 2: pull a missing model, logging progress as it streams."""
    logger.info("pulling %s (first start only)", model)
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST", f"{url}/api/pull", json={"model": model, "stream": True}
        ) as response:
            response.raise_for_status()
            last = ""
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    status = json.loads(line).get("status", "")
                except json.JSONDecodeError:
                    continue
                if status and status != last:
                    logger.info("  %s: %s", model, status)
                    last = status


async def warm_up(model: str, url: str = OLLAMA_URL) -> None:
    """Step 3: one real call, so the shared prompt prefix is already cached."""
    from app.llm import Client

    import httpx as _httpx

    client = Client(model=model, url=url)
    async with _httpx.AsyncClient() as http:
        await client.clause(http, "1. hisobot taqdim etilsin.")


async def resident_in_vram(model: str, url: str = OLLAMA_URL) -> tuple[bool, str]:
    """Step 4: did the model actually fit in VRAM, or did it spill to CPU?"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{url}/api/ps", timeout=30)
        response.raise_for_status()
        for entry in response.json().get("models", []):
            if entry.get("name") != model:
                continue
            size = entry.get("size") or 0
            in_vram = entry.get("size_vram") or 0
            if not size:
                return True, "size not reported"
            share = in_vram / size
            return (
                share >= VRAM_RESIDENT_RATIO,
                f"{in_vram / 1e9:.1f} of {size / 1e9:.1f} GB in VRAM ({share:.0%})",
            )
    return True, "not listed by /api/ps"


async def prepare(state, url: str = OLLAMA_URL) -> None:
    """Run the whole startup lifecycle of section 8.3, stepping down on a spill."""
    state.status = "waiting for ollama"
    if not await wait_for_ollama(url):
        state.status = "error"
        state.reason = f"Ollama did not answer at {url}"
        return

    free_mb = host_gpu_free_mb()
    tier = select_tier(free_mb)

    while True:
        model = model_for(tier)
        state.tier, state.model = tier, model

        state.status = "pulling"
        state.reason = f"preparing {model} for tier {tier}"
        if model not in await installed_models(url):
            await pull(model, url)

        state.status = "warming up"
        await warm_up(model, url)

        # A GPU tier that spilled to system memory would be far slower than the
        # tier below it, so step down and try again.
        if tier != "T0":
            fits, detail = await resident_in_vram(model, url)
            if not fits:
                index = TIER_ORDER.index(tier)
                if index > 0:
                    logger.warning(
                        "%s spilled out of VRAM (%s); stepping down to %s",
                        model, detail, TIER_ORDER[index - 1],
                    )
                    tier = TIER_ORDER[index - 1]
                    continue

        state.status = "ready"
        state.reason = (
            f"detected {free_mb} MB free VRAM"
            if free_mb
            else "no GPU detected, running on CPU"
        )
        logger.info("Using %s (tier %s, %s)", model, tier, state.reason)
        return
