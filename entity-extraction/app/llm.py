"""Ollama client for the details call (A) and the per-chunk task call (B).

Plan sections 7.1, 7.2 and 7.4.

Every request uses the same settings: constrained decoding against a JSON
schema (so output is always parseable and never needs a retry), temperature 0,
thinking off, a capped output length, and ``keep_alive: -1`` so the model is
never evicted between requests.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field

import httpx

from app import prompts

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

#: Section 9: small context, capped output. Raised only if measurement shows a
#: chunk plus prompt does not fit.
NUM_CTX = 4096
NUM_PREDICT_DETAILS = 256
NUM_PREDICT_TASK = 192
#: A paragraph call returns a *list* of tasks, so it needs far more room than a
#: clause call, which returns one small object. At 192 a three-task paragraph
#: was truncated mid-JSON; the parse then failed and every task in that chunk
#: was dropped silently. Measured: three tasks cost about 200 output tokens.
NUM_PREDICT_PARAGRAPH = 512

#: Matches OLLAMA_NUM_PARALLEL; the launcher sets 4 on GPU and 1 on CPU.
DEFAULT_CONCURRENCY = int(os.environ.get("OLLAMA_NUM_PARALLEL", "4"))

#: Clauses sent per call. **Default 1: one clause per call.**
#:
#: Batching is implemented and roughly halves the LLM stage (a 5-page act goes
#: from ~13 s to ~8 s, with 4x fewer prompt tokens), but it was measured and
#: rejected as a default. At 4 per call, over the 52-document set:
#:
#:     precision 87% -> 82%, executors 87% -> 83%
#:     and on the *real* lex.uz documents: precision 68% -> 58%,
#:     executors 64% -> 51%
#:
#: Synthetic documents are unaffected (their clauses are uniform), so the cost
#: appears only on real ones: asked to track four heterogeneous clauses at
#: once, the model drifts and attributes a deadline or an executor to the wrong
#: clause. The length check in `clause_batch` catches a *missing* entry; it
#: cannot catch a *swapped* one.
#:
#: Set CLAUSE_BATCH_SIZE=4 to trade that accuracy for speed.
CLAUSE_BATCH_SIZE = int(os.environ.get("CLAUSE_BATCH_SIZE", "1"))

REQUEST_TIMEOUT = 300.0


@dataclass
class Usage:
    """Token counts and durations Ollama reports, for `eval/bench.py`."""

    prompt_eval_count: int = 0
    prompt_eval_duration: int = 0
    eval_count: int = 0
    eval_duration: int = 0
    total_duration: int = 0
    calls: int = 0

    def add(self, response: dict) -> None:
        for name in (
            "prompt_eval_count", "prompt_eval_duration",
            "eval_count", "eval_duration", "total_duration",
        ):
            setattr(self, name, getattr(self, name) + response.get(name, 0) or 0)
        self.calls += 1

    @property
    def tokens_per_second(self) -> float:
        return self.eval_count / (self.eval_duration / 1e9) if self.eval_duration else 0.0


@dataclass
class Client:
    model: str
    url: str = OLLAMA_URL
    num_ctx: int = NUM_CTX
    concurrency: int = DEFAULT_CONCURRENCY
    #: Set to 0 to force CPU, for the CPU-vs-GPU comparison in section 9.
    num_gpu: int | None = None
    usage: Usage = field(default_factory=Usage)

    async def chat(
        self,
        client: httpx.AsyncClient,
        messages: list[dict],
        schema: dict,
        num_predict: int,
    ) -> dict:
        """One constrained-decoding call. Returns the parsed JSON object."""
        options = {
            "temperature": 0,
            "num_ctx": self.num_ctx,
            "num_predict": num_predict,
        }
        if self.num_gpu is not None:
            options["num_gpu"] = self.num_gpu

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": schema,
            "think": False,  # thinking models must not reason here
            "keep_alive": -1,
            "options": options,
        }
        response = await client.post(
            f"{self.url}/api/chat", json=payload, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        body = response.json()
        self.usage.add(body)
        return json.loads(body["message"]["content"])

    # --- the two calls ----------------------------------------------------

    async def details(self, header: str, footer: str, fields: list[str]) -> dict:
        schema = prompts.details_schema(fields)
        async with httpx.AsyncClient() as client:
            return await self.chat(
                client,
                prompts.details_messages(header, footer),
                schema,
                NUM_PREDICT_DETAILS,
            )

    async def clause(self, client: httpx.AsyncClient, text: str, context: str = "") -> dict:
        return await self.chat(
            client,
            prompts.clause_messages(text, context),
            prompts.CLAUSE_SCHEMA,
            NUM_PREDICT_TASK,
        )

    async def clause_batch(
        self, client: httpx.AsyncClient, items: list[tuple[str, str]]
    ) -> list[dict]:
        """Several clauses in one call, returning one result per clause.

        Raises if the model returns the wrong number of results, so the caller
        can fall back to individual calls rather than silently misattributing
        deadlines and executors across clauses.
        """
        if len(items) == 1:
            return [await self.clause(client, items[0][0], items[0][1])]

        answer = await self.chat(
            client,
            prompts.batch_messages(items),
            prompts.BATCH_SCHEMA,
            NUM_PREDICT_TASK * len(items),
        )
        results = answer.get("results") or []
        if len(results) != len(items):
            raise ValueError(
                f"batch returned {len(results)} results for {len(items)} clauses"
            )
        return results

    async def paragraph(self, client: httpx.AsyncClient, text: str) -> dict:
        return await self.chat(
            client,
            prompts.paragraph_messages(text),
            prompts.PARAGRAPH_SCHEMA,
            NUM_PREDICT_PARAGRAPH,
        )

    async def clauses(self, items: list[tuple[str, str]]) -> list[dict | None]:
        """Run many clauses concurrently, bounded by ``concurrency``.

        A chunk that fails returns ``None`` rather than losing the whole
        document: one bad clause should not fail an upload.
        """
        limit = asyncio.Semaphore(self.concurrency)

        async with httpx.AsyncClient() as client:
            async def one(text: str, context: str):
                async with limit:
                    try:
                        return await self.clause(client, text, context)
                    except Exception:
                        return None

            return await asyncio.gather(*(one(t, c) for t, c in items))


async def wait_until_ready(url: str = OLLAMA_URL, attempts: int = 60) -> bool:
    """Poll ``/api/tags`` until Ollama answers (plan section 8.3, step 1)."""
    async with httpx.AsyncClient() as client:
        for _ in range(attempts):
            try:
                if (await client.get(f"{url}/api/tags", timeout=5)).status_code == 200:
                    return True
            except Exception:
                pass
            await asyncio.sleep(1)
    return False
