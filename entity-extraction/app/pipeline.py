"""End-to-end pipeline: a PDF in, the response of section 12 out.

Plan section 1::

    PDF -> extract -> script detect -> normalize + translit (offset map)
        -> regex (number, date) -> LLM A (details) -> chunk
        -> LLM B per chunk (concurrent) -> grounding -> deadlines -> JSON

Debug CLI::

    python -m app.pipeline path/to/file.pdf
"""

from __future__ import annotations

import asyncio
import json
import sys
import time

import httpx

from app.chunker import Chunk, chunk
from app.deadlines import parse_deadline
from app.extract_text import extract
from app.grounding import TASK_THRESHOLD, ground, ground_all, source_for
from app.llm import CLAUSE_BATCH_SIZE, Client
from app.regex_rules import find_document_date, find_document_number
from app.schemas import Analysis, DateField, Details, Field_, Signatory, SourceInfo, Task, Timings
from app.script_utils import detect_script, to_latin_with_offsets

#: Section 7.2: the details call sees only the top of page 1 and the end of the
#: document, not the whole text.
HEADER_CHARS = 1500
FOOTER_CHARS = 1000


def _regex_details(document) -> tuple[Field_, DateField]:
    page_one = document.page_text(1)
    latin, offsets = to_latin_with_offsets(page_one)
    number = find_document_number(latin, page_one, offsets)
    date = find_document_date(latin, page_one, offsets)
    return (
        Field_(value=number["raw"], method="regex") if number else Field_(),
        DateField(value=date["value"], raw=date["raw"], method="regex") if date else DateField(),
    )


async def _details(client: Client, document, missing: list[str]):
    """Call A — only the fields regex did not supply (section 7.2)."""
    if not missing:
        return {}, None
    header = document.page_text(1)[:HEADER_CHARS]
    footer = document.text[-FOOTER_CHARS:]
    header_latin, _ = to_latin_with_offsets(header)
    footer_latin, _ = to_latin_with_offsets(footer)
    try:
        answer = await client.details(header_latin, footer_latin, missing)
    except Exception:
        return {}, None
    return answer, [source_for(header), source_for(footer)]


def _groups(chunks: list[Chunk], batch_size: int) -> list[list[int]]:
    """Group chunk indices into LLM calls.

    Consecutive clause and table-row chunks share a call; a paragraph always
    gets its own, because it returns a variable-length list of tasks and cannot
    be aligned positionally with anything else.
    """
    groups: list[list[int]] = []
    pending: list[int] = []
    for index, piece in enumerate(chunks):
        if piece.kind == "paragraph":
            if pending:
                groups.append(pending)
                pending = []
            groups.append([index])
            continue
        pending.append(index)
        if len(pending) >= batch_size:
            groups.append(pending)
            pending = []
    if pending:
        groups.append(pending)
    return groups


async def _tasks(client: Client, chunks: list[Chunk], document_text: str):
    """Call B — every chunk, batched where possible, fanned out concurrently."""
    limit = asyncio.Semaphore(client.concurrency)
    answers: list[dict | None] = [None] * len(chunks)

    def latin(piece: Chunk) -> tuple[str, str]:
        return (
            to_latin_with_offsets(piece.text)[0],
            to_latin_with_offsets(piece.context)[0],
        )

    async with httpx.AsyncClient() as http:
        async def run(group: list[int]) -> None:
            async with limit:
                first = chunks[group[0]]
                if first.kind == "paragraph":
                    try:
                        answers[group[0]] = await client.paragraph(http, latin(first)[0])
                    except Exception as exc:
                        print(f"chunk {first.chunk_id} (paragraph) failed: "
                              f"{type(exc).__name__}: {exc}", file=sys.stderr)
                    return

                items = [latin(chunks[i]) for i in group]
                try:
                    for index, result in zip(group, await client.clause_batch(http, items)):
                        answers[index] = result
                    return
                except Exception as exc:
                    # A misaligned or failed batch would corrupt every clause in
                    # it, so retry them one at a time rather than guess.
                    print(f"batch of {len(group)} failed ({type(exc).__name__}: {exc}); "
                          "retrying individually", file=sys.stderr)

                for index in group:
                    text, context = latin(chunks[index])
                    try:
                        answers[index] = await client.clause(http, text, context)
                    except Exception as exc:
                        print(f"chunk {chunks[index].chunk_id} failed: "
                              f"{type(exc).__name__}: {exc}", file=sys.stderr)

        await asyncio.gather(*(run(g) for g in _groups(chunks, CLAUSE_BATCH_SIZE)))

    return answers


def _build_tasks(chunks, answers, document_text, document_date) -> list[Task]:
    tasks: list[Task] = []
    for piece, answer in zip(chunks, answers):
        if not answer:
            continue
        original = document_text[piece.start : piece.end]
        sources = [source_for(original)]
        if piece.context:
            sources.append(source_for(piece.context))

        if piece.kind == "paragraph":
            found = answer.get("tasks") or []
        else:
            if not answer.get("is_task"):
                continue
            # Section 7.4: the task text *is* the chunk, so the model never
            # regenerates it — fewer output tokens and nothing to hallucinate.
            found = [{
                "task": original,
                "deadline": answer.get("deadline"),
                "executors": answer.get("executors") or [],
            }]

        for item in found:
            grounded = True
            if piece.kind == "paragraph":
                text_found = ground(item.get("task"), sources, TASK_THRESHOLD)
                if not text_found.ok or not text_found.value:
                    continue  # section 7.5: an ungroundable task is dropped
                # Docling sometimes strips a clause's number, so a real clause
                # arrives as a paragraph and the model returns a fragment of
                # it. Widen to the sentence so the user sees a whole task.
                task_text = text_found.sentence() or text_found.value
            else:
                task_text = original.strip()

            deadline_found = ground(item.get("deadline"), sources)
            if item.get("deadline") and not deadline_found.ok:
                grounded = False
            executors, all_ok = ground_all(item.get("executors") or [], sources)
            if not all_ok:
                grounded = False

            deadline = parse_deadline(deadline_found.value, document_date)
            tasks.append(
                Task(
                    id=len(tasks) + 1,
                    task=" ".join(task_text.split()),
                    deadline_raw=deadline["deadline_raw"],
                    deadline_date=deadline["deadline_date"],
                    deadline_type=deadline["deadline_type"],
                    executors=executors,
                    page=piece.page,
                    source=piece.kind,
                    grounded=grounded,
                )
            )
    return tasks


async def analyze(
    pdf_path: str, model: str, ocr: bool = True, document=None
) -> Analysis:
    """Analyse one PDF.

    ``document`` lets a caller supply an already-extracted document, which the
    evaluation harness uses to avoid re-running OCR over the whole corpus.
    """
    started = time.time()

    if document is None:
        document = extract(pdf_path, ocr=ocr)
    extract_ms = int((time.time() - started) * 1000)

    number, date = _regex_details(document)
    missing = ["organization", "signatory_name", "signatory_position"]
    if not number.value:
        missing.append("document_number")
    if not date.value:
        missing.append("document_date")

    client = Client(model=model)
    llm_started = time.time()
    answer, detail_sources = await _details(client, document, missing)
    chunks = chunk(document)
    answers = await _tasks(client, chunks, document.text)
    llm_ms = int((time.time() - llm_started) * 1000)

    def from_llm(field: str) -> str | None:
        """LLM details are grounded too, so what is shown is what the page says."""
        value = answer.get(field)
        if not value or not detail_sources:
            return None
        found = ground(value, detail_sources)
        return found.value if found.ok else None

    if not number.value and answer.get("document_number"):
        number = Field_(value=from_llm("document_number"), method="llm")
    if not date.value and answer.get("document_date"):
        raw = from_llm("document_date")
        parsed = parse_deadline(raw)
        date = DateField(value=parsed["deadline_date"], raw=raw, method="llm")

    details = Details(
        organization=Field_(value=from_llm("organization"), method="llm"),
        document_number=number,
        document_date=date,
        signatory=Signatory(
            name=from_llm("signatory_name"),
            position=from_llm("signatory_position"),
            method="llm",
        ),
    )

    tasks = _build_tasks(chunks, answers, document.text, date.value)

    return Analysis(
        model=model,
        source=SourceInfo(
            pages=len(document.pages),
            scanned_pages=document.scanned_pages,
            script=detect_script(document.text),
        ),
        details=details,
        tasks=tasks,
        timings_ms=Timings(
            extract=extract_ms,
            llm=llm_ms,
            total=int((time.time() - started) * 1000),
        ),
    )


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("-")]
    if len(args) != 1:
        print("usage: python -m app.pipeline [--no-ocr] [--model M] file.pdf", file=sys.stderr)
        return 2
    model = "qwen3:4b"
    if "--model" in argv:
        model = argv[argv.index("--model") + 1]
    result = asyncio.run(analyze(args[0], model, ocr="--no-ocr" not in argv))
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
