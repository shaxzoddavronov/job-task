"""Split the document body into clause / table-row / paragraph chunks.

Plan section 7.3, with one change the test set forced.

Section 7.3 splits exported markdown at line starts. Measurement showed that
does not survive real input:

* Docling renumbers lettered sub-items when it serialises a list — ``а) б) в)``
  come back as ``2. 3. 4.``, colliding with the document's own clause 2.
* Docling sometimes does the opposite and flattens a parent clause and all its
  sub-items into a single block.
* OCR can merge every line of a page into one, so there are no line starts to
  split on.

So chunking works from Docling's structured blocks (which carry the page and a
span) and splits *within* a block on clause markers wherever they fall — at the
start of a line or mid-sentence. Nothing depends on Docling's numbering or on
line breaks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.extract_text import Block, Document

#: Shorter paragraphs are merged with the next one (plan section 7.3).
MIN_PARAGRAPH_CHARS = 200

#: Clause markers, at a block start or inline after whitespace or punctuation:
#: ``1.`` ``2)`` ``1.2.`` ``a)`` ``(a)`` ``а)`` ``(в)`` ``(i)`` ``(vii)``.
#: Real acts on lex.uz use the parenthesised forms and roman numerals, neither
#: of which section 7.3's pattern covers. Numbers are capped at three digits so
#: a year ("2026. yil") is not read as a clause marker.
CLAUSE_MARKER = re.compile(
    r"(?:^|(?<=[\s;:.]))"
    r"(\d{1,3}(?:\.\d{1,3})*[.)]"
    r"|\(?(?:i{1,3}|iv|v|vi{1,3}|ix|x)\)"
    r"|\(?[a-zA-Zа-яёА-ЯЁ]\))"
    r"(?=\s+\S)"
)

#: A top-level marker closes a parent group; a lettered sub-item does not.
TOP_LEVEL = re.compile(r"^\(?\d")


@dataclass
class Chunk:
    chunk_id: int
    kind: str  # "clause" | "table_row" | "paragraph"
    page: int
    start: int  # span into Document.text
    end: int
    text: str
    context: str = ""  # parent clause, where the executor is often named


def _split_clauses(block: Block) -> list[tuple[int, int]]:
    """Spans of the clauses inside one block, or [] if it holds none."""
    starts = [m.start(1) for m in CLAUSE_MARKER.finditer(block.text)]
    if not starts:
        return []
    if starts[0] != 0:
        starts.insert(0, 0)  # text before the first marker is its own piece

    bounds = starts + [len(block.text)]
    spans = []
    for a, b in zip(bounds, bounds[1:]):
        piece = block.text[a:b]
        if piece.strip():
            spans.append((block.start + a, block.start + a + len(piece.rstrip())))
    return spans


def _table_rows(block: Block) -> list[tuple[int, int]]:
    """Spans of each rendered row inside a table block."""
    spans = []
    cursor = 0
    for line in block.text.split("\n"):
        if line.strip():
            spans.append((block.start + cursor, block.start + cursor + len(line)))
        cursor += len(line) + 1
    return spans


#: A trailing block this short *and* carrying a signature signal is the
#: signature line. Length alone is not enough: "2. Nazorat Vazir zimmasiga
#: yuklatilsin." is shorter than this and is a real task.
SIGNATURE_MAX_CHARS = 80

#: A position word, or initials followed by a surname ("A. Karimov").
SIGNATURE_HINT = re.compile(
    r"\b(vazir|hokim|rais|direktor|boshli|prezident|oʻrinbosar|вазир|ҳоким|раис|"
    r"директор|президент|ўринбосар)|[A-ZА-ЯЁ]\.\s*[A-ZА-ЯЁ]",
    re.I,
)

#: The first block this long (or any table) marks where the body starts.
BODY_MIN_CHARS = 150

#: Cap on merging short paragraphs, so a run of them cannot become one chunk.
MAX_PARAGRAPH_CHARS = 600


#: A block ending in one of these is finished; anything else is mid-sentence.
TERMINAL = (".", ";", ":", "!", "?", "»", '"')


def _join_continuations(blocks: list[Block], text: str) -> list[Block]:
    """Join a block to the next when it stops mid-sentence.

    A clause that crosses a page break arrives as several blocks — lex.uz act
    PQ-358 splits one clause into "... ushbu", "qarordan ... Vazirlar" and
    "Mahkamasiga taklif kiritsin. 4. ...". Chunked as they come, the task text
    and its deadline end up in different chunks.
    """
    joined: list[Block] = []
    for block in blocks:
        if (
            joined
            and joined[-1].kind == "text"
            and block.kind == "text"
            and not joined[-1].text.rstrip().endswith(TERMINAL)
        ):
            previous = joined[-1]
            joined[-1] = Block(
                text=text[previous.start : block.end],
                kind="text",
                page=previous.page,
                start=previous.start,
                end=block.end,
            )
        else:
            joined.append(block)
    return joined


def body_blocks(document: Document) -> list[Block]:
    """The blocks between the header region and the signature (section 7.3).

    The header is the letterhead, the date/number line and the title; the
    signature is the name at the end. Both are handled by the details call, and
    chunking them would spend an LLM call per document to be told they assign
    no work.

    The header boundary is the last heading *before the body starts*, not the
    last heading in the document: Docling classifies the signature line of a
    lex.uz act as a heading too, and keying off that dropped the entire body.
    """
    blocks = list(document.blocks)

    body_start = next(
        (i for i, b in enumerate(blocks)
         if b.kind == "table" or (b.kind == "text" and len(b.text) >= BODY_MIN_CHARS)),
        None,
    )
    headings = [
        i for i, b in enumerate(blocks)
        if b.kind == "heading" and (body_start is None or i < body_start)
    ]
    if not headings and body_start is None:
        headings = [i for i, b in enumerate(blocks[:3]) if b.kind == "heading"]

    first = headings[-1] + 1 if headings else 0
    body = [b for b in blocks[first:] if b.kind != "heading"]

    # Drop the signature: a short trailing *text* block that names a position
    # or a person and carries no clause marker. Tables are never signatures.
    while len(body) > 1:
        last = body[-1]
        text = last.text.strip()
        if (
            last.kind == "text"
            and len(text) <= SIGNATURE_MAX_CHARS
            and not CLAUSE_MARKER.match(text)
            and SIGNATURE_HINT.search(text)
        ):
            body.pop()
        else:
            break
    return _join_continuations(body, document.text)


def chunk(document: Document) -> list[Chunk]:
    """Chunk a document body for the per-chunk task call (section 7.4)."""
    chunks: list[Chunk] = []
    context = ""
    pending: Chunk | None = None  # a short paragraph waiting to be merged

    def emit(kind: str, page: int, start: int, end: int, text: str, ctx: str) -> None:
        chunks.append(
            Chunk(
                chunk_id=len(chunks) + 1,
                kind=kind, page=page, start=start, end=end,
                text=text, context=ctx,
            )
        )

    def flush() -> None:
        """Emit the held paragraph, and let it introduce the clauses after it.

        A parent clause whose number Docling stripped arrives as a paragraph
        ("Adliya vazirligi quyidagi vazifalarni bajarsin:"), so setting context
        only from *clause* chunks left every sub-item without the executor
        named in its parent.
        """
        nonlocal pending, context
        if pending:
            emit(pending.kind, pending.page, pending.start, pending.end,
                 pending.text, pending.context)
            stripped = pending.text.strip()
            if stripped.endswith(":"):
                context = stripped
            pending = None

    for block in body_blocks(document):
        if block.kind == "table":
            flush()
            for start, end in _table_rows(block):
                emit("table_row", block.page, start, end,
                     document.text[start:end], context)
            continue

        clause_spans = _split_clauses(block)
        if clause_spans:
            flush()
            for start, end in clause_spans:
                text = document.text[start:end]
                stripped = text.strip()
                # A new top-level clause closes the previous parent group, so
                # the reset happens *before* this chunk is emitted.
                if TOP_LEVEL.match(stripped) and not stripped.endswith(":"):
                    context = ""
                emit("clause", block.page, start, end, text, context)
                if stripped.endswith(":"):
                    context = stripped
            continue

        text = document.text[block.start : block.end]
        if pending and len(pending.text) + len(text) > MAX_PARAGRAPH_CHARS:
            flush()
        if pending:
            pending = Chunk(
                chunk_id=0, kind="paragraph", page=pending.page,
                start=pending.start, end=block.end,
                text=document.text[pending.start : block.end], context=context,
            )
        else:
            pending = Chunk(
                chunk_id=0, kind="paragraph", page=block.page,
                start=block.start, end=block.end, text=text, context=context,
            )
        if len(pending.text) >= MIN_PARAGRAPH_CHARS:
            flush()
        elif pending and len(pending.text) >= MAX_PARAGRAPH_CHARS:
            flush()

    flush()
    return chunks
