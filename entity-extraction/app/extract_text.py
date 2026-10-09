"""Text extraction: Docling + Tesseract OCR -> per-page markdown and tables.

Plan section 4.

One path handles both kinds of input. Docling uses a page's native text layer
where there is one and OCRs the rest, so a PDF that mixes typed and scanned
pages needs no special casing here. PyMuPDF is used only to *label* pages as
scanned or not: that label decides whether to force full-page OCR, and it ends
up in the API response (``source.scanned_pages``).

Debug CLI::

    python -m app.extract_text path/to/file.pdf
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass, field

import pymupdf

#: A page with no more text than this is a candidate for being scanned
#: (plan section 4).
TEXT_LAYER_MIN_CHARS = 50

#: Tesseract language packs, in the order Tesseract should try them.
OCR_LANGS = ["uzb", "uzb_cyrl"]


@dataclass
class Table:
    """One detected table. ``header`` may be empty if Docling found no header row."""

    page: int
    header: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)

    def row_text(self, row: list[str]) -> str:
        """One row as ``header: value | header: value`` (plan section 7.3)."""
        if self.header:
            pairs = [
                f"{h.strip()}: {v.strip()}"
                for h, v in zip(self.header, row)
                if v.strip()
            ]
        else:
            pairs = [v.strip() for v in row if v.strip()]
        return " | ".join(pairs)


@dataclass
class Block:
    """One structural element of the document, with its span in ``Document.text``.

    Chunking works from these rather than from exported markdown: Docling
    renumbers lettered sub-items when it serialises a list (``а) б) в)`` come
    back as ``2. 3. 4.``, colliding with the document's own clause 2), and OCR
    can merge every line of a page into one. Blocks carry the page number and
    the span, so a value found downstream can always be traced back to the
    original text.
    """

    text: str
    kind: str  # "heading" | "text" | "table"
    page: int
    start: int
    end: int
    table: Table | None = None


@dataclass
class Page:
    number: int  # 1-based, as Docling and PyMuPDF both number pages
    text: str  # markdown; tables render as markdown tables
    scanned: bool
    tables: list[Table] = field(default_factory=list)


#: Blocks are joined by this when the document text is assembled, so every
#: block span indexes straight into ``Document.text``.
BLOCK_SEPARATOR = "\n\n"


@dataclass
class Document:
    pages: list[Page]
    blocks: list[Block] = field(default_factory=list)
    text: str = ""

    @property
    def scanned_pages(self) -> int:
        return sum(1 for p in self.pages if p.scanned)

    def page_text(self, number: int) -> str:
        """The text of one page, as a slice of ``self.text``.

        Section 6 searches page 1 only, and needs the offsets to line up with
        the whole-document text the rest of the pipeline uses.
        """
        spans = [b for b in self.blocks if b.page == number]
        return self.text[spans[0].start : spans[-1].end] if spans else ""


def _is_scanned(page) -> bool:
    """True when a page looks like a picture of text rather than text.

    Plan section 4 classifies on text length alone, but that misreads a sparse
    page: a signature page holding only "Vazir A. Karimov" has a perfectly good
    text layer and fewer than 50 characters. Calling it scanned would inflate
    the ``scanned_pages`` figure shown to the user, and — in a short document
    where every page is sparse — would push the whole file through full-page
    OCR instead of reading the text layer that is already there.

    So a page must also *carry an image* to count as scanned. A page with
    neither text nor images is blank; OCR would find nothing on it either way.
    """
    if len(page.get_text().strip()) > TEXT_LAYER_MIN_CHARS:
        return False
    return bool(page.get_images(full=True))


def scanned_flags(pdf_path: str) -> list[bool]:
    """Per page: True when the page has no usable text layer.

    Cheap — this reads the text layer and the image list, it renders nothing.
    """
    with pymupdf.open(pdf_path) as doc:
        return [_is_scanned(page) for page in doc]


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


#: Converters are expensive to build and safe to reuse, so they are kept per
#: (ocr, full-page) combination — at most three ever exist. Rebuilding one per
#: document cost about 1.8 s on a 5-page act, roughly 8x the conversion itself.
_CONVERTERS: dict[tuple[bool, bool], object] = {}


def get_converter(force_full_page_ocr: bool, ocr: bool = True):
    key = (ocr, force_full_page_ocr)
    if key not in _CONVERTERS:
        _CONVERTERS[key] = _build_converter(force_full_page_ocr, ocr)
    return _CONVERTERS[key]


def _build_converter(force_full_page_ocr: bool, ocr: bool = True):
    """A Docling converter, with Tesseract CLI OCR for Uzbek unless disabled.

    Built per call because the OCR mode depends on the document. Docling caches
    the heavy models at module level, so this is not the expensive part.

    Docling constructs its OCR model eagerly, so a missing Tesseract binary
    fails the conversion even for a PDF that is pure text layer. Checking up
    front turns that into an actionable message.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        OcrMode,
        PdfPipelineOptions,
        TesseractCliOcrOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    if ocr and not tesseract_available():
        raise RuntimeError(
            "Tesseract is not on PATH, so scanned pages cannot be read. Install it with:\n"
            "  sudo apt-get install -y tesseract-ocr tesseract-ocr-uzb tesseract-ocr-uzb-cyrl\n"
            "To extract a text-layer PDF without OCR, pass ocr=False "
            "(CLI: --no-ocr)."
        )

    options = PdfPipelineOptions()
    options.do_ocr = ocr
    options.do_table_structure = True
    if ocr:
        options.ocr_options = TesseractCliOcrOptions(
            lang=OCR_LANGS,
            mode=OcrMode.FULL_PAGE if force_full_page_ocr else OcrMode.DEFAULT,
        )
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def _cell_text(cell) -> str:
    return (getattr(cell, "text", "") or "").strip()


def _clean_header(text: str) -> str:
    """Drop a stray leading character from a column heading.

    OCR picks up the table's rule lines as characters, so a heading arrives as
    "B Topshiriq mazmuni" or "Г Masʼul ijrochi". Those headings are rendered
    into the text the model reads and the user sees, so the noise propagates.
    A single non-digit character followed by a real word is never a heading in
    its own right.
    """
    parts = text.split()
    if len(parts) > 1 and len(parts[0]) == 1 and not parts[0].isdigit():
        return " ".join(parts[1:])
    return text


def _tables_by_page(docling_doc) -> dict[int, list[Table]]:
    """Pull tables out of a converted document, grouped by page.

    Docling's table model has changed shape across releases, so this reads
    defensively: a table we cannot parse is skipped rather than crashing the
    extraction — the same rows are still present in the page markdown.
    """
    by_page: dict[int, list[Table]] = {}
    for item in getattr(docling_doc, "tables", []):
        prov = getattr(item, "prov", None)
        page_no = prov[0].page_no if prov else 1

        grid = getattr(getattr(item, "data", None), "grid", None)
        if not grid:
            continue
        rows = [[_cell_text(c) for c in row] for row in grid]
        if not rows:
            continue

        header, data = [_clean_header(c) for c in rows[0]], rows[1:]
        if not any(header):  # no header row detected; treat everything as data
            header, data = [], rows
        by_page.setdefault(page_no, []).append(
            Table(page=page_no, header=header, rows=data)
        )
    return by_page


def _assemble(raw_blocks: list[tuple[str, str, int, Table | None]]) -> tuple[list[Block], str]:
    """Join block texts into one string, recording each block's span."""
    blocks: list[Block] = []
    pieces: list[str] = []
    cursor = 0
    for text, kind, page, table in raw_blocks:
        if not text.strip():
            continue
        if pieces:
            cursor += len(BLOCK_SEPARATOR)
            pieces.append(BLOCK_SEPARATOR)
        blocks.append(
            Block(
                text=text, kind=kind, page=page,
                start=cursor, end=cursor + len(text), table=table,
            )
        )
        pieces.append(text)
        cursor += len(text)
    return blocks, "".join(pieces)


def _docling_blocks(docling_doc, tables_by_page: dict[int, list[Table]]):
    """Walk Docling's structured items in reading order."""
    seen_tables: dict[int, int] = {}
    for item, _level in docling_doc.iterate_items():
        label = str(getattr(item, "label", "") or "")
        prov = getattr(item, "prov", None)
        page = prov[0].page_no if prov else 1

        if label == "table":
            index = seen_tables.get(page, 0)
            seen_tables[page] = index + 1
            candidates = tables_by_page.get(page, [])
            if index < len(candidates):
                table = candidates[index]
                text = "\n".join(
                    table.row_text(row) for row in table.rows if table.row_text(row)
                )
                yield text, "table", page, table
            continue

        text = (getattr(item, "text", "") or "").strip()
        if not text:
            continue
        kind = "heading" if "header" in label or "title" in label else "text"
        yield text, kind, page, None


def extract(pdf_path: str, ocr: bool = True) -> Document:
    """Extract one PDF into per-page markdown plus structured tables.

    ``ocr=False`` is a debugging convenience for text-layer PDFs on a machine
    without Tesseract; it silently loses every scanned page, so the pipeline
    never uses it.
    """
    flags = scanned_flags(pdf_path)
    converter = get_converter(force_full_page_ocr=all(flags), ocr=ocr)
    docling_doc = converter.convert(pdf_path).document

    tables = _tables_by_page(docling_doc)
    page_numbers = sorted(docling_doc.pages) or [1]

    pages = []
    for n in page_numbers:
        pages.append(
            Page(
                number=n,
                text=docling_doc.export_to_markdown(page_no=n),
                # A PDF whose page count Docling and PyMuPDF disagree on is odd
                # enough that defaulting to "has text" is the safer guess.
                scanned=flags[n - 1] if n - 1 < len(flags) else False,
                tables=tables.get(n, []),
            )
        )

    blocks, text = _assemble(list(_docling_blocks(docling_doc, tables)))
    return Document(pages=pages, blocks=blocks, text=text)


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("-")]
    if len(args) != 1:
        print("usage: python -m app.extract_text [--no-ocr] file.pdf", file=sys.stderr)
        return 2

    doc = extract(args[0], ocr="--no-ocr" not in argv)
    print(
        f"# {len(doc.pages)} page(s), {doc.scanned_pages} scanned\n",
        file=sys.stderr,
    )
    for page in doc.pages:
        kind = "scanned" if page.scanned else "text"
        print(f"\n===== page {page.number} ({kind}) =====")
        print(page.text)
        for i, table in enumerate(page.tables, 1):
            print(f"\n--- table {i}: header={table.header} ({len(table.rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
