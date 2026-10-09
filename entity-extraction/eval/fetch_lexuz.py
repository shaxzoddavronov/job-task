"""Fetch public documents from lex.uz and render them as test PDFs.

Plan section 11, Phase 2. These stand in for the employer's samples, which are
not available: they are real Uzbek legal acts, with the clause structure,
phrasing and deadline wording that synthetic documents only approximate.

lex.uz serves the act text as HTML, not PDF, so the layout here is ours — what
is real is the *language*. Each document is rendered Latin and Cyrillic, text
and scanned, exactly like the synthetic set.

The Cyrillic toggle on lex.uz is client-side, so Cyrillic variants are produced
with ``eval/translit_lat2cyr.py``.

Usage::

    python eval/fetch_lexuz.py --list          # show the pinned document ids
    python eval/fetch_lexuz.py                 # fetch, render, write label stubs
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import re
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_synthetic import cyrillicize_text_nodes, scan_pdf, write_pdf
from translit_lat2cyr import to_cyrillic

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "eval" / "data" / "lexuz"
LABEL_DIR = ROOT / "eval" / "labels"

BASE = "https://lex.uz/uz/docs/"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"

#: Pinned so the corpus is reproducible. Harvested from the lex.uz front page
#: on 2026-10-09; re-run with --discover to refresh.
DOC_IDS = [
    "-8519155", "8517274", "8517342", "-8517348", "-8517651",
    "-8519451", "-8523611", "-8527137", "8518014", "8524386",
]

MONTHS_GENITIVE = {
    1: "yanvar", 2: "fevral", 3: "mart", 4: "aprel", 5: "may", 6: "iyun",
    7: "iyul", 8: "avgust", 9: "sentabr", 10: "oktabr", 11: "noyabr", 12: "dekabr",
}


@dataclass
class LexDoc:
    doc_id: str
    kind_line: str  # "... Prezidentining Farmoni, 03.10.2026 yildagi PF-213-son"
    title: str
    paragraphs: list[str] = field(default_factory=list)
    signatory_position: str = ""
    signatory_name: str = ""
    document_number: str = ""
    document_date: str = ""  # ISO

    @property
    def organization(self) -> str:
        """The issuing body, taken from the act-type line."""
        head = self.kind_line.split(",")[0]
        for suffix in ("ning Farmoni", "ning Qarori", "ning qarori", " Farmoni", " Qarori"):
            if head.endswith(suffix):
                return head[: -len(suffix)]
        return head


# --- fetching and parsing --------------------------------------------------


def fetch(doc_id: str, pause: float = 1.0) -> str:
    request = urllib.request.Request(BASE + doc_id, headers={"User-Agent": UA})
    with urllib.request.urlopen(request, timeout=30) as response:
        html = response.read().decode("utf-8", errors="replace")
    time.sleep(pause)  # be polite to a public service
    return html


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


#: Per-paragraph toolbars that lex.uz injects inside every act element.
TOOLBAR_TEXT = ("Hujjatga taklif yuborish", "Audioni tinglash", "Hujjat elementidan havola olish")


def parse(doc_id: str, html: str) -> LexDoc:
    import lxml.html

    tree = lxml.html.fromstring(html)
    for junk in tree.cssselect("script, style, .lx_elem2, .lx_elem3"):
        junk.getparent().remove(junk)

    header = tree.cssselect("div.docHeader__item")
    kind_line = _clean(header[0].text_content()) if header else ""

    # ACT_TEXT holds the act body; BY_DEFAULT holds annex and appendix text,
    # which carries tasks too.
    paragraphs = []
    for block in tree.cssselect("div.ACT_TEXT, div.BY_DEFAULT"):
        text = _clean(block.text_content())
        for toolbar in TOOLBAR_TEXT:
            text = text.replace(toolbar, " ")
        text = _clean(text)
        if text:
            paragraphs.append(text)

    title = _clean(tree.cssselect("title")[0].text_content()) if tree.cssselect("title") else ""
    title = re.sub(r"^\s*\S+-сон\s+[\d.]+\.\s*", "", title)

    doc = LexDoc(doc_id=doc_id, kind_line=kind_line, title=title, paragraphs=paragraphs)

    # Decrees and resolutions number as "PF-213-son"; acts that are merely
    # registered carry a registry number instead ("ro'yxat raqami 3948"), and
    # the page title repeats it as "3948-сон".
    for pattern, source in (
        (r"([A-Z]{1,4}-\d+)-son", kind_line),
        (r"ro[ʻ'‘]yxat raqami\s+(\d+)", kind_line),
        (r"(\d{3,5})-сон", title),
    ):
        found = re.search(pattern, source)
        if found:
            doc.document_number = found.group(1)
            break
    date = re.search(r"(\d{2})\.(\d{2})\.(\d{4})", kind_line)
    if date:
        d, m, y = (int(g) for g in date.groups())
        doc.document_date = dt.date(y, m, d).isoformat()

    # lex.uz marks the signature block with its own class. Earlier this was
    # guessed from the body text, which matched the *control* clause
    # ("... Bosh vaziri A.N. Aripov zimmasiga yuklansin") and then deleted that
    # clause from the body — losing a real task.
    signature = tree.cssselect("div.SIGNATURE")
    if signature:
        text = _clean(signature[0].text_content())
        for toolbar in TOOLBAR_TEXT:
            text = text.replace(toolbar, " ")
        text = _clean(text)
        match = re.match(r"^(.*?)\s+((?:[A-ZА-ЯЁ][a-zа-яё]?\.\s*)+[A-ZА-ЯЁʻʼ]{2,})$", text)
        if match:
            doc.signatory_position = _clean(match.group(1))
            doc.signatory_name = _clean(match.group(2))
        else:
            doc.signatory_position = text
    return doc


# --- rendering -------------------------------------------------------------

CSS = """
@page { size: A4; margin: 2cm 2cm 2cm 3cm; }
body { font-family: 'DejaVu Sans', sans-serif; font-size: 11.5pt; line-height: 1.55; }
.kind { text-align: center; font-weight: bold; text-transform: uppercase; font-size: 11pt; }
.title { text-align: center; font-weight: bold; margin: 1.4em 0; }
p { margin: 0.45em 0; text-align: justify; }
.sign { margin-top: 2.5em; display: flex; justify-content: space-between; font-weight: bold; }
"""


def render_html(doc: LexDoc) -> str:
    body = "".join(f"<p>{p}</p>" for p in doc.paragraphs)
    sign = (
        f'<div class="sign"><span>{doc.signatory_position}</span>'
        f"<span>{doc.signatory_name}</span></div>"
        if doc.signatory_name
        else ""
    )
    return (
        f'<meta charset="utf-8"><style>{CSS}</style>'
        f'<div class="kind">{doc.kind_line}</div>'
        f'<div class="title">{doc.title}</div>{body}{sign}'
    )


# --- label stubs -----------------------------------------------------------


def label_stub(doc: LexDoc, pdf: Path, script: str, scanned: bool) -> dict:
    """Details are filled from lex.uz metadata; tasks must be labelled by hand."""
    convert = to_cyrillic if script == "cyrillic" else (lambda s: s)
    return {
        "pdf": str(pdf.relative_to(ROOT)),
        "script": script,
        "scanned": scanned,
        "source": "lexuz",
        "lexuz_id": doc.doc_id,
        "details": {
            "organization": convert(doc.organization),
            "document_number": doc.document_number,
            "document_date": doc.document_date,
            "signatory_name": convert(doc.signatory_name),
            "signatory_position": convert(doc.signatory_position),
        },
        "tasks": [],
        "_needs_task_labels": True,
    }


# --- driver ----------------------------------------------------------------


def build(doc_ids: list[str], seed: int = 42) -> int:
    rng = random.Random(seed)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    made = 0
    for doc_id in doc_ids:
        try:
            doc = parse(doc_id, fetch(doc_id))
        except Exception as exc:  # a public site; one bad id must not stop the run
            print(f"  {doc_id}: FAILED ({exc})")
            continue
        if len(doc.paragraphs) < 3:
            print(f"  {doc_id}: skipped (only {len(doc.paragraphs)} paragraphs)")
            continue

        stem = f"lexuz_{doc_id.lstrip('-')}"
        for script in ("latin", "cyrillic"):
            html = render_html(doc)
            if script == "cyrillic":
                html = cyrillicize_text_nodes(html)
            tag = "lat" if script == "latin" else "cyr"

            text_pdf = PDF_DIR / f"{stem}_{tag}.pdf"
            write_pdf(html, text_pdf)
            scan = PDF_DIR / f"{stem}_{tag}_scan.pdf"
            scan_pdf(text_pdf, scan, seed=rng.randrange(2**32))

            for pdf, scanned in ((text_pdf, False), (scan, True)):
                stub = label_stub(doc, pdf, script, scanned)
                (LABEL_DIR / f"{pdf.stem}.json").write_text(
                    json.dumps(stub, ensure_ascii=False, indent=2), encoding="utf-8"
                )
        made += 1
        print(f"  {doc_id}: {doc.document_number or '?'} {doc.document_date or '?'} "
              f"({len(doc.paragraphs)} paragraphs) -> {stem}")
    return made


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="show pinned ids and exit")
    parser.add_argument("--ids", nargs="*", help="override the pinned ids")
    args = parser.parse_args()

    if args.list:
        print("\n".join(DOC_IDS))
        return 0

    made = build(args.ids or DOC_IDS)
    print(f"\n{made} documents -> {PDF_DIR}")
    print("Task labels are empty stubs; fill them in by hand.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
