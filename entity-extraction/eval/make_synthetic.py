"""Generate synthetic Uzbek PDFs and their labels (plan section 11, Phase 2).

Each base document is produced in four variants — Latin/Cyrillic × text/scanned
— so every path through the pipeline has labelled data behind it. Labels are
transliterated alongside the document, so a Cyrillic variant's labels hold
Cyrillic strings.

Usage::

    python eval/make_synthetic.py            # 10 documents, 40 PDFs
    python eval/make_synthetic.py --count 4 --seed 7
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import random
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vocab
from translit_lat2cyr import to_cyrillic

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "eval" / "data" / "synthetic"
LABEL_DIR = ROOT / "eval" / "labels"

TEMPLATES = ("letter", "order", "plan", "paragraphs")


# --- the document specification -------------------------------------------


@dataclass
class Task:
    task: str
    deadline_raw: str
    deadline_date: str | None
    executors: list[str]


#: Sub-item markers, per script. Uzbek Latin documents skip "c"; the Cyrillic
#: convention is а) б) в) г). Transliterating the Latin run would produce
#: "к" for "c", which is why this is chosen per script rather than converted.
SUBITEM_LETTERS = {
    "latin": "abvgd",
    "cyrillic": "абвгд",
}

#: Real acts on lex.uz write sub-items both ways, so the set covers both.
SUBITEM_STYLES = ("{}) ", "({}) ")


@dataclass
class Spec:
    template: str
    script: str
    subitem_style: str
    organization: str
    document_number: str
    document_date: str  # ISO, for the label
    document_date_text: str  # as printed in the document
    signatory_name: str
    signatory_position: str
    title: str
    parent_intro: str
    tasks: list[Task] = field(default_factory=list)


# --- randomised pieces -----------------------------------------------------


def _document_number(rng: random.Random) -> str:
    """Covers every shape section 6.1 must match."""
    style = rng.choice(["plain", "slashed", "prefixed", "no"])
    if style == "plain":
        return f"№ {rng.randint(10, 999)}"
    if style == "slashed":
        return f"№ {rng.randint(1, 30)}/{rng.randint(1, 9)}-{rng.randint(100, 999)}"
    if style == "prefixed":
        return f"№ {rng.choice(['PF', 'PQ', 'F'])}-{rng.randint(1000, 9999)}"
    return f"No {rng.randint(10, 999)}"


def _date_text(rng: random.Random, date: dt.date) -> str:
    """One of the three formats section 6.2 parses."""
    month = vocab.MONTHS[date.month - 1]
    style = rng.choice(["dots", "ymd", "dmy"])
    if style == "dots":
        return f"{date.day:02d}.{date.month:02d}.{date.year}"
    if style == "ymd":
        return f"{date.year}-yil {date.day}-{month}"
    return f"«{date.day}» {month} {date.year} yil"


def _add_months(date: dt.date, months: int) -> dt.date:
    month = date.month - 1 + months
    year = date.year + month // 12
    month = month % 12 + 1
    day = min(date.day, [31, 29 if year % 4 == 0 else 28, 31, 30, 31, 30,
                         31, 31, 30, 31, 30, 31][month - 1])
    return dt.date(year, month, day)


def _deadline(rng: random.Random, doc_date: dt.date) -> tuple[str, str | None]:
    """Return ``(raw_text, iso_date_or_None)`` covering section 7.6's categories.

    Also covers the *relative* form ("uch oy muddatda", "bir hafta muddatda")
    that dominates real acts on lex.uz but that section 7.6 does not mention.
    """
    kind = rng.choices(
        ["date", "date", "relative", "relative", "recurring", "ongoing"], k=1
    )[0]
    if kind == "relative":
        count, word = rng.choice(
            [(1, "hafta"), (2, "hafta"), (1, "oy"), (2, "oy"), (3, "oy"), (6, "oy")]
        )
        spelled = {1: "bir", 2: "ikki", 3: "uch", 6: "olti"}[count]
        target = (
            doc_date + dt.timedelta(weeks=count)
            if word == "hafta"
            else _add_months(doc_date, count)
        )
        return f"{spelled} {word} muddatda", target.isoformat()
    if kind == "recurring":
        return rng.choice(["har oyda", "har chorakda", "har yili"]), None
    if kind == "ongoing":
        return rng.choice(["doimiy", "doimiy ravishda"]), None

    target = doc_date + dt.timedelta(days=rng.randint(20, 300))
    month = vocab.MONTHS[target.month - 1]
    style = rng.choice(["gacha", "yil_gacha", "qadar", "dots"])
    if style == "gacha":
        raw = f"{target.day}-{month}gacha"
    elif style == "yil_gacha":
        raw = f"{target.year}-yil {target.day}-{month}gacha"
    elif style == "qadar":
        raw = f"{target.day} {month}ga qadar"
    else:
        raw = f"{target.day:02d}.{target.month:02d}.{target.year} gacha"
    return raw, target.isoformat()


def make_spec(rng: random.Random, template: str) -> Spec:
    doc_date = dt.date(2026, 1, 1) + dt.timedelta(days=rng.randint(0, 300))
    n_tasks = rng.randint(2, 4)

    tasks = []
    for _ in range(n_tasks):
        raw, iso = _deadline(rng, doc_date)
        tasks.append(
            Task(
                task=rng.choice(vocab.ACTIONS),
                deadline_raw=raw,
                deadline_date=iso,
                executors=rng.sample(vocab.EXECUTORS, k=rng.randint(1, 2)),
            )
        )

    # An "order" names its executor once, in the parent clause, and the
    # sub-items inherit it — the case section 7.3 passes as `context`.
    if template == "order":
        shared = rng.sample(vocab.EXECUTORS, k=1)
        for task in tasks:
            task.executors = list(shared)

    spec = Spec(
        template=template,
        script="latin",
        subitem_style=rng.choice(SUBITEM_STYLES),
        organization=rng.choice(vocab.ORGANIZATIONS),
        document_number=_document_number(rng),
        document_date=doc_date.isoformat(),
        document_date_text=_date_text(rng, doc_date),
        signatory_name=rng.choice(vocab.NAMES),
        signatory_position=rng.choice(vocab.POSITIONS),
        title=rng.choice(vocab.TITLES),
        parent_intro=rng.choice(vocab.PARENT_INTROS),
        tasks=tasks,
    )
    if template == "order":
        # Assigning oversight of execution is a task: it names a responsible
        # person, just no deadline. The lex.uz labels treat it that way, and
        # the two label sets have to agree or the model comparison is noise.
        spec.tasks.append(
            Task(
                task=f"Mazkur qarorning ijrosini nazorat qilish "
                f"{spec.signatory_position} zimmasiga yuklatilsin",
                deadline_raw="",
                deadline_date=None,
                executors=[spec.signatory_position],
            )
        )
    return spec


# --- rendering -------------------------------------------------------------

CSS = """
@page { size: A4; margin: 2cm 2cm 2cm 3cm; }
body { font-family: 'DejaVu Sans', sans-serif; font-size: 11.5pt; line-height: 1.5; }
.header { text-align: center; font-weight: bold; text-transform: uppercase; }
.meta { margin-top: 1.2em; }
.title { text-align: center; font-weight: bold; margin: 1.5em 0; }
p { margin: 0.5em 0; text-align: justify; }
table { border-collapse: collapse; width: 100%; margin-top: 1em; font-size: 10.5pt; }
th, td { border: 1px solid #000; padding: 5px 7px; vertical-align: top; }
th { font-weight: bold; }
.sign { margin-top: 3em; }
.sign td { border: none; padding: 0; }
"""


def clauses(spec: Spec) -> list[tuple[str, int | None]]:
    """The document body as plain-text clauses.

    Each entry is ``(text, task_index)``, where ``task_index`` points into
    ``spec.tasks`` or is ``None`` for a clause that assigns no work — a
    preamble, a parent clause, a control clause. Those negatives matter as much
    as the positives: a model that calls everything a task scores well on
    recall and badly in practice.

    This is the single source of truth for clause text: ``_body`` renders these
    same strings into HTML, so labels and document cannot drift apart.
    """
    t = spec.tasks

    if spec.template == "letter":
        out = [(f"{spec.organization} tomonidan quyidagilar belgilansin:", None)]
        out += [
            (
                f"{i}. {x.task}, {x.deadline_raw}. Masʼul ijrochi — "
                f"{', '.join(x.executors)}.",
                i - 1,
            )
            for i, x in enumerate(t, 1)
        ]
        return out

    if spec.template == "order":
        letters = SUBITEM_LETTERS[spec.script]
        sub, control = t[:-1], len(t) - 1  # the last task is the control clause
        out = [(f"1. {', '.join(sub[0].executors)} {spec.parent_intro}:", None)]
        out += [
            (
                f"{spec.subitem_style.format(letters[i])}{x.task}, "
                f"{x.deadline_raw};",
                i,
            )
            for i, x in enumerate(sub)
        ]
        out.append((f"2. {t[control].task}.", control))
        return out

    if spec.template == "plan":
        out = [("Quyidagi chora-tadbirlar rejasi tasdiqlansin:", None)]
        # Table rows are rendered the way section 7.3 renders them for the LLM.
        out += [
            (
                f"№: {i} | Tadbir nomi: {x.task} | Ijro muddati: {x.deadline_raw}"
                f" | Masʼul ijrochi: {', '.join(x.executors)}",
                i - 1,
            )
            for i, x in enumerate(t, 1)
        ]
        return out

    out = [(f"{spec.organization} quyidagilarni maʼlum qiladi.", None)]
    out += [
        (
            f"{', '.join(x.executors)} zimmasiga {x.task} vazifasi yuklatiladi. "
            f"Ushbu vazifa {x.deadline_raw} bajarilishi lozim.",
            i,
        )
        for i, x in enumerate(t)
    ]
    return out


def _body(spec: Spec) -> str:
    items = clauses(spec)
    if spec.template == "plan":
        rows = "".join(
            f"<tr><td>{i}</td><td>{x.task}</td><td>{x.deadline_raw}</td>"
            f"<td>{', '.join(x.executors)}</td></tr>"
            for i, x in enumerate(spec.tasks, 1)
        )
        return (
            f"<p>{items[0][0]}</p>"
            "<table><tr><th>№</th><th>Tadbir nomi</th><th>Ijro muddati</th>"
            f"<th>Masʼul ijrochi</th></tr>{rows}</table>"
        )
    return "".join(f"<p>{text}</p>" for text, _ in items)


def render_html(spec: Spec) -> str:
    return f"""<meta charset="utf-8"><style>{CSS}</style>
<div class="header">{spec.organization}</div>
<div class="meta">{spec.document_date_text}{chr(0x00a0) * 6}{spec.document_number}</div>
<div class="title">{spec.title}</div>
{_body(spec)}
<table class="sign"><tr>
  <td>{spec.signatory_position}</td><td style="text-align:right">{spec.signatory_name}</td>
</tr></table>
"""


_MARKUP = re.compile(r"(<style>.*?</style>|<[^>]*>)", re.S)


def cyrillicize_text_nodes(html: str) -> str:
    """Transliterate the boilerplate text in rendered HTML, leaving markup alone.

    The templates carry fixed Uzbek phrases ("Masʼul ijrochi", "tomonidan
    quyidagilar belgilansin") that are not part of the spec, so a Cyrillic
    document rendered from a converted spec would still be half Latin. This
    converts the text nodes and skips tags and the stylesheet.

    Safe to run over a document rendered from an already-Cyrillic spec:
    ``to_cyrillic`` leaves Cyrillic untouched, so those fields pass through.
    """
    parts = _MARKUP.split(html)
    return "".join(p if p.startswith("<") else to_cyrillic(p) for p in parts)


def write_pdf(html: str, path: Path) -> None:
    from weasyprint import HTML

    HTML(string=html).write_pdf(str(path))


# --- scan simulation -------------------------------------------------------


def scan_pdf(src: Path, dst: Path, seed: int) -> None:
    """Rasterize at 200 dpi, then degrade it the way a desk scanner would.

    Rotation, blur, sensor noise and JPEG artefacts, per plan section 11.

    Takes a seed rather than the caller's generator so that it draws from its
    own stream. Sharing the caller's generator would mean that skipping the
    render (``--labels-only``) consumed a different number of draws and every
    later document came out different from the PDF already on disk.
    """
    rng = random.Random(seed)
    import cv2
    import numpy as np
    import pymupdf
    from PIL import Image

    doc = pymupdf.open(src)
    out = pymupdf.open()
    for page in doc:
        pix = page.get_pixmap(dpi=200)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY if pix.n == 3 else cv2.COLOR_RGBA2GRAY)

        angle = rng.uniform(-1.5, 1.5)
        h, w = img.shape
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(
            img, matrix, (w, h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=255,
        )

        img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.4, 0.9))
        noise = np.random.default_rng(rng.randrange(2**32)).normal(0, 4.0, img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=rng.randint(55, 75))

        new = out.new_page(width=page.rect.width, height=page.rect.height)
        new.insert_image(new.rect, stream=buf.getvalue())

    out.save(dst)
    out.close()
    doc.close()


# --- Cyrillic variant ------------------------------------------------------


def to_cyrillic_spec(spec: Spec) -> Spec:
    """Transliterate the *specification*, not the rendered HTML.

    Converting the HTML would mangle tags and CSS (``<div class="header">``
    becomes ``<див цласс="ҳеадер">``). Converting the spec and re-rendering
    keeps the markup intact, and the labels are built from the same converted
    spec, so document and labels cannot drift apart.
    """
    c = to_cyrillic
    number = spec.document_number
    if number.startswith("No "):  # Cyrillic documents write the sign, not "No"
        number = "№ " + number[3:]
    # The rendered HTML goes through `cyrillicize_text_nodes`, which converts a
    # letter prefix too: the page shows "Ф-1188", so the label must as well.
    # Digits and № pass through `to_cyrillic` unchanged.
    number = c(number)
    return Spec(
        template=spec.template,
        script="cyrillic",
        subitem_style=spec.subitem_style,
        organization=c(spec.organization),
        document_number=number,  # digits and № carry over unchanged
        document_date=spec.document_date,  # ISO, never transliterated
        document_date_text=c(spec.document_date_text),
        signatory_name=c(spec.signatory_name),
        signatory_position=c(spec.signatory_position),
        title=c(spec.title),
        parent_intro=c(spec.parent_intro),
        tasks=[
            Task(
                task=c(t.task),
                deadline_raw=c(t.deadline_raw),
                deadline_date=t.deadline_date,
                executors=[c(e) for e in t.executors],
            )
            for t in spec.tasks
        ],
    )


# --- labels ----------------------------------------------------------------


def label_for(spec: Spec, pdf: Path, script: str, scanned: bool) -> dict:
    """Build the label from an already script-correct spec."""
    return {
        "pdf": str(pdf.relative_to(ROOT)),
        "script": script,
        "scanned": scanned,
        "template": spec.template,
        "details": {
            "organization": spec.organization,
            "document_number": spec.document_number.split(maxsplit=1)[1],
            "document_date": spec.document_date,
            "signatory_name": spec.signatory_name,
            "signatory_position": spec.signatory_position,
        },
        "clauses": [
            {
                # The spec fields are already in the right script, but the
                # template boilerplate ("Masʼul ijrochi") is not — the same
                # reason `cyrillicize_text_nodes` exists for the HTML.
                "text": to_cyrillic(text) if script == "cyrillic" else text,
                "is_task": index is not None,
                "deadline_raw": spec.tasks[index].deadline_raw if index is not None else "",
                "executors": list(spec.tasks[index].executors) if index is not None else [],
            }
            for text, index in clauses(spec)
        ],
        "tasks": [
            {
                "task": t.task,
                "deadline_raw": t.deadline_raw,
                "deadline_date": t.deadline_date,
                "executors": list(t.executors),
            }
            for t in spec.tasks
        ],
    }


# --- driver ----------------------------------------------------------------


def generate(count: int, seed: int, labels_only: bool = False) -> list[Path]:
    rng = random.Random(seed)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    written = []
    for i in range(1, count + 1):
        latin = make_spec(rng, TEMPLATES[(i - 1) % len(TEMPLATES)])

        for script, spec in (
            ("latin", latin),
            ("cyrillic", to_cyrillic_spec(latin)),
        ):
            tag = "lat" if script == "latin" else "cyr"
            html = render_html(spec)
            if script == "cyrillic":
                html = cyrillicize_text_nodes(html)
            text_pdf = PDF_DIR / f"doc_{i:02d}_{tag}.pdf"
            scan = PDF_DIR / f"doc_{i:02d}_{tag}_scan.pdf"
            if not labels_only:
                write_pdf(html, text_pdf)
                scan_pdf(text_pdf, scan, seed=hash((seed, text_pdf.name)) & 0xFFFFFFFF)

            for pdf, scanned in ((text_pdf, False), (scan, True)):
                label = label_for(spec, pdf, script, scanned)
                (LABEL_DIR / f"{pdf.stem}.json").write_text(
                    json.dumps(label, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                written.append(pdf)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10, help="base documents")
    parser.add_argument("--seed", type=int, default=42, help="for reproducibility")
    parser.add_argument(
        "--labels-only",
        action="store_true",
        help="rewrite labels without re-rendering the PDFs (keeps the OCR cache valid)",
    )
    args = parser.parse_args()

    written = generate(args.count, args.seed, args.labels_only)
    print(f"{len(written)} PDFs -> {PDF_DIR}")
    print(f"{len(written)} labels -> {LABEL_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
