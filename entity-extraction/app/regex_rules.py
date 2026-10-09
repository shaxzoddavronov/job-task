"""Document number and date extraction from page 1.

Plan section 6.

Both run on the *normalized Latin* text (see ``app.script_utils``) and take the
earliest match on page 1, which is where Uzbek official documents put their
registration line. Anything these two find is never asked of the LLM; anything
they miss, or that fails validation, is.

Each function returns ``None`` or a dict::

    {"value": "2026-03-05", "raw": "11.09.2026", "method": "regex"}

``raw`` is the substring in the document's *original* script, which is what the
API shows the user, so callers pass the original text and its offset map.
"""

from __future__ import annotations

import datetime as dt
import re

#: Plausible range for a document date. Outside it, the match is almost
#: certainly something else — a sum of money, an article reference, a year in
#: running text.
MIN_YEAR, MAX_YEAR = 2000, 2035

#: Both the modern Latin spellings (sentabr) and the ones that come out of
#: transliterating Cyrillic (сентябр -> sentyabr).
MONTHS = (
    r"(yanvar|fevral|mart|aprel|may|iyun|iyul|avgust|"
    r"sent(?:y)?abr|okt(?:y)?abr|noyabr|dekabr)"
)

MONTH_NUMBERS = {
    "yanvar": 1, "fevral": 2, "mart": 3, "aprel": 4, "may": 5, "iyun": 6,
    "iyul": 7, "avgust": 8, "sentabr": 9, "sentyabr": 9, "oktabr": 10,
    "oktyabr": 10, "noyabr": 11, "dekabr": 12,
}

#: The tail shared by every number pattern: ``160``, ``15/2-784``, ``PF-6079``.
_NUMBER = r"([A-Za-z]{0,4}-?\d+(?:[/\-]\d+)*)"

#: Sign-prefixed: ``№ 160``, ``№ 15/2-784``, ``№160``, ``No 45``, ``N° 7``.
#:
#: The ``Ne`` and ``M`` branches are OCR spellings of ``№``, all verified on
#: the test set: Tesseract renders the sign as ``Ne`` on scanned Latin pages
#: and as ``М``, ``М:`` or ``Ме`` on scanned Cyrillic ones. Without them the
#: number is lost on nearly every scan. Each demands a digit-bearing tail and
#: refuses a preceding word character or full stop, so ``M. Karimov``,
#: ``Mart oyida`` and ``Nega 5`` do not match.
NUM_RE = re.compile(
    r"(?<!\w)(?:№|N[oº°e]\.?|(?<![\w.])[MМ][eе]?[:.]?(?=\s*[A-Za-z]{0,4}-?\d))"
    r"\s*" + _NUMBER
)

#: Suffix form, with no sign at all: ``PQ-358-son``, ``PF-213-son``,
#: ``16/18-son``. This is how Uzbek legal acts actually print their number —
#: lex.uz documents carry no ``№`` whatsoever, so section 6.1's sign-prefixed
#: pattern alone scores 0% on them.
NUM_SUFFIX_RE = re.compile(r"(?<!\w)" + _NUMBER + r"-son(?:li)?\b", re.I)

#: Departmental acts that are merely registered carry a registry number:
#: ``roʻyxat raqami 3948``.
NUM_REGISTRY_RE = re.compile(r"ro[ʻʼ\'`]?yxat\s+raqami\s+(\d+)", re.I)

#: 11.09.2026
DATE_DOTS = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b")
#: 2026-yil 5-mart, 2026 yil 5 martdagi  (no \b after the month: suffixes count)
DATE_YMD = re.compile(
    r"\b(\d{4})\s*-?\s*yil\s+[«\"']?(\d{1,2})[»\"']?\s*-?\s*" + MONTHS, re.I
)
#: «5» mart 2026 yil, 5-mart 2026
DATE_DMY = re.compile(
    r"[«\"']?\b(\d{1,2})[»\"']?\s*-?\s*" + MONTHS + r"\w*\s+(\d{4})", re.I
)


def _original(latin: str, start: int, end: int, original, offsets) -> str:
    """The matched span as it reads in the document's own script."""
    if original is None or offsets is None:
        return latin[start:end]
    from app.script_utils import map_span

    a, b = map_span(start, end, offsets)
    return original[a:b]


def find_document_number(latin: str, original=None, offsets=None) -> dict | None:
    """Earliest document number in any of the three forms, or ``None``.

    All patterns run and the leftmost match wins, as with dates, so a document
    that prints ``№ 160`` in its letterhead is not overridden by a ``-son``
    reference further down.
    """
    candidates = []
    for pattern in (NUM_RE, NUM_SUFFIX_RE, NUM_REGISTRY_RE):
        for match in pattern.finditer(latin):
            value = match.group(1)
            if any(c.isdigit() for c in value):  # section 6.1 validation
                candidates.append((match.start(), match.start(1), match.end(1), value))

    best = min(candidates, default=None)
    if best is None:
        return None
    _, start, end, value = best
    return {
        "value": value,
        "raw": _original(latin, start, end, original, offsets),
        "method": "regex",
    }


def _valid(year: int, month: int, day: int) -> dt.date | None:
    if not MIN_YEAR <= year <= MAX_YEAR:
        return None
    try:
        return dt.date(year, month, day)
    except ValueError:
        return None


def _candidates(latin: str):
    """Every well-formed date in the text, as ``(start, end, date)``."""
    for match in DATE_DOTS.finditer(latin):
        day, month, year = (int(g) for g in match.groups())
        date = _valid(year, month, day)
        if date:
            yield match.start(), match.end(), date

    for match in DATE_YMD.finditer(latin):
        year, day, month_name = match.groups()
        month = MONTH_NUMBERS.get(month_name.lower())
        date = _valid(int(year), month, int(day)) if month else None
        if date:
            yield match.start(), match.end(), date

    for match in DATE_DMY.finditer(latin):
        day, month_name, year = match.groups()
        month = MONTH_NUMBERS.get(month_name.lower())
        date = _valid(int(year), month, int(day)) if month else None
        if date:
            yield match.start(), match.end(), date


def find_document_date(latin: str, original=None, offsets=None) -> dict | None:
    """Earliest valid document date in any of the three formats, or ``None``.

    All patterns run and the leftmost match wins, so a document that writes
    ``2026-yil 5-mart`` is not beaten by a dotted date further down the page.
    """
    best = min(_candidates(latin), key=lambda c: c[0], default=None)
    if best is None:
        return None
    start, end, date = best
    return {
        "value": date.isoformat(),
        "raw": _original(latin, start, end, original, offsets),
        "method": "regex",
    }
