"""Deadline string -> ISO date / relative / recurring / ongoing / unknown.

Plan section 7.6, plus a ``relative`` type the plan does not mention.

Reading real acts on lex.uz showed that the commonest deadline form by far is
relative — "bir hafta muddatda", "uch oy muddatda", "within N weeks/months of
this act". Section 7.6 lists only date / recurring / ongoing / unknown, so all
of those would land in ``unknown`` and lose a date that resolves exactly
against the document date.
"""

from __future__ import annotations

import datetime as dt
import re

from app.regex_rules import MONTH_NUMBERS, MONTHS

#: "har oyda", "har chorakda", "har yili", "har hafta"
RECURRING = re.compile(r"\bhar\s+(oy|chorak|yil|hafta|kun)", re.I)

#: "doimiy", "doimiy ravishda"
ONGOING = re.compile(r"\bdoimiy\b", re.I)

#: Uzbek number words that appear in relative deadlines.
NUMBER_WORDS = {
    "bir": 1, "ikki": 2, "uch": 3, "toʻrt": 4, "tort": 4, "besh": 5,
    "olti": 6, "yetti": 7, "sakkiz": 8, "toʻqqiz": 9, "oʻn": 10, "on": 10,
}

#: "uch oy muddatda", "2 hafta muddatda", "bir oy ichida"
RELATIVE = re.compile(
    r"\b(\d+|" + "|".join(NUMBER_WORDS) + r")\s+(kun|hafta|oy|yil)"
    r"\s*(?:muddatda|ichida|davomida)",
    re.I,
)

#: "1-noyabrgacha", "1 noyabrga qadar", "15-dekabrgacha"
DAY_MONTH = re.compile(
    r"\b(\d{1,2})\s*-?\s*" + MONTHS + r"(?:ga|gacha|ga\s+qadar|dan)?", re.I
)

#: A full date, reusing section 6's patterns.
from app.regex_rules import DATE_DMY, DATE_DOTS, DATE_YMD  # noqa: E402


def _add_months(date: dt.date, months: int) -> dt.date:
    month = date.month - 1 + months
    year = date.year + month // 12
    month = month % 12 + 1
    last_day = [31, 29 if year % 4 == 0 and (year % 100 or year % 400 == 0) else 28,
                31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1]
    return dt.date(year, month, min(date.day, last_day))


def _full_date(text: str) -> dt.date | None:
    for match in DATE_DOTS.finditer(text):
        day, month, year = (int(g) for g in match.groups())
        try:
            return dt.date(year, month, day)
        except ValueError:
            continue
    for match in DATE_YMD.finditer(text):
        year, day, name = match.groups()
        month = MONTH_NUMBERS.get(name.lower())
        if month:
            try:
                return dt.date(int(year), month, int(day))
            except ValueError:
                continue
    for match in DATE_DMY.finditer(text):
        day, name, year = match.groups()
        month = MONTH_NUMBERS.get(name.lower())
        if month:
            try:
                return dt.date(int(year), month, int(day))
            except ValueError:
                continue
    return None


def parse_deadline(raw: str | None, document_date: str | None = None) -> dict:
    """Classify a deadline string and resolve it to a date where possible.

    ``document_date`` is the ISO date of the document, needed to resolve both
    relative deadlines and bare day/month ones.

    The input arrives in the document's own script, because that is what the
    user is shown, but every pattern here is Latin. So matching runs on the
    transliterated form while ``deadline_raw`` keeps the original: without
    this, every deadline in a Cyrillic document came back "unknown".
    """
    result = {"deadline_raw": raw, "deadline_date": None, "deadline_type": "unknown"}
    if not raw or not raw.strip():
        result["deadline_type"] = "unknown"
        return result

    from app.script_utils import to_latin_with_offsets

    text = to_latin_with_offsets(raw.strip())[0]
    base = None
    if document_date:
        try:
            base = dt.date.fromisoformat(document_date)
        except ValueError:
            base = None

    if ONGOING.search(text):
        result["deadline_type"] = "ongoing"
        return result

    if RECURRING.search(text):
        result["deadline_type"] = "recurring"
        return result

    full = _full_date(text)
    if full:
        result["deadline_type"] = "date"
        result["deadline_date"] = full.isoformat()
        return result

    relative = RELATIVE.search(text)
    if relative:
        amount, unit = relative.group(1).lower(), relative.group(2).lower()
        count = int(amount) if amount.isdigit() else NUMBER_WORDS.get(amount, 0)
        result["deadline_type"] = "relative"
        if base and count:
            if unit == "kun":
                result["deadline_date"] = (base + dt.timedelta(days=count)).isoformat()
            elif unit == "hafta":
                result["deadline_date"] = (base + dt.timedelta(weeks=count)).isoformat()
            elif unit == "oy":
                result["deadline_date"] = _add_months(base, count).isoformat()
            else:
                result["deadline_date"] = _add_months(base, 12 * count).isoformat()
        return result

    day_month = DAY_MONTH.search(text)
    if day_month:
        day = int(day_month.group(1))
        month = MONTH_NUMBERS.get(day_month.group(2).lower())
        if month:
            result["deadline_type"] = "date"
            year = base.year if base else dt.date.today().year
            try:
                target = dt.date(year, month, day)
            except ValueError:
                return result
            # A day/month earlier than the document means the next year.
            if base and target < base:
                target = dt.date(year + 1, month, day)
            result["deadline_date"] = target.isoformat()
    return result
