"""Suggest gazetteer entries from the data, for manual review.

Writes reports/gazetteer_candidates.yaml (git-ignored). Nothing here edits data/gazetteer.yaml;
that file is curated by hand. Run with `make gazetteer`.

Three sources of candidates:
1. Abbreviation pairs written in the text: "Davlat xizmatlari markazi (DXM)" or "STIR (soliq ...)".
   The phrase is trimmed by aligning the abbreviation's letters with word initials.
2. Phrases ending in an organization word (vazirligi, markazi, agentligi, ...).
3. Bare abbreviations (DXM, FHDYO, YaTT, MChJ) with counts.
"""

import re
from collections import Counter, defaultdict

import pandas as pd
import yaml

from app.core.config import get_settings
from app.core.data import FileSource
from app.core.normalize import normalize, normalize_script
from scripts.audit_data import ABBR_RE, LAW_TYPES, ORG_SUFFIXES

DIGRAPHS = ("sh", "ch", "ya", "yo", "yu", "o'", "g'")
FUNCTION_WORDS = set(
    "va yoki orqali uchun bilan hamda esa bo'limi bo'yicha tegishli joyidagi buni so'ng keyin "
    "agar natija bo'lmasa qilib asosan menga ayting iltimos ariza haqida jarayonida chiqarish "
    "tartibni".split()
)
PAIR_BEFORE_RE = re.compile(r"((?:[\w']+[\s-]+){1,8}[\w']+)\s*\(\s*(" + ABBR_RE.pattern + r")\s*\)")
PAIR_AFTER_RE = re.compile(r"(" + ABBR_RE.pattern + r")\s*\(\s*([^()]{4,80}?)\s*\)")
ORG_PHRASE_RE = re.compile(r"((?:[\w']+ ){1,3}(?:" + "|".join(ORG_SUFFIXES) + r"))\b")
MIN_COUNT = 1


def initial_options(word: str) -> set[str]:
    """Initials a word can contribute to an abbreviation: 'shaxsiy' → {'s', 'sh'}."""
    w = word.lower()
    options = {w[:1]}
    options |= {d.replace("'", "") for d in DIGRAPHS if w.startswith(d)}
    return options


def aligns(abbr: str, words: list[str]) -> bool:
    """True if the abbreviation spells the initials of `words` (digraphs allowed)."""
    target = abbr.lower().replace("'", "")

    def walk(i: int, pos: int) -> bool:
        if i == len(words):
            return pos == len(target)
        return any(
            target.startswith(opt, pos) and walk(i + 1, pos + len(opt))
            for opt in initial_options(words[i])
        )

    return walk(0, 0)


def trim_to_abbreviation(abbr: str, phrase: str) -> str | None:
    """Shortest trailing slice of `phrase` whose initials spell `abbr`."""
    words = re.split(r"[\s-]+", phrase.strip())
    for k in range(1, len(words) + 1):
        if aligns(abbr, words[-k:]):
            return " ".join(words[-k:])
    return None


def guess_type(phrase: str) -> str:
    last = phrase.lower().split()[-1] if phrase.strip() else ""
    return "ORG" if any(last.startswith(s[:-1]) for s in ORG_SUFFIXES) else "TERM"


def abbreviation_pairs(texts: list[str]) -> dict[str, Counter]:
    pairs: dict[str, Counter] = defaultdict(Counter)
    for text in texts:
        for m in PAIR_BEFORE_RE.finditer(text):
            phrase = trim_to_abbreviation(m.group(2), m.group(1))
            if phrase:
                pairs[m.group(2)][phrase.lower()] += 1
        for m in PAIR_AFTER_RE.finditer(text):
            inner = m.group(2)
            phrase = trim_to_abbreviation(m.group(1), inner) or f"? {inner}"
            pairs[m.group(1)][phrase.lower()] += 1
    return pairs


def org_phrases(texts: list[str]) -> Counter:
    counts: Counter = Counter()
    for text in texts:
        for m in ORG_PHRASE_RE.finditer(normalize(text)):
            words = m.group(1).split()
            while len(words) > 1 and words[0] in FUNCTION_WORDS:
                words = words[1:]
            for k in range(1, len(words)):  # every trailing slice: "xizmatlari markazi", ...
                counts[" ".join(words[-k - 1 :])] += 1
    return counts


def build(texts: list[str]) -> dict:
    script_texts = [normalize_script(t) for t in texts]
    pairs = abbreviation_pairs(script_texts)
    abbr_counts = Counter(
        m.group(0) for t in script_texts for m in ABBR_RE.finditer(t) if m.group(0) not in LAW_TYPES
    )
    paired = [
        {
            "id": re.sub(r"[^a-z0-9]+", "_", abbr.lower()).strip("_"),
            "type": guess_type(phrases.most_common(1)[0][0]),
            "aliases": [phrases.most_common(1)[0][0].capitalize(), abbr],
            "abbr_count": abbr_counts.get(abbr, 0),
            "phrase_variants": dict(phrases.most_common()),
        }
        for abbr, phrases in sorted(pairs.items(), key=lambda p: -abbr_counts.get(p[0], 0))
    ]
    return {
        "abbreviation_pairs": paired,
        "org_phrases": {p: n for p, n in org_phrases(texts).most_common() if n >= MIN_COUNT},
        "abbreviations": {
            a: {"count": n, "has_expansion": a in pairs} for a, n in abbr_counts.most_common()
        },
    }


def main() -> None:
    settings = get_settings()
    raw = FileSource(settings.answers_path, settings.questions_path).load_raw()
    texts = pd.concat([raw.answers["answer"], raw.questions["question"]]).dropna().astype(str)
    result = build(texts.tolist())
    out = settings.reports_dir / "gazetteer_candidates.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    header = "# Candidates for data/gazetteer.yaml. Review by hand; do not load this file.\n"
    out.write_text(
        header + yaml.safe_dump(result, allow_unicode=True, sort_keys=False, width=100),
        encoding="utf-8",
    )
    print(
        f"{len(result['abbreviation_pairs'])} abbreviation pairs, "
        f"{len(result['org_phrases'])} org phrases, "
        f"{len(result['abbreviations'])} abbreviations → {out}"
    )


if __name__ == "__main__":
    main()
