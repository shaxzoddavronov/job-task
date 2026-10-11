"""Text normalization for Uzbek: apostrophes, Cyrillic → Latin, case and punctuation.

Two stages, because entity tagging sits between them and needs the original case:
`normalize_script` (case kept) → entities.EntityExtractor.tag → `clean_text` (lowercased).
"""

import re
import unicodedata

APOSTROPHE_VARIANTS = "‘’ʻʼ`´"
_APOSTROPHE_TABLE = str.maketrans({ch: "'" for ch in APOSTROPHE_VARIANTS})

# Canonical entity tokens produced by entities.py; clean_text leaves them untouched.
ENTITY_TOKEN_RE = re.compile(r"\b(?:LAW|ART|NUM|ORG|TERM)_[a-z0-9_]+\b")

_CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_WORD_RE = re.compile(r"[\w']+")

# Uzbek Cyrillic → Latin (lowercase). "е" is handled separately (ye/e by position).
_CYR_TO_LAT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ё": "yo", "ж": "j", "з": "z",
    "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p",
    "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "x", "ц": "ts", "ч": "ch",
    "ш": "sh", "щ": "sh", "ъ": "'", "ь": "", "ы": "i", "э": "e", "ю": "yu", "я": "ya",
    "ў": "o'", "қ": "q", "ғ": "g'", "ҳ": "h", "і": "i",
}  # fmt: skip
_CYR_VOWELS = set("аеёиоуўэюяы")

# Cyrillic letters that look like Latin ones; used only inside mixed-script words.
_HOMOGLYPHS = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "к": "k",
    "і": "i", "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "Х": "X", "У": "Y", "І": "I",
}  # fmt: skip


def unify_apostrophes(text: str) -> str:
    """Map every apostrophe-like character to ASCII `'`."""
    return text.translate(_APOSTROPHE_TABLE)


def _transliterate_word(word: str) -> str:
    if _LATIN_RE.search(word):
        word = "".join(_HOMOGLYPHS.get(ch, ch) for ch in word)
        if not _CYRILLIC_RE.search(word):
            return word

    letters = [ch for ch in word if ch.isalpha()]
    all_caps = len(letters) > 1 and all(ch.isupper() for ch in letters)
    out: list[str] = []
    prev = ""
    for ch in word:
        low = ch.lower()
        if low == "е":
            # Abbreviations (all caps) keep a plain "e": ЕРИ → ERI.
            starts_syllable = not prev or prev in _CYR_VOWELS or prev in "ъь"
            lat = "ye" if starts_syllable and not all_caps else "e"
        else:
            lat = _CYR_TO_LAT.get(low)
        if lat is None:  # not Cyrillic
            out.append(ch)
        elif ch.isupper():
            out.append(lat.upper() if all_caps else lat[:1].upper() + lat[1:])
        else:
            out.append(lat)
        prev = low if ch.isalpha() else ""
    return "".join(out)


def transliterate(text: str) -> str:
    """Uzbek Cyrillic → Latin, word by word, keeping case. Fixes homoglyphs in mixed words."""
    if not _CYRILLIC_RE.search(text):
        return text
    return _WORD_RE.sub(lambda m: _transliterate_word(m.group(0)), text)


def normalize_script(text: str) -> str:
    """Unicode NFC, unified apostrophes, Latin script, collapsed whitespace. Case is kept."""
    text = unicodedata.normalize("NFC", str(text))
    text = transliterate(unify_apostrophes(text))
    return re.sub(r"\s+", " ", text).strip()


def _clean_plain(text: str, strip_apostrophes: bool) -> str:
    text = text.lower()
    text = re.sub(r"[^\w'\s]|_", " ", text)  # punctuation and hyphens → space
    if strip_apostrophes:
        return text.replace("'", "")
    return re.sub(r"(?<!\w)'|'(?!\w)", " ", text)  # keep only word-internal apostrophes


def clean_text(text: str, strip_apostrophes: bool = False) -> str:
    """Lowercase and strip punctuation, leaving entity tokens intact. Idempotent."""
    parts: list[str] = []
    last = 0
    for m in ENTITY_TOKEN_RE.finditer(text):
        parts += [_clean_plain(text[last : m.start()], strip_apostrophes), f" {m.group(0)} "]
        last = m.end()
    parts.append(_clean_plain(text[last:], strip_apostrophes))
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def normalize(text: str, strip_apostrophes: bool = False) -> str:
    """Full normalization without entity tagging (for fuzzy matching and audits)."""
    return clean_text(normalize_script(text), strip_apostrophes)


def strip_phrases(text: str, phrases: list[str]) -> str:
    """Remove whole-word phrases from cleaned text (longest first) and collapse spaces."""
    for phrase in sorted(phrases, key=len, reverse=True):
        text = re.sub(rf"(?<![\w']){re.escape(phrase)}(?![\w'])", " ", text)
    return re.sub(r"\s+", " ", text).strip()
