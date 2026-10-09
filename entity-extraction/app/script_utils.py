"""Script detection, normalization and Cyrillic->Latin transliteration.

Plan section 5.

The whole pipeline (regex, chunking, LLM, grounding) runs on *normalized Latin*
text, but everything shown to the user must be in the document's original
script. That round trip is what the offset map is for: every character of the
Latin text records which character of the original it came from, so any span
found downstream can be mapped back.

Typical use::

    latin, offsets = to_latin_with_offsets(raw_text)
    ...                                    # find a span [s, e) in `latin`
    a, b = map_span(s, e, offsets)         # the same span in `raw_text`
    shown = raw_text[a:b]

Both transformations preserve offsets by construction:

* ``normalize`` only ever replaces one character with exactly one character.
* ``translit_with_offsets`` may emit zero (``ь``) or several (``ш`` -> ``sh``)
  characters per input character, and records the source index for each one.
"""

import unicodedata

# --- normalization ---------------------------------------------------------

#: Apostrophe-like characters that stand in for the two Uzbek modifier letters.
APOSTROPHES = "'`‘’ʻʼ"

OKINA = "ʻ"  # U+02BB MODIFIER LETTER TURNED COMMA — the one in oʻ / gʻ
TURNED_COMMA = "ʼ"  # U+02BC MODIFIER LETTER APOSTROPHE — the glottal stop

#: Whitespace that should read as a plain space. Line breaks are left alone:
#: the chunker splits on line starts.
SPACES = "\t     "


def normalize(text: str) -> str:
    """Apply 1-to-1 character replacements. ``len(normalize(t)) == len(t)``.

    Apostrophe variants become ``ʻ`` after ``o``/``g`` (either case) and ``ʼ``
    everywhere else; non-breaking spaces and tabs become spaces. Whitespace is
    deliberately *not* collapsed — that would move every offset. Regexes use
    ``\\s+`` instead.
    """
    out = []
    for ch in text:
        if ch in APOSTROPHES:
            prev = out[-1].lower() if out else ""
            out.append(OKINA if prev in ("o", "g") else TURNED_COMMA)
        elif ch in SPACES:
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


# --- transliteration -------------------------------------------------------

#: Cyrillic -> Latin, lowercase. Plan section 5.3. ``е`` is handled separately
#: (it is context-dependent) and ``ь`` maps to nothing.
TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d",
    "ё": "yo", "ж": "j", "з": "z", "и": "i", "й": "y",
    "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "x", "ц": "ts", "ч": "ch", "ш": "sh",
    "ъ": TURNED_COMMA, "ь": "", "э": "e", "ю": "yu", "я": "ya",
    "ў": "o" + OKINA, "қ": "q", "ҳ": "h", "ғ": "g" + OKINA,
    # Russian letters that appear in Uzbek Cyrillic text (names, borrowings).
    "щ": "sh", "ы": "i",
}

#: ``е`` reads ``ye`` at the start of a word, after a vowel, and after ъ/ь.
VOWELS = set("аеёиоуўэюяыaeiou")

_MISSING = object()


def _reads_as_ye(text: str, i: int) -> bool:
    if i == 0:
        return True
    prev = text[i - 1].lower()
    if prev in ("ъ", "ь"):
        return True
    if prev in VOWELS:
        return True
    return not prev.isalpha()  # word start: preceded by space or punctuation


def _all_caps_here(text: str, i: int) -> bool:
    """True when an uppercase multi-letter output should be fully uppercased.

    ``Шаҳар`` -> ``Shahar`` but ``ШАҲАР`` -> ``SHAHAR``. The next cased letter
    decides; at the end of a word the previous one does.
    """
    for j in range(i + 1, len(text)):
        if text[j].isalpha():
            return text[j].isupper()
    for j in range(i - 1, -1, -1):
        if text[j].isalpha():
            return text[j].isupper()
    return False


def translit_with_offsets(text: str) -> tuple[str, list[int]]:
    """Transliterate Cyrillic to Latin, returning the text and its offset map.

    ``offsets[i]`` is the index in ``text`` of the character that produced
    ``latin[i]``. Characters outside the table — digits, ``№``, punctuation,
    Latin letters — pass through unchanged.
    """
    out: list[str] = []
    offsets: list[int] = []

    for i, ch in enumerate(text):
        lower = ch.lower()
        if lower == "е":
            mapped = "ye" if _reads_as_ye(text, i) else "e"
        else:
            mapped = TRANSLIT.get(lower, _MISSING)
            if mapped is _MISSING:
                out.append(ch)
                offsets.append(i)
                continue

        if mapped and ch.isupper():
            mapped = mapped.upper() if _all_caps_here(text, i) else mapped.capitalize()

        for c in mapped:
            out.append(c)
            offsets.append(i)

    return "".join(out), offsets


def to_latin_with_offsets(text: str) -> tuple[str, list[int]]:
    """``normalize`` then transliterate.

    Normalization is 1-to-1, so the offsets returned by the transliteration
    step index the original text directly — the two compose for free.
    """
    return translit_with_offsets(normalize(text))


def map_span(start: int, end: int, offsets: list[int]) -> tuple[int, int]:
    """Map a half-open span ``[start, end)`` in the Latin text back to the original.

    Raises ``IndexError`` if the span lies outside the offset map, which would
    mean the caller mixed up two different texts.
    """
    if end <= start:
        raise ValueError(f"empty or reversed span: [{start}, {end})")
    return offsets[start], offsets[end - 1] + 1


# --- script detection ------------------------------------------------------


def _script_of(ch: str) -> str | None:
    if not ch.isalpha():
        return None
    name = unicodedata.name(ch, "")
    if name.startswith("CYRILLIC"):
        return "cyrillic"
    if name.startswith("LATIN"):
        return "latin"
    return None


def detect_script(text: str) -> str:
    """``"cyrillic"`` when more than half the letters are Cyrillic, else ``"latin"``.

    Text with no letters at all counts as Latin.
    """
    cyrillic = latin = 0
    for ch in text:
        script = _script_of(ch)
        if script == "cyrillic":
            cyrillic += 1
        elif script == "latin":
            latin += 1
    total = cyrillic + latin
    return "cyrillic" if total and cyrillic > total / 2 else "latin"
