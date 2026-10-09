"""Approximate Latin -> Cyrillic transliteration, for generating test documents.

The inverse of ``app.script_utils``, and deliberately only as good as it needs
to be: plan section 11 (Phase 2) says "approximate is fine as long as labels
use the same text", which holds because the generator transliterates the
document and its labels with this same function.

This lives under ``eval/`` because it is a test-data tool. Nothing in ``app/``
imports it.
"""

# Longest first: digraphs must win over their component letters.
PAIRS = [
    ("oʻ", "ў"), ("gʻ", "ғ"), ("o'", "ў"), ("g'", "ғ"),
    ("sh", "ш"), ("ch", "ч"), ("yo", "ё"), ("yu", "ю"), ("ya", "я"),
    ("ye", "е"), ("ts", "ц"),
    ("a", "а"), ("b", "б"), ("v", "в"), ("g", "г"), ("d", "д"),
    ("e", _E := "e"), ("j", "ж"), ("z", "з"), ("i", "и"), ("y", "й"),
    ("k", "к"), ("l", "л"), ("m", "м"), ("n", "н"), ("o", "о"),
    ("p", "п"), ("r", "р"), ("s", "с"), ("t", "т"), ("u", "у"),
    ("f", "ф"), ("x", "х"), ("h", "ҳ"), ("q", "қ"),
    ("ʼ", "ъ"), ("c", "к"), ("w", "в"),
]


def _match_case(src: str, dst: str) -> str:
    if src.isupper() and len(src) > 1:
        return dst.upper()
    if src[0].isupper():
        return dst[0].upper() + dst[1:]
    return dst


def to_cyrillic(text: str) -> str:
    """Transliterate Uzbek Latin text to Cyrillic, preserving case and digits.

    A bare ``e`` is context-sensitive: Uzbek Cyrillic writes ``э`` at the start
    of a word (``eʼlon`` -> ``эълон``) and ``е`` inside one (``Oʻzbekiston`` ->
    ``Ўзбекистон``, not ``Ўзбэкистон``).
    """
    out = []
    i = 0
    while i < len(text):
        for latin, cyrillic in PAIRS:
            chunk = text[i : i + len(latin)]
            if chunk.lower() == latin:
                if cyrillic == _E:
                    at_word_start = not (out and out[-1].isalpha())
                    cyrillic = "э" if at_word_start else "е"
                out.append(_match_case(chunk, cyrillic))
                i += len(latin)
                break
        else:
            out.append(text[i])
            i += 1
    return "".join(out)
