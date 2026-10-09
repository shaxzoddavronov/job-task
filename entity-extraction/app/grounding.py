"""Fuzzy grounding of LLM output against the source text.

Plan section 7.5.

The model is told to copy values verbatim, but it still paraphrases, trims a
suffix or merges two executors now and then. Grounding checks that each value
really occurs in the chunk it came from, and — just as importantly — recovers
the span so the value can be shown in the document's *original* script rather
than in the transliterated Latin the pipeline works in.

A value that cannot be found is dropped, never shown. Section 0: missing values
are null, never invented.
"""

from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import fuzz

from app.script_utils import APOSTROPHES, map_span

#: Section 7.5 thresholds.
VALUE_THRESHOLD = 90  # a deadline or an executor
TASK_THRESHOLD = 80  # a paragraph task, which the model writes itself


@dataclass
class Grounded:
    value: str | None  # the matched text, in the document's original script
    score: float
    ok: bool
    start: int = -1  # span in the matched source's original text
    end: int = -1
    source: "Source | None" = None

    def sentence(self) -> str | None:
        """The whole sentence the match sits in.

        For a paragraph task the model writes the text itself and often returns
        only a fragment of the sentence. Widening to the sentence gives the
        user a task they can actually read, still copied from the document.
        """
        if not self.ok or self.source is None:
            return self.value
        text = self.source.original
        start, end = self.start, self.end
        while start > 0 and text[start - 1] not in ".;\n":
            start -= 1
        while end < len(text) and text[end - 1] not in ".;\n":
            end += 1
        return text[start:end].strip() or self.value


def _normalize(text: str) -> str:
    """Lowercase and flatten apostrophes, one character in one character out.

    Length has to be preserved: the match offsets are used to slice the
    original text, so any change in length would shift every span after it.
    """
    out = []
    for ch in text:
        if ch in APOSTROPHES:
            out.append("'")
            continue
        lowered = ch.lower()
        out.append(lowered if len(lowered) == 1 else ch)
    return "".join(out)


def _snap_to_words(text: str, start: int, end: int) -> tuple[int, int]:
    """Grow a span outwards until it sits on word boundaries.

    Fuzzy alignment can land one character inside a word, which surfaces as a
    value like "espublika ijro etuvchi ..." — correct but for a missing first
    letter. Snapping makes that impossible regardless of how the alignment
    falls.
    """
    while start > 0 and text[start - 1].isalnum() and text[start].isalnum():
        start -= 1
    while end < len(text) and text[end - 1].isalnum() and text[end].isalnum():
        end += 1
    return start, end


def find_span(value: str, haystack: str) -> tuple[int, int, float] | None:
    """Best fuzzy match of ``value`` inside ``haystack`` as ``(start, end, score)``."""
    if not value or not value.strip() or not haystack:
        return None
    alignment = fuzz.partial_ratio_alignment(_normalize(value), _normalize(haystack))
    if alignment is None:
        return None
    start, end = _snap_to_words(haystack, alignment.dest_start, alignment.dest_end)
    return start, end, alignment.score


@dataclass
class Source:
    """One place a value may be grounded in: its Latin form, its original-script
    form, and the offset map between them."""

    latin: str
    original: str
    offsets: list[int]


def ground(value: str | None, sources: list[Source], threshold: int = VALUE_THRESHOLD) -> Grounded:
    """Locate ``value`` in any source and return it in the original script.

    The chunk's context counts as a source because an executor is often named
    only in the parent clause — which is why section 7.3 carries it.
    """
    if value is None or not value.strip():
        return Grounded(None, 0.0, False)

    best: tuple[float, int, int, Source] | None = None
    for source in sources:
        found = find_span(value, source.latin)
        if found is None:
            continue
        start, end, score = found
        if best is None or score > best[0]:
            best = (score, start, end, source)

    if best is None or best[0] < threshold:
        return Grounded(None, best[0] if best else 0.0, False)

    score, start, end, source = best
    if end <= start:
        return Grounded(None, score, False)
    try:
        a, b = map_span(start, end, source.offsets)
    except (IndexError, ValueError):
        return Grounded(None, score, False)
    return Grounded(source.original[a:b].strip(), score, True, a, b, source)


def ground_all(
    values: list[str], sources: list[Source], threshold: int = VALUE_THRESHOLD
) -> tuple[list[str], bool]:
    """Ground a list of values, dropping any that cannot be found.

    Returns the survivors and whether every one of them grounded.
    """
    kept, all_ok = [], True
    for value in values:
        found = ground(value, sources, threshold)
        if found.ok and found.value:
            kept.append(found.value)
        else:
            all_ok = False
    return kept, all_ok


def source_for(original: str) -> Source:
    """Build a grounding source from original-script text."""
    from app.script_utils import to_latin_with_offsets

    latin, offsets = to_latin_with_offsets(original)
    return Source(latin=latin, original=original, offsets=offsets)
