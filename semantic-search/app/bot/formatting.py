"""Pure helpers for turning API responses into Telegram messages (no I/O, easy to test)."""

import re

from app.bot import texts

TELEGRAM_LIMIT = 4096
BUTTON_CHARS = 60


def split_message(text: str, limit: int = TELEGRAM_LIMIT) -> list[str]:
    """Split text into chunks ≤ limit at paragraph, then sentence, then word boundaries."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current = ""
    units = [u for p in text.split("\n\n") for u in _fit(p, limit)]
    for unit in units:
        sep = "\n\n" if current and not current.endswith(" ") else ""
        if len(current) + len(sep) + len(unit) <= limit:
            current += sep + unit
        else:
            if current:
                chunks.append(current.strip())
            current = unit
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _fit(paragraph: str, limit: int) -> list[str]:
    """Break one paragraph into pieces ≤ limit: sentences, then words, then hard cuts."""
    if len(paragraph) <= limit:
        return [paragraph]
    pieces: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
        while len(sentence) > limit:
            cut = sentence.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            pieces.append(sentence[:cut])
            sentence = sentence[cut:].lstrip()
        pieces.append(sentence)
    # re-pack sentences greedily so chunks stay large
    packed, current = [], ""
    for piece in pieces:
        if current and len(current) + 1 + len(piece) > limit:
            packed.append(current)
            current = piece
        else:
            current = f"{current} {piece}" if current else piece
    if current:
        packed.append(current)
    return packed


def button_label(question: str, limit: int = BUTTON_CHARS) -> str:
    question = " ".join(question.split())
    return question if len(question) <= limit else question[: limit - 1].rstrip() + "…"


def unknown_laws(result: dict) -> str:
    """Law / article references from the query, as the user wrote them."""
    surfaces = [e["surface"] for e in result.get("entities", []) if e["type"] in ("LAW", "ART")]
    return ", ".join(dict.fromkeys(surfaces)) or "?"


def _pct(value: float | None) -> str:
    return "–" if value is None else f"{100 * value:.0f}%"


def about_text(stats: dict) -> str:
    ev = stats.get("eval") or {}
    realistic, ood, test = ev.get("realistic"), ev.get("ood"), ev.get("test")
    if not realistic:
        return texts.ABOUT.format(metrics=texts.ABOUT_NO_METRICS)
    latency = test.get("latency_p50_ms") if test else None
    metrics = texts.ABOUT_METRICS.format(
        realistic_top1=_pct(realistic.get("top1")),
        realistic_top3=_pct(realistic.get("top3")),
        ood_rejected=_pct(ood.get("not_found_rate")) if ood else "–",
        latency="–" if latency is None else f"{latency:.0f} ms",
    )
    return texts.ABOUT.format(metrics=metrics)
