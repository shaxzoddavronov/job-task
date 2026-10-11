"""Entity tagging: law / article / number references and gazetteer names → canonical tokens.

Tokens: LAW_pq_4567, ART_15, NUM_4567, ORG_<id>, TERM_<id>. Tagging runs on text that went
through normalize_script (Latin, case kept) so case-sensitive aliases like `ERI` still work.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from app.core.config import Settings
from app.core.normalize import clean_text, normalize_script

# Optional Uzbek suffix chain after a gazetteer alias: plural, possessive, case.
# Covers DXMga, DXM-ga, DXM ga, markaziga, ID-kartamni, inspeksiyasiga, ...
_CASE = r"(?:ning|ni|ga|ka|qa|dagi|da|ta|dan|tan|gacha|dek|day)"
UZBEK_SUFFIX = (
    r"(?i:\s" + _CASE + r"(?![\w'])"  # detached case suffix: "DXM ga"
    r"|-?(?:lar)?(?:ingiz|ngiz|imiz|miz|ing|ng|im|m|si|i)?" + _CASE + r"?)"
)
_LEFT = r"(?<![\w'])"
_RIGHT = r"(?![\w'])"

LAW_RE = re.compile(
    _LEFT + r"(pf|pq|o'?rq|vmq)\s*[-–—]?\s*(\d{1,5})"
    r"(?:\s*[-–—]?\s*son(?:li|i|ga|da|dan|ning)?)?" + _RIGHT,
    re.IGNORECASE,
)
ARTICLE_RE = re.compile(_LEFT + r"(\d{1,4})\s*[-–—]?\s*modda\w*", re.IGNORECASE)
NUMBER_RE = re.compile(
    _LEFT + r"(\d{1,6})\s*[-–—]?\s*son(?:li|i|ga|da|dan|ning)?" + _RIGHT, re.IGNORECASE
)

ENTITY_TYPES = ("ORG", "TERM")
_ID_RE = re.compile(r"^[a-z0-9_]+$")


@dataclass(frozen=True)
class Entity:
    type: str  # LAW | ART | NUM | ORG | TERM
    value: str  # canonical value, e.g. "pq_4567", "15", "dxm"
    token: str  # token written into the text, e.g. "LAW_pq_4567"
    surface: str  # text as it appeared (after script normalization)


@dataclass(frozen=True)
class TaggedText:
    text: str  # cleaned text with entity tokens
    entities: list[Entity] = field(default_factory=list)

    @property
    def tokens(self) -> set[str]:
        return {e.token for e in self.entities}


@dataclass(frozen=True)
class GazetteerEntry:
    id: str
    type: str
    aliases: tuple[str, ...]
    case_sensitive: tuple[str, ...] = ()

    @property
    def token(self) -> str:
        return f"{self.type}_{self.id}"


def _alias_key(text: str) -> str:
    return re.sub(r"[-\s]+", " ", text.lower()).strip()


def _alias_pattern(alias: str) -> str:
    words = re.split(r"[-\s]+", alias.strip())
    return r"[-\s]*".join(re.escape(w) for w in words)


class Gazetteer:
    """Organization and term aliases from data/gazetteer.yaml, matched longest first."""

    def __init__(self, entries: list[GazetteerEntry]) -> None:
        self.entries = entries
        self._by_key: dict[str, GazetteerEntry] = {}
        patterns: list[tuple[int, str]] = []
        for entry in entries:
            sensitive = {normalize_script(a) for a in entry.case_sensitive}
            for alias in {normalize_script(a) for a in (*entry.aliases, *sensitive)}:
                key = _alias_key(alias)
                other = self._by_key.setdefault(key, entry)
                if other is not entry:
                    raise ValueError(f"Alias {alias!r} is used by both {other.id} and {entry.id}")
                body = _alias_pattern(alias)
                patterns.append((len(alias), body if alias in sensitive else f"(?i:{body})"))
        patterns.sort(key=lambda p: -p[0])
        self._regex = (
            re.compile(
                _LEFT
                + "(?P<alias>"
                + "|".join(p for _, p in patterns)
                + ")"
                + UZBEK_SUFFIX
                + _RIGHT
            )
            if patterns
            else None
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "Gazetteer":
        if not path.is_file():
            return cls([])
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
        entries = []
        for item in raw:
            entry = GazetteerEntry(
                id=str(item["id"]),
                type=str(item.get("type", "ORG")).upper(),
                aliases=tuple(str(a) for a in item.get("aliases", [])),
                case_sensitive=tuple(str(a) for a in item.get("case_sensitive", [])),
            )
            if not _ID_RE.match(entry.id) or entry.type not in ENTITY_TYPES:
                raise ValueError(
                    f"Bad gazetteer entry {item!r}: id must match [a-z0-9_]+, "
                    f"type one of {ENTITY_TYPES}"
                )
            if not entry.aliases and not entry.case_sensitive:
                raise ValueError(f"Gazetteer entry {entry.id!r} has no aliases")
            entries.append(entry)
        return cls(entries)

    def tag(self, text: str, found: list[Entity]) -> str:
        if self._regex is None:
            return text

        def repl(m: re.Match) -> str:
            entry = self._by_key[_alias_key(m.group("alias"))]
            found.append(Entity(entry.type, entry.id, entry.token, m.group(0)))
            return f" {entry.token} "

        return self._regex.sub(repl, text)


class EntityExtractor:
    """Runs the full preprocessing pipeline and returns tagged text plus entities."""

    def __init__(self, gazetteer: Gazetteer | None = None, strip_apostrophes: bool = False):
        self.gazetteer = gazetteer or Gazetteer([])
        self.strip_apostrophes = strip_apostrophes

    @classmethod
    def from_settings(cls, settings: Settings) -> "EntityExtractor":
        return cls(
            Gazetteer.from_yaml(settings.gazetteer_path),
            strip_apostrophes=settings.normalize_strip_apostrophes,
        )

    def tag(self, text: str) -> tuple[str, list[Entity]]:
        """Replace entities in script-normalized text with tokens. Laws go first."""
        found: list[Entity] = []

        def law(m: re.Match) -> str:
            prefix = m.group(1).lower().replace("'", "")
            value = f"{prefix}_{int(m.group(2))}"
            found.append(Entity("LAW", value, f"LAW_{value}", m.group(0)))
            return f" LAW_{value} "

        def numbered(kind: str):
            def repl(m: re.Match) -> str:
                value = str(int(m.group(1)))
                found.append(Entity(kind, value, f"{kind}_{value}", m.group(0)))
                return f" {kind}_{value} "

            return repl

        text = LAW_RE.sub(law, text)
        text = ARTICLE_RE.sub(numbered("ART"), text)
        text = NUMBER_RE.sub(numbered("NUM"), text)
        text = self.gazetteer.tag(text, found)
        return text, found

    def process(self, raw: str) -> TaggedText:
        tagged, entities = self.tag(normalize_script(raw))
        return TaggedText(clean_text(tagged, self.strip_apostrophes), entities)
