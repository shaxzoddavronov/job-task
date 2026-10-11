"""Query-side synonym expansion from data/synonyms.yaml.

Colloquial and Russian words ("prava", "spravka", "propiska") are mapped to the wording used
in the answer bank. A match appends the canonical phrase to the query; the query itself is kept.

    - canonical: haydovchilik guvohnomasi
      variants: [prava, voditelskiy prava]
"""

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.core.entities import UZBEK_SUFFIX
from app.core.normalize import normalize


@dataclass(frozen=True)
class SynonymGroup:
    canonical: str
    variants: tuple[str, ...]


class QueryExpander:
    def __init__(self, groups: list[SynonymGroup]) -> None:
        self.groups = groups
        self._rules: list[tuple[re.Pattern, str]] = []
        for group in groups:
            canonical = normalize(group.canonical)
            variants = sorted({normalize(v) for v in group.variants} - {""}, key=len, reverse=True)
            if not canonical or not variants:
                raise ValueError(f"Synonym group {group!r} needs a canonical form and variants")
            body = "|".join(r"\s+".join(map(re.escape, v.split())) for v in variants)
            pattern = re.compile(r"(?<![\w'])(?:" + body + ")" + UZBEK_SUFFIX + r"(?![\w'])")
            self._rules.append((pattern, canonical))

    @classmethod
    def from_yaml(cls, path: Path) -> "QueryExpander":
        if not path.is_file():
            return cls([])
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
        groups = []
        for item in raw:
            if not isinstance(item, dict) or "canonical" not in item or "variants" not in item:
                raise ValueError(f"Bad synonym entry {item!r}: needs 'canonical' and 'variants'")
            groups.append(SynonymGroup(str(item["canonical"]), tuple(map(str, item["variants"]))))
        return cls(groups)

    def expand(self, text: str) -> str:
        """Append the canonical phrase of every group whose variant occurs in `text`."""
        additions = []
        for pattern, canonical in self._rules:
            if pattern.search(text) and canonical not in text and canonical not in additions:
                additions.append(canonical)
        return " ".join([text, *additions]) if additions else text
