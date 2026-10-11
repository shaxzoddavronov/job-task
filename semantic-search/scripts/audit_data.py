"""Data quality audit: prints a summary and writes reports/data_audit.{md,json}.

Run with `make audit` (or `python -m scripts.audit_data`). Nothing is cleaned here; the
report is the basis for proposing cleaning rules.

Text is normalized and entities are tagged with the same code the engine uses
(app/core/normalize.py, app/core/entities.py). The org-phrase and abbreviation patterns below
are audit heuristics only; scripts/build_gazetteer.py reuses them to suggest gazetteer entries.
"""

import json
import re
from collections import Counter
from collections.abc import Iterator
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from app.core.config import get_settings
from app.core.data import FileSource, RawData, as_bool
from app.core.entities import EntityExtractor
from app.core.normalize import APOSTROPHE_VARIANTS, normalize

APOSTROPHES = {ch: f"U+{ord(ch):04X}" for ch in "'" + APOSTROPHE_VARIANTS}
CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
ORG_SUFFIXES = (
    "vazirligi",
    "qo'mitasi",
    "markazi",
    "agentligi",
    "inspeksiyasi",
    "boshqarmasi",
    "xizmati",
    "palatasi",
    "sudi",
    "hokimligi",
    "departamenti",
    "jamg'armasi",
)
ORG_RE = re.compile(r"((?:[\w']+\s+){1,3}(?:" + "|".join(ORG_SUFFIXES) + r"))\b")
# All-caps (DXM, FHDYO) and Uzbek digraph abbreviations with lowercase letters (YaTT, MChJ).
ABBR_RE = re.compile(r"\b[A-ZЎҚҒҲ](?:[a-z]?[A-ZЎҚҒҲ])+\b")
LAW_TYPES = {"PF", "PQ", "ORQ", "VMQ", "ПФ", "ПҚ", "ЎРҚ", "ВМҚ"}

# Filler patterns seen in generated paraphrases ("<stem> haqida ma'lumot bering").
TEMPLATES = {
    "... haqida ma'lumot bering": r"haqida ma'lumot bering\W*$",
    "... bo'yicha ma'lumot kerak": r"bo'yicha ma'lumot kerak\W*$",
    "Menga ... haqida ayting": r"^menga .* haqida ayting",
    "Iltimos, ayting: ...": r"^iltimos,? ayting",
    "... qanaqa ...": r"\bqanaqa\b",
}
TEMPLATE_WORDS = set(
    [
        "qanday",
        "qanaqa",
        "haqida",
        "ma'lumot",
        "bering",
        "bo'yicha",
        "kerak",
        "menga",
        "ayting",
        "iltimos",
    ]
)
NEAR_DUP_THRESHOLD = 0.9
MAX_EXAMPLES = 15


def abbreviations(text: str) -> list[str]:
    return [m.group(0) for m in ABBR_RE.finditer(text) if m.group(0) not in LAW_TYPES]


def has(series: pd.Series, pattern: re.Pattern) -> pd.Series:
    return series.map(lambda t: bool(pattern.search(t)))


def describe_lengths(series: pd.Series) -> dict:
    if series.empty:
        return {}
    q = series.quantile([0.05, 0.5, 0.95])
    return {
        "min": int(series.min()),
        "p5": round(float(q[0.05]), 1),
        "median": round(float(q[0.5]), 1),
        "p95": round(float(q[0.95]), 1),
        "max": int(series.max()),
        "mean": round(float(series.mean()), 1),
    }


def md_table(headers: list[str], rows: list[list]) -> list[str]:
    def cell(v) -> str:
        return str(v).replace("|", "\\|").replace("\n", " ")

    if not rows:
        return ["_none_", ""]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return lines + [""]


def short(text, n: int = 90) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def pct(part: int, total: int) -> str:
    return f"{part} ({100 * part / total:.1f}%)" if total else "0"


def near_duplicate_pairs(
    texts: list[str], threshold: float, chunk: int = 1000
) -> Iterator[tuple[int, int, float]]:
    """Yield (i, j, cosine) for i < j with char-TF-IDF cosine >= threshold."""
    if len(texts) < 2:
        return
    matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(texts)
    for start in range(0, matrix.shape[0], chunk):
        sims = (matrix[start : start + chunk] @ matrix.T).tocoo()
        for r, c, v in zip(sims.row, sims.col, sims.data, strict=True):
            i = start + r
            if i < c and v >= threshold:
                yield i, int(c), float(v)


def audit(raw: RawData, extractor: EntityExtractor | None = None) -> tuple[list[str], dict]:
    extractor = extractor or EntityExtractor()
    answers, questions = raw.answers.copy(), raw.questions.copy()
    md: list[str] = ["# Data audit", ""]
    summary: dict = {}

    # --- Counts, statuses, activity -------------------------------------------------------
    questions["_active"] = questions["is_active"].map(as_bool)
    active = questions[questions["_active"] == True]  # noqa: E712
    summary["counts"] = {
        "answers": len(answers),
        "questions": len(questions),
        "questions_active": len(active),
        "questions_inactive": int((questions["_active"] == False).sum()),  # noqa: E712
        "questions_is_active_unparsed": int(questions["_active"].isna().sum()),
    }
    md += ["## 1. Row counts and status", ""]
    md += md_table(["metric", "value"], [[k, v] for k, v in summary["counts"].items()])

    status_counts = answers["status"].astype(str).value_counts(dropna=False)
    active_raw = questions["is_active"].astype(str).value_counts(dropna=False)
    summary["answer_status_values"] = status_counts.to_dict()
    summary["is_active_values"] = active_raw.to_dict()
    md += ["Answer `status` values:", ""]
    md += md_table(["status", "answers"], [[k, v] for k, v in status_counts.items()])
    md += ["Raw `is_active` values:", ""]
    md += md_table(["is_active", "questions"], [[k, v] for k, v in active_raw.items()])

    # Active questions per answer status: tells us whether `status` marks inactive answers.
    status_by_id = answers.set_index("id")["status"].astype(str)
    if answers["id"].is_unique:
        aq_status = active["answer_id"].map(status_by_id).fillna("<orphan>")
        rows = aq_status.value_counts().items()
        md += ["Active questions by the status of their answer:", ""]
        md += md_table(["answer status", "active questions"], [[k, v] for k, v in rows])

    # --- Integrity --------------------------------------------------------------------------
    md += ["## 2. Integrity", ""]
    answer_ids = set(answers["id"])
    dup_answer_ids = int(answers["id"].duplicated().sum())
    dup_question_ids = int(questions["id"].duplicated().sum())
    orphan_q = active[~active["answer_id"].isin(answer_ids)]
    answers_without_active = answers[~answers["id"].isin(set(active["answer_id"]))]
    answers_without_any = answers[~answers["id"].isin(set(questions["answer_id"]))]
    nulls = {f"answers.{c}": int(answers[c].isna().sum()) for c in answers.columns}
    nulls |= {
        f"questions.{c}": int(questions[c].isna().sum())
        for c in questions.columns
        if not c.startswith("_")
    }
    summary["integrity"] = {
        "duplicate_answer_ids": dup_answer_ids,
        "duplicate_question_ids": dup_question_ids,
        "orphan_active_questions": len(orphan_q),
        "answers_without_active_questions": len(answers_without_active),
        "answers_without_any_questions": len(answers_without_any),
        "nulls": {k: v for k, v in nulls.items() if v},
    }
    md += md_table(
        ["check", "count"],
        [[k, v] for k, v in summary["integrity"].items() if k != "nulls"],
    )
    md += ["Null values per column (non-zero only):", ""]
    md += md_table(["column", "nulls"], [[k, v] for k, v in summary["integrity"]["nulls"].items()])
    if len(orphan_q):
        md += ["Orphan question examples:", ""]
        md += md_table(
            ["question id", "answer_id", "question"],
            [
                [r.id, r.answer_id, short(r.question)]
                for r in orphan_q.head(MAX_EXAMPLES).itertuples()
            ],
        )

    # --- Questions per answer --------------------------------------------------------------
    md += ["## 3. Active questions per answer", ""]
    per_answer = active[active["answer_id"].isin(answer_ids)].groupby("answer_id").size()
    per_answer = per_answer.reindex(sorted(answer_ids, key=str), fill_value=0)
    buckets = pd.cut(
        per_answer,
        bins=[-1, 0, 1, 2, 5, 10, 20, np.inf],
        labels=["0", "1", "2", "3-5", "6-10", "11-20", "21+"],
    ).value_counts(sort=False)
    summary["questions_per_answer"] = {
        **describe_lengths(per_answer),
        "answers_with_1": int((per_answer == 1).sum()),
        "eval_pool_ge2": int((per_answer >= 2).sum()),
        "with_validation_ge3": int((per_answer >= 3).sum()),
        "buckets": {str(k): int(v) for k, v in buckets.items()},
    }
    md += md_table(["questions per answer", "answers"], [[k, v] for k, v in buckets.items()])
    md += md_table(
        ["stat", "value"],
        [[k, v] for k, v in summary["questions_per_answer"].items() if k != "buckets"],
    )

    # --- Duplicates and conflicts ----------------------------------------------------------
    md += ["## 4. Duplicates and conflicts (active questions, rough normalization)", ""]
    active = active.assign(_norm=active["question"].fillna("").map(normalize))
    groups = active.groupby("_norm")["answer_id"].agg(["size", "nunique"])
    exact_dup_rows = int((groups["size"] - 1).clip(lower=0).sum())
    conflicts = groups[groups["nunique"] > 1]
    same_answer_dups = int(((groups["size"] - groups["nunique"]).clip(lower=0)).sum())

    texts = active["_norm"].tolist()
    answer_of = active["answer_id"].tolist()
    near_same, near_diff, diff_examples = 0, 0, []
    for i, j, sim in near_duplicate_pairs(texts, NEAR_DUP_THRESHOLD):
        if texts[i] == texts[j]:
            continue
        if answer_of[i] == answer_of[j]:
            near_same += 1
        else:
            near_diff += 1
            diff_examples.append((sim, i, j))
    diff_examples.sort(reverse=True)

    summary["duplicates"] = {
        "exact_duplicate_rows": exact_dup_rows,
        "exact_duplicates_same_answer": same_answer_dups,
        "conflicting_texts": len(conflicts),
        "conflicting_rows": int(conflicts["size"].sum()),
        f"near_dup_pairs_same_answer(cos>={NEAR_DUP_THRESHOLD})": near_same,
        f"near_dup_pairs_diff_answer(cos>={NEAR_DUP_THRESHOLD})": near_diff,
    }
    md += md_table(["check", "count"], [[k, v] for k, v in summary["duplicates"].items()])
    if len(conflicts):
        md += ["Conflicts: identical normalized question → different answers:", ""]
        rows = []
        for norm in conflicts.sort_values("size", ascending=False).index[:MAX_EXAMPLES]:
            ids = sorted(
                active.loc[active["_norm"] == norm, "answer_id"].unique().tolist(), key=str
            )
            rows.append([short(norm), ", ".join(map(str, ids))])
        md += md_table(["question", "answer_ids"], rows)
    if diff_examples:
        md += ["Near-duplicates pointing to different answers (highest similarity first):", ""]
        md += md_table(
            ["cos", "question A (answer)", "question B (answer)"],
            [
                [
                    f"{s:.3f}",
                    f"{short(texts[i], 60)} ({answer_of[i]})",
                    f"{short(texts[j], 60)} ({answer_of[j]})",
                ]
                for s, i, j in diff_examples[:MAX_EXAMPLES]
            ],
        )

    # --- Short texts and lengths -----------------------------------------------------------
    md += ["## 5. Empty / short texts and lengths", ""]
    q_text = active["question"].fillna("").astype(str).str.strip()
    a_text = answers["answer"].fillna("").astype(str).str.strip()
    q_words, a_words = q_text.str.split().str.len(), a_text.str.split().str.len()
    short_q = active[(q_words < 3) | (q_text.str.len() < 10)]
    short_a = answers[a_text.str.len() < 30]
    summary["short_texts"] = {
        "empty_questions": int((q_text == "").sum()),
        "short_questions(<3 words or <10 chars)": len(short_q),
        "empty_answers": int((a_text == "").sum()),
        "short_answers(<30 chars)": len(short_a),
    }
    summary["lengths"] = {
        "question_chars": describe_lengths(q_text.str.len()),
        "question_words": describe_lengths(q_words),
        "answer_chars": describe_lengths(a_text.str.len()),
        "answer_words": describe_lengths(a_words),
    }
    md += md_table(["check", "count"], [[k, v] for k, v in summary["short_texts"].items()])
    md += md_table(
        ["field", "min", "p5", "median", "p95", "max", "mean"],
        [[k, *v.values()] for k, v in summary["lengths"].items() if v],
    )
    if len(short_q):
        md += ["Short question examples:", ""]
        md += md_table(
            ["id", "answer_id", "question"],
            [
                [r.id, r.answer_id, short(r.question)]
                for r in short_q.head(MAX_EXAMPLES).itertuples()
            ],
        )
    if len(short_a):
        md += ["Short answer examples:", ""]
        md += md_table(
            ["id", "answer"],
            [[r.id, short(r.answer)] for r in short_a.head(MAX_EXAMPLES).itertuples()],
        )

    # --- Script and apostrophes ------------------------------------------------------------
    md += ["## 6. Script and apostrophes", ""]
    q_cyr = int(has(q_text, CYRILLIC_RE).sum())
    a_cyr = int(has(a_text, CYRILLIC_RE).sum())
    apos_rows = []
    summary["script"] = {"questions_with_cyrillic": q_cyr, "answers_with_cyrillic": a_cyr}
    summary["apostrophes"] = {}
    for ch, code in APOSTROPHES.items():
        nq, na = (
            int(q_text.str.contains(ch, regex=False).sum()),
            int(a_text.str.contains(ch, regex=False).sum()),
        )
        summary["apostrophes"][code] = {"questions": nq, "answers": na}
        apos_rows.append([f"{code} ( {ch} )", nq, na])
    md += md_table(
        ["script", "questions", "answers"],
        [["contains Cyrillic", pct(q_cyr, len(q_text)), pct(a_cyr, len(a_text))]],
    )
    md += md_table(["apostrophe", "questions", "answers"], apos_rows)

    # --- Entities --------------------------------------------------------------------------
    md += ["## 7. Entities", ""]
    q_ents = q_text.map(lambda t: extractor.process(t).entities)
    a_ents = a_text.map(lambda t: extractor.process(t).entities)

    def of_type(ents: pd.Series, kind: str) -> pd.Series:
        return ents.map(lambda es: {e.value for e in es if e.type == kind})

    q_laws, a_laws = of_type(q_ents, "LAW"), of_type(a_ents, "LAW")
    q_norm_text, a_norm_text = q_text.map(normalize), a_text.map(normalize)
    ent_rows, summary["entities"] = [], {}

    checks = {
        "law number": (q_laws.map(bool), a_laws.map(bool)),
        "article (N-modda)": (of_type(q_ents, "ART").map(bool), of_type(a_ents, "ART").map(bool)),
        "N-son": (of_type(q_ents, "NUM").map(bool), of_type(a_ents, "NUM").map(bool)),
        "gazetteer ORG": (of_type(q_ents, "ORG").map(bool), of_type(a_ents, "ORG").map(bool)),
        "gazetteer TERM": (of_type(q_ents, "TERM").map(bool), of_type(a_ents, "TERM").map(bool)),
        "org phrase (suffix)": (has(q_norm_text, ORG_RE), has(a_norm_text, ORG_RE)),
        "all-caps abbreviation": (
            q_text.map(abbreviations).map(bool),
            a_text.map(abbreviations).map(bool),
        ),
    }
    for name, (qm, am) in checks.items():
        summary["entities"][name] = {"questions": int(qm.sum()), "answers": int(am.sum())}
        ent_rows.append([name, pct(int(qm.sum()), len(q_text)), pct(int(am.sum()), len(a_text))])
    md += md_table(["entity", "questions", "answers"], ent_rows)

    # Does the gold answer mention the law cited in the question? Justifies the answer index.
    answer_laws = dict(zip(answers["id"], a_laws, strict=True))
    q_with_law = [
        (laws, answer_laws.get(aid, set()))
        for laws, aid in zip(q_laws, active["answer_id"], strict=True)
        if laws
    ]
    in_gold = sum(1 for ql, al in q_with_law if ql & al)
    all_laws_in_data = set().union(*a_laws, *q_laws) if len(a_laws) else set()
    summary["entities"]["distinct_law_numbers"] = len(all_laws_in_data)
    summary["entities"]["questions_with_law_whose_answer_has_it"] = in_gold
    md += [
        f"- Distinct law numbers in the data: **{len(all_laws_in_data)}**",
        f"- Questions citing a law whose gold answer also cites it: "
        f"**{pct(in_gold, len(q_with_law))}** of {len(q_with_law)}",
        "",
    ]

    law_counter = Counter(k for laws in list(q_laws) + list(a_laws) for k in laws)
    org_counter = Counter(m.group(1) for t in a_norm_text for m in ORG_RE.finditer(t))
    abbr_counter = Counter(abbr for t in pd.concat([q_text, a_text]) for abbr in abbreviations(t))
    md += ["Top law numbers:", ""] + md_table(["law", "mentions"], law_counter.most_common(20))
    md += ["Top org phrases in answers (candidates; gazetteer is built in Phase 2):", ""]
    md += md_table(["phrase", "mentions"], org_counter.most_common(25))
    md += ["Top all-caps abbreviations:", ""] + md_table(
        ["abbr", "mentions"], abbr_counter.most_common(25)
    )
    summary["entities"]["top_laws"] = dict(law_counter.most_common(20))
    summary["entities"]["top_abbreviations"] = dict(abbr_counter.most_common(25))

    # --- Paraphrase structure ----------------------------------------------------------------
    # High overlap between paraphrases of one answer makes a random split optimistic.
    md += ["## 8. Paraphrase structure", ""]
    q_low = active["question"].fillna("").astype(str).str.lower()
    hits = pd.DataFrame({name: has(q_low, re.compile(rx)) for name, rx in TEMPLATES.items()})
    templated = hits.any(axis=1)
    per_answer_templated = templated.groupby(active["answer_id"]).sum().value_counts().sort_index()
    jaccards = []
    for _, group in active.groupby("answer_id"):
        token_sets = [set(t.split()) - TEMPLATE_WORDS for t in group["_norm"]]
        jaccards += [len(x & y) / max(1, len(x | y)) for x, y in combinations(token_sets, 2)]
    jac = pd.Series(jaccards, dtype=float)
    summary["paraphrases"] = {
        "questions_matching_a_template": int(templated.sum()),
        "template_counts": {k: int(v) for k, v in hits.sum().items()},
        "answers_by_templated_question_count": {
            str(k): int(v) for k, v in per_answer_templated.items()
        },
        "same_answer_content_jaccard_median": round(float(jac.median()), 3) if len(jac) else None,
        "same_answer_content_jaccard_ge_0.6": round(float((jac >= 0.6).mean()), 3)
        if len(jac)
        else None,
    }
    md += md_table(["template", "questions"], [[k, v] for k, v in hits.sum().items()])
    md += [
        f"- Questions matching any template: **{pct(int(templated.sum()), len(active))}**",
        f"- Content-word Jaccard between paraphrases of the same answer (template words removed): "
        f"median **{summary['paraphrases']['same_answer_content_jaccard_median']}**, "
        f"share ≥ 0.6: **{summary['paraphrases']['same_answer_content_jaccard_ge_0.6']}**",
        "",
    ]
    md += md_table(
        ["templated questions per answer", "answers"], list(per_answer_templated.items())
    )

    # --- Categories and topics -------------------------------------------------------------
    md += ["## 9. Categories and topics", ""]
    summary["taxonomy"] = {}
    for col in ("category_id", "topic_id"):
        counts = answers[col].astype(str).value_counts()
        summary["taxonomy"][col] = {
            "distinct": int(counts.size),
            **describe_lengths(counts),
            "top": counts.head(15).to_dict(),
        }
        md += [
            f"`{col}`: {counts.size} distinct values; answers per value "
            f"min {counts.min()} / median {counts.median():g} / max {counts.max()}",
            "",
        ]
        md += md_table([col, "answers"], [[k, v] for k, v in counts.head(15).items()])

    return md, summary


def main() -> None:
    settings = get_settings()
    raw = FileSource(settings.answers_path, settings.questions_path).load_raw()
    md, summary = audit(raw, EntityExtractor.from_settings(settings))
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    (settings.reports_dir / "data_audit.md").write_text("\n".join(md), encoding="utf-8")
    (settings.reports_dir / "data_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "counts",
                    "integrity",
                    "questions_per_answer",
                    "duplicates",
                    "short_texts",
                    "script",
                )
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    print(f"\nFull report: {settings.reports_dir / 'data_audit.md'}")


if __name__ == "__main__":
    main()
