"""Hand-written task labels for the short lex.uz documents.

Plan section 11, Phase 2. Details (organization, number, date, signatory) come
from lex.uz metadata via ``fetch_lexuz.py``; the *tasks* below were read off
the documents by hand, which is the part no script can do.

Labels live here as code rather than as edited JSON so the judgement calls are
reviewable in one place, and so all four variants of each document (Latin and
Cyrillic × text and scanned) are generated from one source.

Judgement calls, applied consistently:

* A clause is a task when it assigns work — typically the passive imperative
  ``-sin``/``-sun`` ("tashkil etilsin", "kiritsin", "koʻrsin").
* ``Belgilab qoʻyilsinki … hisoblanadi`` defines status rather than assigning
  work, so those clauses and their sub-items are *not* tasks. They are the
  negative examples the synthetic set lacks.
* Where a parent clause assigns a bundle of functions and its sub-items merely
  enumerate them, the parent is one task and the sub-items are not separate
  tasks.
* An entry-into-force clause ("… kuchga kiradi") is not a task.
* ``executors`` is empty when the clause names nobody. Plan section 0: missing
  values are never invented.
* ``deadline_date`` resolves a relative deadline ("ikki oy muddatda") against
  the document date.

Run ``python eval/label_lexuz.py`` after ``fetch_lexuz.py``.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_synthetic import _add_months
from translit_lat2cyr import to_cyrillic

ROOT = Path(__file__).resolve().parent.parent
LABEL_DIR = ROOT / "eval" / "labels"

#: doc stem -> (document date, [(task, deadline_raw, deadline_spec, executors)])
#: ``deadline_spec`` is None, ("weeks", n) or ("months", n).
LABELS: dict[str, tuple[str, list[tuple[str, str, tuple | None, list[str]]]]] = {
    "lexuz_8519155": (  # PF-213, 03.10.2026 — establishing a ministry
        "2026-10-03",
        [
            (
                "Maktabgacha va maktab taʼlimi vazirligi hamda Oliy taʼlim, fan va "
                "innovatsiyalar vazirligi negizida Taʼlim va ilm-fan vazirligi tashkil etilsin",
                "", None, [],
            ),
            (
                "Vazirlar Mahkamasi bir hafta muddatda Taʼlim va ilm-fan vazirligi "
                "faoliyatini natijadorlikka yoʻnaltirilgan va samarali boshqaruv tamoyillari "
                "asosida tashkil etish, uning vazifalari va tashkiliy tuzilmasini belgilash "
                "boʻyicha Oʻzbekiston Respublikasi Prezidentining qarori loyihasini kiritsin",
                "bir hafta muddatda", ("weeks", 1), ["Vazirlar Mahkamasi"],
            ),
            (
                "Taʼlim va ilm-fan vazirligi manfaatdor vazirlik va idoralar bilan birgalikda "
                "ikki oy muddatda qonunchilik hujjatlariga ushbu Farmondan kelib chiqadigan "
                "oʻzgartirish va qoʻshimchalar toʻgʻrisida Vazirlar Mahkamasiga takliflar kiritsin",
                "ikki oy muddatda", ("months", 2),
                ["Taʼlim va ilm-fan vazirligi", "manfaatdor vazirlik va idoralar"],
            ),
            (
                "Ushbu Farmonning ijrosini nazorat qilish Oʻzbekiston Respublikasi Bosh vaziri "
                "A.N. Aripov zimmasiga yuklansin",
                "", None, ["Oʻzbekiston Respublikasi Bosh vaziri A.N. Aripov"],
            ),
        ],
    ),
    "lexuz_8517348": (  # PQ-358, 02.10.2026 — university governance
        "2026-10-02",
        [
            (
                "Maktabgacha va maktab taʼlimi vazirligi hamda Ixtisoslashtirilgan taʼlim "
                "muassasalari agentligi tizimidagi oliy taʼlim muassasalari ilovaga muvofiq "
                "Oliy taʼlim, fan va innovatsiyalar vazirligi tizimiga oʻtkazilsin",
                "", None, [],
            ),
            (
                "Maktabgacha va maktab taʼlimi vazirligi markaziy apparati tuzilmasida "
                "vazirning pedagogik taʼlimni rivojlantirish masalalari boʻyicha oʻrinbosari — "
                "Nizomiy nomidagi Oʻzbekiston milliy pedagogika universiteti rektori lavozimi "
                "tugatilsin",
                "", None, [],
            ),
            (
                "Oʻzbekiston milliy pedagogika universiteti Nizomiy Ganjaviy nomidagi "
                "Oʻzbekiston milliy pedagogika universiteti deb qayta nomlansin",
                "", None, [],
            ),
            (
                "Maktabgacha va maktab taʼlimi vazirligining pedagog kadrlar tayyorlash "
                "sohasidagi quyidagi vazifalari Oliy taʼlim, fan va innovatsiyalar vazirligiga "
                "yuklansin",
                "", None, ["Oliy taʼlim, fan va innovatsiyalar vazirligi"],
            ),
            (
                "Maktabgacha va maktab taʼlimi vazirligi Oliy taʼlim, fan va innovatsiyalar "
                "vazirligi bilan birgalikda uch oy muddatda qonunchilik hujjatlariga ushbu "
                "qarordan kelib chiqadigan oʻzgartirish va qoʻshimchalar toʻgʻrisida Vazirlar "
                "Mahkamasiga taklif kiritsin",
                "uch oy muddatda", ("months", 3),
                ["Maktabgacha va maktab taʼlimi vazirligi",
                 "Oliy taʼlim, fan va innovatsiyalar vazirligi"],
            ),
            (
                # Quoted in full: an abbreviated label scored only 75 against
                # the document and was counted as both a miss and a false
                # positive. Labels must be verbatim.
                "Iqtisodiyot va moliya vazirligi Oliy taʼlim, fan va innovatsiyalar vazirligi "
                "hamda Maktabgacha va maktab taʼlimi vazirligi bilan birgalikda bir oy muddatda "
                "2026-yil uchun Oʻzbekiston Respublikasining respublika budjeti parametrlari "
                "doirasida Maktabgacha va maktab taʼlimi vazirligi hamda Oliy taʼlim, fan va "
                "innovatsiyalar vazirligiga birinchi darajali budjet mablagʻlarini "
                "taqsimlovchilar sifatida ajratilgan budjet mablagʻlarini ushbu qaror "
                "talablaridan kelib chiqqan holda qayta taqsimlash va budjet parametrlariga "
                "tegishli oʻzgartirishlar kiritish choralarini koʻrsin",
                "bir oy muddatda", ("months", 1),
                ["Iqtisodiyot va moliya vazirligi",
                 "Oliy taʼlim, fan va innovatsiyalar vazirligi",
                 "Maktabgacha va maktab taʼlimi vazirligi"],
            ),
            (
                "Ushbu qaror ijrosini nazorat qilish Oʻzbekiston Respublikasi Bosh vaziri "
                "A.N. Aripov zimmasiga yuklansin",
                "", None, ["Oʻzbekiston Respublikasi Bosh vaziri A.N. Aripov"],
            ),
        ],
    ),
    "lexuz_8517274": (  # 3948, 30.09.2026 — Central Bank amending act
        "2026-09-30",
        [
            (
                "Oʻzbekiston Respublikasi Markaziy banki boshqaruvi tomonidan qabul qilingan "
                "ayrim idoraviy normativ-huquqiy hujjatlarga ilovaga muvofiq qoʻshimchalar "
                "kiritilsin",
                "", None, [],
            ),
        ],
    ),
}


def _resolve(doc_date: str, spec: tuple | None) -> str | None:
    if spec is None:
        return None
    unit, n = spec
    base = dt.date.fromisoformat(doc_date)
    target = base + dt.timedelta(weeks=n) if unit == "weeks" else _add_months(base, n)
    return target.isoformat()


def apply_labels() -> int:
    written = 0
    for stem, (doc_date, tasks) in LABELS.items():
        for tag, script in (("lat", "latin"), ("cyr", "cyrillic")):
            convert = to_cyrillic if script == "cyrillic" else (lambda s: s)
            for suffix in ("", "_scan"):
                path = LABEL_DIR / f"{stem}_{tag}{suffix}.json"
                if not path.exists():
                    print(f"  missing stub: {path.name} (run fetch_lexuz.py first)")
                    continue
                data = json.loads(path.read_text(encoding="utf-8"))
                data["tasks"] = [
                    {
                        "task": convert(task),
                        "deadline_raw": convert(raw),
                        "deadline_date": _resolve(doc_date, spec),
                        "executors": [convert(e) for e in executors],
                    }
                    for task, raw, spec, executors in tasks
                ]
                data.pop("_needs_task_labels", None)
                path.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                written += 1
    return written


def main() -> int:
    n = apply_labels()
    labelled = len(LABELS)
    total_tasks = sum(len(t) for _, t in LABELS.values())
    print(f"{n} label files updated — {labelled} documents, {total_tasks} tasks each variant")
    print("Long documents (lexuz_8517342, lexuz_8517651) are left unlabelled "
          "on purpose; they exercise multi-page extraction qualitatively.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
