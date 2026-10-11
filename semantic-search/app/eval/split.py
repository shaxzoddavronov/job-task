"""Deterministic index / validation / test split over questions."""

import random
from collections import defaultdict
from dataclasses import dataclass

from app.core.data import Question


@dataclass(frozen=True)
class Split:
    index: list[Question]
    validation: list[Question]
    test: list[Question]


def split_questions(questions: list[Question], seed: int) -> Split:
    """Per answer with >= 2 questions: 1 → test, 1 → validation (if >= 3), rest → index.

    Answers with a single question stay in the index so they remain searchable.
    """
    by_answer: dict[int, list[Question]] = defaultdict(list)
    for q in questions:
        by_answer[q.answer_id].append(q)

    rng = random.Random(seed)
    index, validation, test = [], [], []
    for answer_id in sorted(by_answer):
        group = sorted(by_answer[answer_id], key=lambda q: q.id)
        if len(group) < 2:
            index += group
            continue
        rng.shuffle(group)
        test.append(group[0])
        if len(group) >= 3:
            validation.append(group[1])
            index += group[2:]
        else:
            index += group[1:]
    return Split(index=index, validation=validation, test=test)
