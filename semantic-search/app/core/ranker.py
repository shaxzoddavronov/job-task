"""From document hits to answer candidates. Entity-aware re-ranking is added in Phase 5."""


def aggregate_max(
    hits: list[tuple[int, float]], doc_answer_ids: list[int]
) -> list[tuple[int, float]]:
    """Map document hits to answer ids, keep each answer's best score, sort best first."""
    best: dict[int, float] = {}
    for doc_idx, score in hits:
        answer_id = doc_answer_ids[doc_idx]
        if score > best.get(answer_id, float("-inf")):
            best[answer_id] = score
    return sorted(best.items(), key=lambda item: -item[1])
