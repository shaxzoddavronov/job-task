# Threshold calibration — rerank mode

Decision sets only: validation (template paraphrases), realistic-dev (hand-written) and
OOD-dev (`data/ood_dev.txt`). Rule: highest dev coverage with precision when answering
≥ 95% on dev and validation and ≤ 5% of OOD-dev answered; then the suggest threshold with
the highest dev "useful" rate (correct answer, or gold in the top-3 suggestions) such
that ≥ 80% of OOD-dev gets not_found (a suggestion for an off-topic question is cheap;
a 90% floor would reject a third of realistic queries — see the table below).

**Mode `rerank`. Chosen:** `ANSWER_THRESHOLD_RERANK=0.56`, `SUGGEST_THRESHOLD_RERANK=0.45`, `ANSWER_MIN_MARGIN_RERANK=0.02`

| set | coverage | precision when answering | suggest | not_found | useful |
|---|---|---|---|---|---|
| realistic-dev | 85.0 | 95.6 | 14.0 | 0.9 | 95.3 |
| validation | 98.6 | 99.8 | 1.4 | 0.0 | 99.8 |
| OOD-dev | 0.0 | – | 6.7 | 93.3 | 0.0 |

Trade-off at margin 0.02 (suggest threshold capped at the answer threshold), percentages:

| answer ≥ | dev coverage | dev precision | dev useful | val coverage | val precision | OOD answered | OOD not_found |
|---|---|---|---|---|---|---|---|
| 0.40 | 95.3 | 93.1 | 92.5 | 98.6 | 99.8 | 10.0 | 70.0 |
| 0.45 | 95.3 | 93.1 | 92.5 | 98.6 | 99.8 | 3.3 | 93.3 |
| 0.50 | 91.6 | 93.9 | 93.5 | 98.6 | 99.8 | 0.0 | 93.3 |
| 0.55 | 88.8 | 93.7 | 93.5 | 98.6 | 99.8 | 0.0 | 93.3 |
| 0.60 | 81.3 | 95.4 | 95.3 | 98.6 | 99.8 | 0.0 | 93.3 |
| 0.65 | 74.8 | 96.2 | 96.3 | 98.6 | 99.8 | 0.0 | 93.3 |
| 0.70 | 65.4 | 97.1 | 97.2 | 98.2 | 99.8 | 0.0 | 93.3 |
| 0.75 | 57.9 | 96.8 | 97.2 | 98.2 | 99.8 | 0.0 | 93.3 |
| 0.80 | 46.7 | 96.0 | 97.2 | 98.0 | 99.8 | 0.0 | 93.3 |
| 0.85 | 23.4 | 100.0 | 99.1 | 95.6 | 99.8 | 0.0 | 93.3 |
| 0.90 | 4.7 | 100.0 | 99.1 | 73.1 | 99.7 | 0.0 | 93.3 |
| 0.95 | 0.0 | – | 99.1 | 3.4 | 100.0 | 0.0 | 93.3 |
