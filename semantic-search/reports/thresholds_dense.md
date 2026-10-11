# Threshold calibration — dense mode

Decision sets only: validation (template paraphrases), realistic-dev (hand-written) and
OOD-dev (`data/ood_dev.txt`). Rule: highest dev coverage with precision when answering
≥ 95% on dev and validation and ≤ 5% of OOD-dev answered; then the suggest threshold with
the highest dev "useful" rate (correct answer, or gold in the top-3 suggestions) such
that ≥ 80% of OOD-dev gets not_found (a suggestion for an off-topic question is cheap;
a 90% floor would reject a third of realistic queries — see the table below).

**Mode `dense`. Chosen:** `ANSWER_THRESHOLD_DENSE=0.65`, `SUGGEST_THRESHOLD_DENSE=0.62`, `ANSWER_MIN_MARGIN_DENSE=0.02`

| set | coverage | precision when answering | suggest | not_found | useful |
|---|---|---|---|---|---|
| realistic-dev | 81.3 | 95.4 | 13.1 | 5.6 | 90.7 |
| validation | 98.6 | 99.6 | 1.4 | 0.0 | 99.6 |
| OOD-dev | 3.3 | 0.0 | 13.3 | 83.3 | 0.0 |

Trade-off at margin 0.02 (suggest threshold capped at the answer threshold), percentages:

| answer ≥ | dev coverage | dev precision | dev useful | val coverage | val precision | OOD answered | OOD not_found |
|---|---|---|---|---|---|---|---|
| 0.40 | 89.7 | 94.8 | 95.3 | 98.6 | 99.6 | 43.3 | 0.0 |
| 0.45 | 89.7 | 94.8 | 95.3 | 98.6 | 99.6 | 43.3 | 3.3 |
| 0.50 | 89.7 | 94.8 | 95.3 | 98.6 | 99.6 | 43.3 | 10.0 |
| 0.55 | 89.7 | 94.8 | 94.4 | 98.6 | 99.6 | 33.3 | 36.7 |
| 0.60 | 86.9 | 95.7 | 92.5 | 98.6 | 99.6 | 13.3 | 70.0 |
| 0.65 | 81.3 | 95.4 | 90.7 | 98.6 | 99.6 | 3.3 | 83.3 |
| 0.70 | 66.4 | 95.8 | 91.6 | 98.6 | 99.6 | 3.3 | 83.3 |
| 0.75 | 52.3 | 96.4 | 92.5 | 98.2 | 99.6 | 0.0 | 83.3 |
| 0.80 | 24.3 | 100.0 | 93.5 | 96.4 | 99.8 | 0.0 | 83.3 |
| 0.85 | 5.6 | 100.0 | 93.5 | 80.2 | 99.8 | 0.0 | 83.3 |
| 0.90 | 1.9 | 100.0 | 93.5 | 23.2 | 100.0 | 0.0 | 83.3 |
| 0.95 | 0.0 | – | 93.5 | 0.0 | – | 0.0 | 83.3 |
