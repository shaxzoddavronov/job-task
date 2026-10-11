# Threshold calibration

Decision sets only: validation (template paraphrases), realistic-dev (hand-written) and
OOD-dev (`data/ood_dev.txt`). Rule: highest dev coverage with precision when answering
≥ 95% on dev and validation and ≤ 5% of OOD-dev answered; then the suggest threshold with
the highest dev "useful" rate (correct answer, or gold in the top-3 suggestions) such
that ≥ 80% of OOD-dev gets not_found (a suggestion for an off-topic question is cheap;
a 90% floor would reject a third of realistic queries — see the table below).

**Chosen:** `ANSWER_THRESHOLD=0.68`, `SUGGEST_THRESHOLD=0.61`, `ANSWER_MIN_MARGIN=0.05`

| set | coverage | precision when answering | suggest | not_found | useful |
|---|---|---|---|---|---|
| realistic-dev | 58.9 | 95.2 | 24.3 | 16.8 | 78.5 |
| validation | 98.0 | 99.6 | 2.0 | 0.0 | 99.6 |
| OOD-dev | 3.3 | 0.0 | 13.3 | 83.3 | 0.0 |

Trade-off at margin 0.05 (suggest threshold capped at the answer threshold), percentages:

| answer ≥ | dev coverage | dev precision | dev useful | val coverage | val precision | OOD answered | OOD not_found |
|---|---|---|---|---|---|---|---|
| 0.40 | 81.3 | 87.4 | 84.1 | 98.0 | 99.6 | 50.0 | 0.0 |
| 0.45 | 81.3 | 87.4 | 84.1 | 98.0 | 99.6 | 43.3 | 10.0 |
| 0.50 | 81.3 | 87.4 | 84.1 | 98.0 | 99.6 | 36.7 | 36.7 |
| 0.55 | 81.3 | 87.4 | 82.2 | 98.0 | 99.6 | 26.7 | 60.0 |
| 0.60 | 77.6 | 91.6 | 76.6 | 98.0 | 99.6 | 16.7 | 76.7 |
| 0.65 | 71.0 | 92.1 | 76.6 | 98.0 | 99.6 | 13.3 | 83.3 |
| 0.70 | 52.3 | 94.6 | 78.5 | 97.6 | 99.6 | 3.3 | 83.3 |
| 0.75 | 30.8 | 93.9 | 79.4 | 95.4 | 99.6 | 3.3 | 83.3 |
| 0.80 | 15.0 | 87.5 | 79.4 | 84.6 | 99.5 | 0.0 | 83.3 |
| 0.85 | 8.4 | 77.8 | 79.4 | 61.4 | 99.4 | 0.0 | 83.3 |
| 0.90 | 3.7 | 75.0 | 80.4 | 9.5 | 97.9 | 0.0 | 83.3 |
| 0.95 | 1.9 | 50.0 | 80.4 | 6.5 | 100.0 | 0.0 | 83.3 |
