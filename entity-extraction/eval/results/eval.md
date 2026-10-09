# End-to-end evaluation (plan section 11, Phase 6)

Model: `qwen3:8b-q8_0`. Documents: 52. Labelled tasks: 156.

## Details

| Field | Exact | Fuzzy | Method |
|---|---:|---:|---|
| organization | 77% | 100% | llm |
| document_number | 81% | 83% | regex, llm fallback |
| document_date | 92% | 92% | regex, llm fallback |
| signatory_name | 81% | 98% | llm |
| signatory_position | 71% | 87% | llm |

`Exact` is equality after normalising case, spacing and apostrophes; `fuzzy` accepts a partial match at 90, which separates a near-miss (an organisation name with an extra word) from a wrong answer.

## Tasks

| Slice | Docs | Precision | Recall | Deadline date | Deadline raw | Executors |
|---|---:|---:|---:|---:|---:|---:|
| **all** | 52 | 82% | 86% | 96% | 95% | 83% |
| lexuz | 12 | 58% | 81% | 100% | 90% | 51% |
| synthetic | 40 | 99% | 88% | 94% | 97% | 96% |
| latin | 26 | 83% | 86% | 100% | 97% | 82% |
| cyrillic | 26 | 82% | 86% | 91% | 93% | 84% |
| text layer | 26 | 81% | 90% | 99% | 97% | 79% |
| scanned | 26 | 83% | 82% | 92% | 92% | 88% |
| latin / text | 13 | 81% | 90% | 100% | 97% | 77% |
| latin / scan | 13 | 84% | 82% | 100% | 97% | 88% |
| cyrillic / text | 13 | 81% | 90% | 97% | 97% | 80% |
| cyrillic / scan | 13 | 82% | 82% | 84% | 88% | 88% |

A predicted task matches a labelled one at `partial_ratio >= 80`. Deadline and executor accuracy are scored on matched pairs only.

Total pipeline time 158s for 52 documents (3.0s each), extraction cached.

## How to read these numbers

**lex.uz precision is depressed by a labelling convention, not only by model error.** Act PQ-358 clause 2 assigns a bundle of functions and then enumerates them as sub-items (a)-(d). The label treats the parent as one task; the pipeline returns each sub-item. Both readings are defensible, and the label's reading charges four false positives per variant — about two thirds of all lex.uz false positives. Counting those as correct would put lex.uz precision near 85%. The labels were left alone rather than rewritten to match the model.

**`Exact` understates the details.** `organization` is 77% exact but 100% fuzzy: the model returns the right body with an extra word ("... PREZIDENTINING FARMONI" for "... Prezidenti"). These are near-misses, not wrong answers.

**Scanned pages have an OCR ceiling.** Residual deadline and number errors on scans are character corruption — `/` read as `1`, a day digit read as a letter — which the LLM fallback cannot recover either, since it reads the same corrupted text. See `eval/results/regex_accuracy.md`.

**Remaining genuine model weaknesses**, both on executors: on a passive clause the model names the organisation being transferred or renamed as the executor, and it drops the second of two joint executors ("X ... Y bilan birgalikda").

## Worst documents by recall

| Document | Labelled | Predicted | Matched |
|---|---:|---:|---:|
| doc_09_lat_scan | 2 | 0 | 0 |
| doc_02_cyr_scan | 3 | 1 | 1 |
| doc_02_cyr | 3 | 2 | 2 |
| doc_02_lat_scan | 3 | 2 | 2 |
| doc_06_cyr | 3 | 2 | 2 |
| doc_06_cyr_scan | 3 | 2 | 2 |
| doc_06_lat | 3 | 2 | 2 |
| doc_06_lat_scan | 3 | 2 | 2 |
| lexuz_8517348_cyr_scan | 7 | 10 | 5 |
| doc_10_cyr_scan | 4 | 3 | 3 |
