# Model comparison (plan section 11, Phase 4)

Clauses: 80 (60 tasks, 20 non-tasks; 40 Cyrillic). Text-layer documents only, so this measures the model and not OCR.

| Model | is_task | deadline | executors | exec. overlap | s/clause | tok/s | failed |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen3:8b` | 99% | 100% | 85% | 86% | 0.32 | 119 | 0 |
| `qwen3:8b-q8_0` | 99% | 100% | 87% | 88% | 0.48 | 78 | 0 |
| `qwen3:4b` | 96% | 90% | 85% | 86% | 0.29 | 172 | 0 |
| `qwen3:4b-q8_0` | 96% | 92% | 90% | 90% | 0.30 | 129 | 0 |
| `qwen3:4b-instruct-2507-q8_0` | 90% | 97% | 93% | 93% | 0.35 | 115 | 0 |

`deadline` and `executors` are exact matches after normalising apostrophes, case and spacing, scored only on clauses that really are tasks. `exec. overlap` is the Jaccard overlap, which gives partial credit when one executor of several is missed.
