# Uzbek document analysis

Upload a PDF, press **Analyze**, and get back the document's details and every
task it assigns — with the deadline and who is responsible.

Works with Uzbek documents in **Latin or Cyrillic**, **typed or scanned**.
Everything runs on your own machine; no document is ever sent anywhere.

---

## Quick start

**Linux / macOS**

```bash
git clone https://github.com/shaxzoddavronov/job-task.git
cd job-task/entity-extraction
./start.sh
```

**Windows** (PowerShell — note the `.\`, PowerShell does not run programs from
the current directory without it)

```powershell
git clone https://github.com/shaxzoddavronov/job-task.git
cd job-task\entity-extraction
.\start.bat
```

In the older `cmd.exe` prompt, plain `start.bat` works too.

Then open <http://localhost:8000> and upload a PDF.

The first start builds the image and downloads the language model (2.5–9 GB
depending on your hardware), so expect 10–20 minutes. Later starts reuse both
and are ready in seconds.

To see when it is ready, and which model it chose for your hardware:

```bash
curl http://localhost:8000/health
# {"status":"ready","model":"qwen3:4b","tier":"T1","reason":"detected 5100 MB free VRAM"}
```

Until that says `ready`, uploads are refused with a message rather than queued
behind the download.

### What you need

| | |
|---|---|
| Everywhere | Docker (with Compose), ~8 GB free disk |
| Windows | Docker Desktop with the **WSL2 backend** |
| GPU (optional) | An NVIDIA card and a recent driver |

No GPU is fine — the app detects that and runs on CPU, just slower. **You never
have to configure anything**: the launcher measures your hardware and the app
picks a model to match.

If you have an NVIDIA card, this should print your GPU. If it does not, the app
will still run, on CPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### Test documents

No sample PDFs are included — use your own Uzbek documents, or generate a set:

```bash
pip install -r requirements-dev.txt
python eval/make_synthetic.py      # 40 PDFs in eval/data/synthetic/
python eval/fetch_lexuz.py         # real documents from lex.uz
```

### Stopping it

```bash
docker compose down
```

---

## What you get

The page shows two things.

**Document details** — issuing organisation, document number, date, signatory.

**Tasks** — one row each, with the deadline as written *and* resolved to a
date where possible, plus the responsible organisations.

```json
{
  "details": {
    "document_number": { "value": "PF-213", "method": "regex" },
    "document_date":   { "value": "2026-10-03", "raw": "03.10.2026", "method": "regex" },
    "organization":    { "value": "Oʻzbekiston Respublikasi Prezidenti", "method": "llm" },
    "signatory":       { "name": "Sh. MIRZIYOYEV",
                         "position": "Oʻzbekiston Respublikasi Prezidenti" }
  },
  "tasks": [
    {
      "task": "Taʼlim va ilm-fan vazirligi ... takliflar kiritsin.",
      "deadline_raw": "ikki oy muddatda",
      "deadline_date": "2026-12-03",
      "deadline_type": "relative",
      "executors": ["Taʼlim va ilm-fan vazirligi", "manfaatdor vazirlik va idoralar"],
      "page": 1
    }
  ]
}
```

Everything shown is copied from the document, in the document's own script.
**A value that isn't in the document comes back empty, never invented.**

---

## Privacy

Your documents never leave the machine.

- The model runs locally, in a container on your own hardware.
- The only network traffic is **downloading the model and OCR weights**, and
  that happens at build and startup — never while a document is being analysed
  (the image sets `HF_HUB_OFFLINE=1` to guarantee it).
- Nothing is stored. The uploaded file is deleted as soon as the response is
  sent.

---

## How it works

```
PDF
 └─> 1. Text extraction        Docling + Tesseract OCR for scanned pages
 └─> 2. Script detection       Latin or Cyrillic
 └─> 3. Transliteration        Cyrillic -> Latin, keeping an offset map
 └─> 4. Regex                  document number + date
 └─> 5. LLM call A             organisation + signatory (only what regex missed)
 └─> 6. Chunking               clauses, table rows, paragraphs
 └─> 7. LLM call B             per chunk, in parallel: task? deadline? who?
 └─> 8. Grounding              check each value is really in the text
 └─> 9. Deadline parsing       "ikki oy muddatda" -> 2026-12-03
 └─> JSON -> web page
```

### Why regex *and* an LLM

They are good at different things.

- **Numbers and dates are patterns.** Regex gets them right **100% of the
  time** on typed documents, instantly and for free. Asking a model would be
  slower and occasionally wrong.
- **Everything else is language.** Who is responsible, what counts as a task,
  what the deadline refers to — these need reading comprehension.

So regex goes first and the model is asked only for the fields regex didn't
find. That is fewer tokens, lower latency, and fewer chances to hallucinate.

### Why transliteration

The pipeline works in one script internally, so one set of rules handles both.
Every character keeps an **offset map** back to the original, so what you see
is the document's own text — `Молия вазирлиги`, not `Moliya vazirligi`.

### Grounding

The model is told to copy values exactly, but models paraphrase. Before any
value is shown, it is searched for in the chunk it came from. **If it isn't
there, it is dropped** — so the page shows a missing value rather than a
confident wrong one. Values that only partly matched are flagged *tekshiring*
("check this").

---

## Model selection

The launcher reads your free VRAM and the app picks a tier. No configuration.

| Tier | Free VRAM | Model | Size |
|---|---|---|---|
| T0 | no GPU | `qwen3:4b` Q4_K_M | 2.5 GB |
| T1 | ≥ 4 GB | `qwen3:4b` Q4_K_M | 2.5 GB |
| T2 | ≥ 10 GB | `qwen3:8b` Q4_K_M | 5.2 GB |
| T3 | ≥ 22 GB | `qwen3:8b-q8_0` | 8.9 GB |

If a model doesn't actually fit in VRAM, the app notices and steps down a tier
rather than crawling.

**Tested:** T0, T1 and T3 end to end on a Linux server with NVIDIA L40S cards.
**Not tested:** T2, and T1 under real 6 GB memory pressure — the tier-selection
and step-down logic is covered by tests, but not by a 6 GB card.

These models were chosen by measurement, not assumption — see
[`eval/results/model_comparison.md`](eval/results/model_comparison.md). Two
things that surprised us:

- **Bigger was worse.** `qwen3:30b-a3b` (18 GB) scored *below* the 4B model on
  two of three metrics. T3 spends its spare VRAM on precision, not parameters.
- **Uzbek-tuned models did badly.** `uzbek-llama-3.1-8B-Uz` and an `alloma-8b`
  quant scored 58% and 47% on deadlines against 90–100% for Qwen3. They are
  tuned for Uzbek conversation, not for structured extraction.
  (`uzlm/alloma-3B/8B-Instruct` have no GGUF at all and cannot be loaded.)

---

## How well does it work

Measured on 52 labelled documents (156 tasks) with `qwen3:8b` —
[full report](eval/results/eval.md).

**Document details**

| Field | Correct |
|---|---:|
| Document date | 92% |
| Document number | 81% |
| Signatory name | 79% |
| Organisation | 77% (100% allowing an extra word) |
| Signatory position | 73% |

**Tasks**

| | Precision | Recall | Deadline | Executors |
|---|---:|---:|---:|---:|
| **Overall** | 84% | 84% | 95% | 82% |
| Typed documents | 83% | 87% | 99% | 82% |
| Scanned documents | 85% | 81% | 92% | 81% |
| Latin | 84% | 83% | 100% | 85% |
| Cyrillic | 84% | 85% | 91% | 79% |

### The test set

| Source | Documents | Tasks | What they are |
|---|---:|---:|---|
| Synthetic | 40 | 128 | Generated: 4 layouts × Latin/Cyrillic × typed/scanned |
| lex.uz | 12 | 12 | Real Uzbek presidential decrees and resolutions |
| lex.uz (unlabelled) | 8 | — | Long multi-page acts, used for robustness only |

**The employer's own documents were not available**, so none of these numbers
come from them. The synthetic documents were generated by us and share the
generator's habits, which flatters them (99% precision). **The lex.uz figures
are the realistic ones.**

On lex.uz, precision is 60% — but roughly two thirds of those "errors" are a
labelling disagreement, not a mistake: where an act assigns a bundle of duties
and then lists them (a), (b), (c), our labels call that one task and the system
returns each item. Counting those as correct puts lex.uz precision near 85%. We
left the labels alone rather than rewrite them to match the output.

---

## Speed

| | Extraction (OCR) | Model | Total |
|---|---:|---:|---:|
| GPU, 1-page scan | ~10 s | ~1 s | **~11 s** |
| CPU, 1-page scan | ~8 s | ~13 s | **~20 s** |
| CPU, 2-page typed | ~8 s | ~40 s | **~48 s** |

A typed page extracts in well under a second — the ~10 s above is Tesseract
doing OCR on a scan. On GPU, OCR dominates; on CPU, the model does.

The pipeline applies the usual inference optimizations (constrained decoding, a
cached prompt prefix, small context, capped output, parallel calls, quantized
KV cache). Measured honestly, they are worth **1.2× on CPU and nothing on GPU** —
[the benchmark](eval/results/bench.md) explains why. Their real value is
accuracy: processing a document clause by clause is what makes the numbers
above possible at all.

Two further options exist and were measured:

- **`CLAUSE_BATCH_SIZE=4`** packs several clauses into one call, roughly halving
  the model stage. It is **off by default** because it costs accuracy on real
  documents — executors fall from 87% to 83% overall, and from 64% to 51% on
  lex.uz, as the model loses track of which answer belongs to which clause.
- Raising `OLLAMA_NUM_PARALLEL` above 4 does **not** help; Ollama splits the KV
  cache across parallel slots.

---

## Limitations

- **Never tested on the employer's real documents.** Every number comes from
  synthetic and public documents.
- **Scanned quality is a ceiling.** OCR misreads `/` as `1` and `№` as `Ne`
  or `М`. The regexes tolerate the sign confusion, but a corrupted digit cannot
  be recovered — the model reads the same corrupted text.
- **Executors are the weakest field (82%).** The model sometimes names an
  organisation that is the *object* of a clause rather than the doer, and
  sometimes drops the second of two joint executors.
- **Only NVIDIA GPUs are accelerated.** AMD and Apple silicon fall back to CPU.
- **PDF only.**

### Next steps

1. Evaluate on real documents and label a larger set — 156 tasks is small.
2. Improve executor extraction; it is the clear weak point.
3. Pre-process scans (deskew, denoise) before OCR.
4. For a busy server, vLLM would give better throughput than Ollama.

---

## Development

Run one stage directly:

```bash
pip install -r requirements-dev.txt
python -m app.extract_text document.pdf     # what the extractor sees
python -m app.pipeline document.pdf         # the full JSON
```

Regenerate the test set and reproduce every number in this README:

```bash
python eval/make_synthetic.py     # 40 synthetic PDFs + their labels
python eval/fetch_lexuz.py        # public documents from lex.uz
python eval/run_eval.py           # accuracy     -> eval/results/eval.md
python eval/bench.py              # speed        -> eval/results/bench.md
python eval/compare_models.py     # model choice -> eval/results/model_comparison.md
```

The test set is generated rather than committed, so the first two commands must
run before the others.

### Layout

```
app/            extract_text, script_utils, regex_rules, chunker,
                llm, prompts, grounding, deadlines, model_select, pipeline, main
eval/           test-set generators, evaluation and benchmark scripts
eval/results/   the reports every figure above comes from
```
