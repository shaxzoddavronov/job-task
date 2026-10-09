# Inference benchmark (plan sections 9 and 11)

Model `qwen3:4b`. Extraction is cached and excluded: this measures the LLM stage only, which is the part the optimizations touch.

`Raw tasks` is what each configuration returned before grounding and deadline parsing. It is here to show the two are doing comparable work, not as an accuracy measure — accuracy is in `eval/results/eval.md`.

**baseline** — one call with the whole document, sequential, no output cap, `num_ctx 8192`, server with flash attention off and an f16 KV cache.

**optimized** — per-chunk calls fanned out 4 at a time, byte-stable prompt prefix, `num_ctx 4096`, capped output, constrained decoding, server with flash attention on and a q8_0 KV cache.

### GPU

| Document | Config | Wall s | LLM calls | Prompt tok | Output tok | Output tok/s | Raw tasks |
|---|---|---:|---:|---:|---:|---:|---:|
| `doc_01_lat.pdf` | baseline | 0.83 | 1 | 294 | 128 | 167.0 | 2 |
| `doc_01_lat.pdf` | optimized | 1.0 | 1 | 752 | 122 | 162.1 | 2 |
| `doc_03_cyr.pdf` | baseline | 2.14 | 1 | 471 | 279 | 169.0 | 4 |
| `doc_03_cyr.pdf` | optimized | 1.62 | 6 | 5745 | 155 | 150.4 | 1 |
| `doc_02_cyr_scan.pdf` | baseline | 1.56 | 1 | 265 | 181 | 162.6 | 3 |
| `doc_02_cyr_scan.pdf` | optimized | 0.7 | 2 | 1726 | 45 | 142.6 | 1 |
| `lexuz_8519155_lat.pdf` | baseline | 3.1 | 1 | 1021 | 436 | 169.1 | 3 |
| `lexuz_8519155_lat.pdf` | optimized | 4.14 | 14 | 14690 | 488 | 163.7 | 4 |

**Totals** — baseline 7.63s in 4 calls producing 1024 output tokens; optimized 7.46s in 23 calls producing 810 output tokens. **1.0x faster**, 1.3x fewer output tokens.

### Techniques measured individually

One workload: the clause chunks of `lexuz_8519155_lat.pdf`, run sequentially unless the row says otherwise.

| Technique | Without | With | Gain |
|---|---|---|---:|
| Concurrency (4 vs 1) | 2.63s | 2.23s | 1.2x |
| Prefix caching (stable vs unique prefix) | 3.88s (17259 prompt tok/s) | 2.62s (28050 prompt tok/s) | 1.5x |
| Flash attention + q8_0 KV (server) | 2.69s | 2.66s | 1.0x |
| Context 4096 vs 8192 | 2.64s | 2.63s | 1.0x |

### CPU (`options.num_gpu = 0`)

| Document | Config | Wall s | LLM calls | Prompt tok | Output tok | Output tok/s | Raw tasks |
|---|---|---:|---:|---:|---:|---:|---:|
| `doc_01_lat.pdf` | baseline | 5.35 | 1 | 294 | 128 | 24.3 | 2 |
| `doc_01_lat.pdf` | optimized | 7.36 | 1 | 752 | 120 | 22.5 | 2 |
| `doc_03_cyr.pdf` | baseline | 13.75 | 1 | 471 | 279 | 22.5 | 4 |
| `doc_03_cyr.pdf` | optimized | 10.05 | 6 | 5745 | 184 | 24.3 | 3 |
| `doc_02_cyr_scan.pdf` | baseline | 8.4 | 1 | 265 | 179 | 23.5 | 3 |
| `doc_02_cyr_scan.pdf` | optimized | 2.46 | 2 | 1726 | 44 | 25.0 | 1 |
| `lexuz_8519155_lat.pdf` | baseline | 27.0 | 1 | 1021 | 436 | 19.0 | 3 |
| `lexuz_8519155_lat.pdf` | optimized | 24.07 | 14 | 14690 | 488 | 25.7 | 4 |

**Totals** — baseline 54.5s in 4 calls producing 1022 output tokens; optimized 43.94s in 23 calls producing 836 output tokens. **1.2x faster**, 1.2x fewer output tokens.


## What these numbers mean

**On GPU the optimized pipeline is not faster than the naive one** (7.46 s
against 7.63 s), and that is the honest headline. Chunking multiplies prompt
tokens — 14 690 against 1 021 on a two-page act — and prefix caching claws most
of that back rather than winning outright. GPUs process prompt tokens so fast
(28 000/s here) that the extra input costs little, but it does not pay either.

**On CPU the optimized pipeline is 1.2x faster** (43.9 s against 54.5 s),
because a CPU generates at 20–25 tokens/s and the optimizations produce 1.2x
fewer output tokens. Where generation dominates, cutting output wins.

The optimizations earn their place mainly on **accuracy**, not speed. Chunking
is what makes 84% precision and 84% recall possible (`eval/results/eval.md`);
asking one question about a whole document returns a plausible-looking list
that is much harder to ground, and nothing in it can be traced to a span.

### Two measurement traps, both of which flattered the optimizations

1. **Cold model load inside the timed window.** The first baseline document
   measured 6.72 s instead of 0.83 s, which made "optimized" look 1.8x faster
   than it is.
2. **Ollama reloads the model when `num_ctx` or the server changes.** Every
   "without" row was being charged a reload, which made flash attention and the
   quantized KV cache look **4.8x** faster. With a warm-up per configuration the
   measured gain is **1.0x**.

Both are now handled by warming each configuration before timing it. The first
set of numbers this benchmark produced was wrong in the flattering direction,
which is the direction to distrust.

### Flash attention and KV quantization: a memory argument, not a speed one

They buy nothing measurable at this workload — short prompts, few output
tokens, low concurrency. `OLLAMA_KV_CACHE_TYPE=q8_0` halves the KV cache, and
that is why it is kept: on the 6 GB card of tier T1 it is the difference between
a model plus four parallel slots fitting in VRAM and spilling to system memory,
where everything would be ~20x slower. The benchmark machine has 45 GB free and
cannot show this; it would appear on the 6 GB card as a tier step-down.

### Per-document cost in practice

Re-measured with `qwen3:4b` after the Docling converter was made reusable —
it had been rebuilt per document, which cost about 1.8 s every time and, on the
first document of a run, closer to 8 s:

| | | Extraction | LLM | Total |
|---|---|---:|---:|---:|
| **GPU** | 1-page typed | 0.6 s | 1.4 s | 2.0 s |
| | 1-page scan | 4.2 s | 1.4 s | 5.6 s |
| | 2-page typed act | 1.3 s | 4.0 s | 5.3 s |
| **CPU** | 1-page typed | 0.6 s | 8.2 s | 8.9 s |
| | 1-page scan | 3.5 s | 7.9 s | 11.4 s |
| | 2-page typed act | 1.2 s | 34.7 s | 35.9 s |

A typed page extracts in well under a second; the seconds on a scan are
Tesseract. The first document after startup costs about 8 s more, while the
layout model loads.

Caching the converter was the single largest latency win in the project outside
chunking itself, and it was a plain oversight rather than a trade-off — worth
more than every technique in the table above combined.
