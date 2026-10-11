# BGE-M3 reference benchmark (benchmark only, not used in the delivered system)

Generated 2026-10-11T09:58:16+00:00 · model `BAAI/bge-m3` (2.27 GB fp32 weights, run in fp16 on GPU) · GPU: NVIDIA L40S · CPU: x86_64

Same split, query sets, aggregation and metrics as `make eval`. Accuracy in %, latency per query in ms. Status-based numbers use our thresholds, which were not calibrated for BGE-M3 cosines.

| System | Test Top-1 | Test Top-3 | Test MRR@10 | Entity Top-1 | Realistic-dev Top-1 | Realistic Top-1 | Realistic Top-3 | Unknown law → not_found | Latency p50 / p95 | Runs on |
|---|---|---|---|---|---|---|---|---|---|---|
| ours: phase 5 (TF-IDF + entities + re-ranker, no embeddings) | 98.8 | 99.8 | 99.4 | 96.4 | 76.6 | 63.6 | 77.6 | 100.0 | 3.6 / 4.5 | CPU |
| ours: final engine (+ Model2Vec) | 99.6 | 100.0 | 99.8 | 100.0 | 82.2 | 67.3 | 82.2 | 100.0 | 5.2 / 6.4 | CPU |
| BGE-M3 dense (benchmark only) | 99.6 | 100.0 | 99.8 | 100.0 | 85.0 | 80.4 | 92.5 | 0.0 | 15.5 / 18.9 | CUDA |
| BGE-M3 + our engine, hybrid (benchmark only) | 99.4 | 100.0 | 99.7 | 100.0 | 85.0 | 72.0 | 88.8 | 100.0 | 37.5 / 50.9 | CUDA |

BGE-M3 dense on CPU (50 queries): p50 181 ms, p95 187 ms per query.
