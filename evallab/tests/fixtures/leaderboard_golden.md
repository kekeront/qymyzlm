_Auto-generated from `evallab/results/*.json` by `python -m kazeval.leaderboard` — do not edit by hand._

### At a glance

_One comparable metric per track, best first. `measured` = a kazeval runner produced it here; `reported` = external number, shown for context until re-measured in-lab. Full per-record metrics below._

**Embedding / retrieval (KazQAD)**

| Model | Task | nDCG@10 | Provenance | Date |
|---|---|---|---|---|
| intfloat/multilingual-e5-base | KazQADRetrieval | 0.4120 | measured | 2026-07-01 |
| intfloat/multilingual-e5-large | KazQAD-HardNeg | — | reported | 2025-12-04 |

**Generative (KazMMLU)**

| Model | Task | Accuracy | Provenance | Date |
|---|---|---|---|---|
| Qwen/Qwen3-0.6B-Base | KazMMLU-kk | 0.3280 | measured | 2026-04-30 |

### Full records

### Embedding / retrieval (KazQAD)

| Model | Revision | Task | Split | ndcg_at_10 | mrr | hits_at_1 | hits_at_5 | mrr_at_10 | recall_at_100 | Provenance | Source | Date |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| intfloat/multilingual-e5-base | d1287505 | KazQADRetrieval | test | 0.4120 | — | — | — | 0.4012 | 0.8000 | measured | kazeval.run_retrieval (mteb 2.16.3) | 2026-07-01 |
| intfloat/multilingual-e5-large | — | KazQAD-HardNeg | test | — | 0.9090 | 0.8500 | 0.9900 | — | — | reported | https://huggingface.co/Nurlykhan/kazembed-v5 | 2025-12-04 |

### Generative (KazMMLU)

| Model | Revision | Task | Split | acc | Provenance | Source | Date |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B-Base | — | KazMMLU-kk | test | 0.3280 | measured | qymyzlm README baseline study | 2026-04-30 |

### Safety (Qorgau)

_No results yet._

### Other

| Model | Revision | Task | Split | score | Provenance | Source | Date |
|---|---|---|---|---|---|---|---|
| demo/model | — | DemoTask | test | 0.5000 | measured | fixture | 2026-07-02 |
