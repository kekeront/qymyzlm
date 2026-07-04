# evallab — Kazakh benchmarking lab (`kazeval`)

Single source of truth for ALL lab numbers: no model claim exists until a runner
here reproduces it. Every leaderboard row is regenerable from a committed JSON
record in `results/` — nothing below the markers is ever edited by hand.

## Tracks

| Track | Tasks | Runner |
|---|---|---|
| Embedding / retrieval | `KazQADRetrieval` (full 825,309-passage corpus, comparable to arXiv:2404.04487), `KazQADReranking` + `KazQAD-HardNeg` (deterministic `kazqad-hardneg-bm25-v1` protocol, `kazeval.hardneg`) | `python -m kazeval.run_retrieval` |
| Generative | `KazMMLU-kk` — Kazakh subset of MBZUAI/KazMMLU, 12 subjects, 9,870 test questions, **3-shot** (dev holds only 3 exemplars/subject; historical "5-shot" numbers were effectively 3-shot) | `python -m kazeval.run_kazmmlu` |
| Safety | `Qorgau-kk` (arXiv:2502.13640) — **no license declared upstream ⇒ eval-only, never train on it**; loaders in `kazeval.qorgau`, judge protocol not wired yet | — |

## Usage

```bash
# kk-MTEB retrieval + reranking (gated dataset: accept conditions on
# https://huggingface.co/datasets/issai/kazqad-retrieval first)
python -m kazeval.run_retrieval --model intfloat/multilingual-e5-large

# KazMMLU Kazakh subset, 3-shot (the dev-split maximum)
python -m kazeval.run_kazmmlu --model Qwen/Qwen3-0.6B-Base

# re-render the leaderboard below from results/*.json
python -m kazeval.leaderboard
# CI-style staleness check (exit 2 if the section is out of date)
python -m kazeval.leaderboard --check
```

**GPU runs happen on Kaggle** (compute policy 2026-07-04: free plan, T4×2/P100 16 GB fp16,
30 GPU-h/week): upload `kaggle/kazeval_kaggle.ipynb`, add `HF_TOKEN` to Kaggle Secrets,
push `main` first (the notebook clones this repo from GitHub). Download the produced
JSON records into `results/` and re-render the leaderboard locally.

Both runners write one validated record per (task, split) into `results/`
(`provenance: measured`); externally sourced numbers are committed as
`provenance: reported` and stay flagged until re-measured in-lab.

Tests are fully offline (`fixtures/` only): `pytest evallab/tests -q`.

## Leaderboard

<!-- LEADERBOARD:START -->
_Auto-generated from `evallab/results/*.json` by `python -m kazeval.leaderboard` — do not edit by hand._

### Embedding / retrieval (KazQAD)

| Model | Revision | Task | Split | ndcg_at_10 | mrr | hits_at_1 | hits_at_5 | Provenance | Source | Date |
|---|---|---|---|---|---|---|---|---|---|---|
| BM25 + fine-tuned reranker (KazQAD paper best pipeline) | — | KazQADRetrieval | test | 0.3890 | 0.3820 | — | — | reported | arXiv:2404.04487 (KazQAD, LREC-COLING 2024) | 2024-04-06 |
| intfloat/multilingual-e5-large | — | KazQAD-HardNeg | test | — | 0.9090 | 0.8500 | 0.9900 | reported | https://huggingface.co/Nurlykhan/kazembed-v5 | 2025-12-04 |

### Generative (KazMMLU)

| Model | Revision | Task | Split | acc | Provenance | Source | Date |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B-Base | — | KazMMLU-kk | test | 0.3280 | measured | qymyzlm README baseline study (scripts/benchmark_baselines.py) | 2026-04-30 |

### Safety (Qorgau)

_No results yet._
<!-- LEADERBOARD:END -->
