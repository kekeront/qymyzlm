# evallab/ — benchmarking lab (single source of truth for ALL numbers)

## Goal
No model claim exists until a runner here reproduces it. Tracks:
1. **Embedding/retrieval** — KazQAD (planka: mE5-large MRR 0.909 on hard negatives),
   kk-MTEB task classes (KazQADRetrieval, KazQADReranking) written here first, then PR'd
   upstream to embeddings-benchmark/mteb — the upstream merge is the deliverable.
2. **Generative** — KazMMLU 3-shot (dev split holds only 3 exemplars/subject; runs previously
   labeled 5-shot were effectively 3-shot). Baseline Qwen3-0.6B 32.8% (measured), ceiling
   Sherkala-8B 41.4% (reported, cross-protocol until re-run here).
3. **Safety** — Qorgau kk-ru guardrails (arXiv 2502.13640; NO license — eval-only,
   never train on its annotations).

## Package
`kazeval`, import `kazeval`, workspace member of the qymyzlm monorepo.

## Hard constraints
- Every leaderboard row regenerable by one command; no hand-edited numbers.
- Results = committed JSON per run in `evallab/results/` (model, revision, task, numbers, date);
  README leaderboard auto-rendered from them — never edited by hand.
- Tests run offline on fixtures in `evallab/tests/fixtures/`; real dataset pulls only behind
  explicit CLI commands (documented per runner).

## Done means
- `ruff check` clean; `pytest evallab/tests` green offline; mteb tasks instantiate;
  leaderboard renders from sample results JSON.
