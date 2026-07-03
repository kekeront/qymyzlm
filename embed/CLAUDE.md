# embed/ — QymyzEmbed: best Kazakh text-embedding model

## Goal
Beat mE5-large MRR 0.909 on KazQAD hard-negatives; #1 on kk-MTEB. Two phases:
1. **v0 (now)**: fine-tune mE5-base per Less-is-More protocol (arXiv 2603.22290) — ~10k clean pairs,
   0.5/0.5 weight averaging with the base model, fits RTX 2070/Colab. Open question we answer:
   Cyrillic script overlap (paper §6.2).
2. **v1 (later)**: embedding head on our own QymyzLM SLM (LLM2Vec post-hoc vs GRIT joint —
   decision goes through /design-panel, not ad-hoc).

## Package
`qymyz-embed`, import `qymyz_embed`, workspace member of the qymyzlm monorepo.

## Hard constraints
- Every eval number comes from `evallab/` runners — never self-reported ad-hoc.
- Training data licenses checked before use (Qorgau: no license — do NOT train on it).
- Free compute only (RTX 2070 / Colab / Kaggle) — locked course 2026-07-03.
- e5 models require "query: " / "passage: " prefixes — keep them consistent in train AND eval.

## Done means
- HF model card with reproducible numbers; MRR > 0.909 on KazQAD; kk-MTEB scores submitted.
- `ruff check` clean; `pytest embed/tests` green offline (fixtures, no downloads).
