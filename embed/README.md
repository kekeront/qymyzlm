# QymyzEmbed — Kazakh text embeddings

v0: fine-tune of `intfloat/multilingual-e5-base` (278M, dim 768) per the **Less-is-More**
protocol — *"Less is More: Adapting Text Embeddings for Low-Resource Languages with Small
Scale Noisy Synthetic Data"*, [arXiv:2603.22290](https://arxiv.org/abs/2603.22290).
Fits a single RTX 2070 (8 GB) / free Colab.

**Target**: MRR > 0.909 on KazQAD hard negatives (off-the-shelf mE5-large reference from the
`Nurlykhan/kazembed-v5` card). Caveat: that number is highly protocol-sensitive — the
99-negative TF-IDF pools were never published, so evallab fixes and publishes its own
hard-negative recipe. **Every number comes from `evallab/` runners** (`kazeval` package);
nothing in `embed/` self-reports metrics.

## Protocol (paper defaults, all pinned in code)

| Knob | Value | Where |
|---|---|---|
| pairs | ~10,000 (cap, seeded shuffle) | `train_lim.py --max-pairs` |
| fine-tune | FULL (no PEFT) | `train_lim.py` |
| loss | CachedMNRL, scale 20 (= temp 0.05); GradCache `mini_batch_size=8` for 8 GB | `train_lim.build_loss` |
| effective batch | 512 = `per_device_train_batch_size` (grad-accum does NOT grow the in-batch-negative pool) | `train_lim.build_args` |
| lr / epochs / schedule | 7e-5 / 5 / linear, warmup ratio 0.2 (`warmup_steps=0.2` float) | `train_lim.build_args` |
| precision | fp16 (Turing: no bf16) | `train_lim.build_args` |
| filtering | drop if semantic drift > 0.05; keep only translation sim > 0.85 (mE5-measured) | `data/synth_filter.py` |
| model soup | 0.5/0.5 weight average with the BASE model | `merge.py` |
| e5 prefixes | `"query: "` / `"passage: "` everywhere (mE5 ships NO prompts in its hub config — ST 5.6 silently applies empty ones) | `prefixes.py` (single source of truth) |

## Pipeline

```bash
PY=.venv/bin/python  # run from the repo root; PYTHONPATH=embed/src until the editable install lands

# 1. Contrastive pairs from KazParC (gated:auto — click "Agree and access" on the HF page first)
$PY -m qymyz_embed.data.kazparc_pairs --output pairs.jsonl --directions kk-ru,ru-kk,kk-en,en-kk
# smoke run: --limit 1000; offline/tests: --input embed/tests/fixtures/kazparc_sample.jsonl

# 2. Less-is-More filtering (precomputed sims, or --model to compute with mE5)
$PY -m qymyz_embed.data.synth_filter --input scored.jsonl --output kept.jsonl

# 3. (optional) hard negatives — python API: qymyz_embed.data.hard_negatives.mine(...)

# 4. Fine-tune (checkpoints -> embed/checkpoints/, gitignored)
$PY -m qymyz_embed.train_lim --data kept.jsonl

# 5. Soup with the base (the paper's 0.5/0.5 merge)
$PY -m qymyz_embed.merge embed/checkpoints/lim-*/final intfloat/multilingual-e5-base \
    --output embed/checkpoints/lim-souped

# 6. Evaluate — forwards to evallab's kazeval runners; the ONLY source of numbers
$PY -m qymyz_embed.evaluate --help
```

## Open research question: Cyrillic script overlap (paper §6.2)

Less-is-More reports its gains on low-resource languages whose relation to the base model's
high-resource languages varies; §6.2 leaves open how much **script/lexical overlap with a
high-resource language** drives transfer. Kazakh is Cyrillic and shares the script (plus heavy
loan vocabulary) with Russian — one of mE5's strongest languages. Our kk↔ru vs kk↔en pair
ablation (same recipe, different pair direction mix via `--directions`) is designed to answer
whether the protocol's gains ride on script overlap or on semantic supervision.

## Data & licenses

- `issai/kazparc` @ `41df65bd` — gated:auto; no license tag on HF (GitHub badge claims
  CC BY 4.0, [UNVERIFIED on HF]). Config `kazparc_raw`: 371,902 rows, columns
  `id/kk/en/ru/tr/domain`.
- KazQAD repos are CC BY-SA 4.0 (share-alike propagates to derived corpora).
- Qorgau has NO license — eval-only, never train on it (embed/CLAUDE.md).

## Tests

Fully offline (fixtures, no downloads):

```bash
.venv/bin/ruff check embed && .venv/bin/python -m pytest embed/tests -q
```
