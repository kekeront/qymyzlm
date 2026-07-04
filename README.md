# QymyzLM

**Reproducible Kazakh language models.** Every number here comes from an open runner, on a pinned dataset, that you can re-run yourself — and each is labelled `measured` (we ran it) or `reported` (someone else's number, shown for context, never as our own).

![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-EE4C2C?logo=pytorch&logoColor=white)
![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue)
![Kazakh](https://img.shields.io/badge/Language-Kazakh-green)

Monorepo of the Kazakh model campaign — one uv workspace, three packages:

| Package | Dir | What it is |
|---------|-----|------------|
| `kazeval` | `evallab/` | Benchmarking lab — **the single source of truth for every number in this repo** (KazQAD retrieval/reranking, KazMMLU). See the [leaderboard](evallab/). |
| `qymyz-embed` | `embed/` | Kazakh text-embedding model — **active track**: beat off-the-shelf mE5-large on KazQAD under evallab's pinned protocol. |
| `qymyzlm` (import `kazllm`) | `src/` | Kazakh SLM engine — **planned track**: ≤600M active params, Engram n-gram memory, QLoRA continual pretraining. |

Sibling project: [kazakh-nlp-atlas](https://github.com/kekeront/kazakh-nlp-atlas) — data-driven survey of Kazakh NLP + frontier knowledge base.

```bash
uv sync --all-packages --all-extras   # one env for the whole workspace
```

---

## Reproducibility first

This is the point of the whole repo. Every published number comes from a `kazeval` runner: a **pinned dataset revision**, the **exact protocol saved in the record**, fp16 on free Kaggle GPUs. Full leaderboard → [`evallab/`](evallab/).

**First measured results (2026-07) — off-the-shelf `intfloat/multilingual-e5-large` on KazQAD:**

| Task | Metric | mE5-large (`measured`) | reference (`reported`) |
|------|--------|:----:|--------|
| KazQAD retrieval — full 825,309-passage corpus | NDCG@10 | **0.329** | 0.389 — KazQAD paper's tuned pipeline |
| KazQAD reranking — pinned hard-neg protocol | MRR@10 | **0.654** | 0.909 — kazembed-v5 card — **unverifiable** (protocol unpublished) |

The widely-cited "mE5-large MRR 0.909 on KazQAD" is **unverifiable**: the candidate pools behind it were never published, and depending on protocol choices (candidate-pool size, negatives source) the same model can score anywhere from **~0.45 to ~0.92**. A number without a reproducible protocol isn't wrong — it's uncheckable, which for a benchmark is worse. Under our open, pinned protocol (`kazqad-hardneg-bm25-v1`: 100 BM25 candidates, fixed corpus revision), off-the-shelf mE5-large scores **MRR@10 0.654**. Turning uncheckable folklore into numbers you can re-run is why this repo exists. (Full-corpus NDCG@10 0.329 sits honestly *below* the paper's tuned pipeline; off-the-shelf, untuned, reported as-is.)

---

## Track 1 — Kazakh embeddings (active)

**Goal:** the best Kazakh text-embedding model. **Bar:** beat off-the-shelf mE5-large on KazQAD under evallab's pinned protocol, then #1 on kk-MTEB. First fine-tune (Less-is-More on mE5) targets **≥ +2.0 NDCG@10** over the measured baseline above.

## Track 2 — Kazakh SLM ≤600M (planned)

**Goal:** a sub-1B-active model competitive on KazMMLU. **Levers:** QLoRA continual pretraining · Engram sparse n-gram memory (0 extra FLOPs) · Kazakh SFT · test-time scaling. Reachability on free compute is an **open question**, gated by a token-budget study before any paid run.

Generative baselines — KazMMLU 3-shot, 9,870 Kazakh questions ([MBZUAI/KazMMLU](https://huggingface.co/datasets/MBZUAI/KazMMLU)). `measured` = our runner, fp16; *italic* = reported (different harness, context only). The dev split holds only 3 exemplars/subject, so earlier "5-shot" runs were effectively **3-shot**.

| Model | Params | KazMMLU 3-shot | Tok/word |
|-------|--------|:----:|:----:|
| Qwen2.5-1.5B | 1.54B | 34.3% | 4.88 |
| **Qwen3-0.6B-Base** (primary target) | 0.6B | **32.8%** | 4.88 |
| Qwen2.5-0.5B | 0.49B | 28.8% | 4.88 |
| Gemma3-1B-it | 1.0B | 28.7% | 3.27 |
| Llama-3.2-1B | 1.24B | 25.1% | 4.80 |
| Ekitil-Qwen3-600M (Kazakh-adapted Qwen) | 0.67B | 23.7% | 1.43 |
| Random baseline | — | 25.0% | — |
| *Sherkala-Chat-8B* | *8B* | *41.4%* | *2.04* |
| *Llama-3.1-70B* | *70B* | *55.2%* | *4.73* |

**Takeaways:** Qwen dominates at small scale (Qwen3-0.6B beats every other sub-2B model). Ekitil — a Kazakh-adapted Qwen with an elite tokenizer (1.43 tok/word) — still scores below random on knowledge: **a good tokenizer is not knowledge**. The gap to Sherkala-8B is cross-protocol until Sherkala is re-run through our runner.

---

## Roadmap

- [x] KazMMLU generative baselines (8 models, `measured`)
- [x] **First reproducible KazQAD embedding numbers (mE5-large) — 0.909 folklore refuted**
- [ ] KazQAD leaderboard v0 — BGE-M3, Qwen3-Embedding, LaBSE, MiniLM
- [ ] First `qymyz-embed` fine-tune (Less-is-More) — beat the mE5-large baseline
- [ ] Generative track: token-budget go/no-go → QLoRA continual PT

## References

- [KazQAD](https://arxiv.org/abs/2404.04487) — Kazakh open-domain QA + retrieval (LREC-COLING 2024)
- [KazMMLU](https://arxiv.org/abs/2502.12829) — Kazakh MMLU, ACL 2025 (23K questions)
- [Sherkala](https://arxiv.org/abs/2503.01493) — 8B Kazakh continual PT (ISSAI)
- [SozKZ](https://arxiv.org/abs/2603.20854) — 600M Kazakh from scratch
- [Engram](https://arxiv.org/abs/2601.07372) — conditional N-gram memory (DeepSeek-AI)

## License

Apache 2.0
