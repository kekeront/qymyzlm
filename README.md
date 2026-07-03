# QymyzLM

**Can a 0.6B model beat an 8B model at Kazakh?**

![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-EE4C2C?logo=pytorch&logoColor=white)
![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue)
![Kazakh](https://img.shields.io/badge/Language-Kazakh-green)

Monorepo of the Kazakh model campaign — three packages, one uv workspace:

| Package | Dir | What it is |
|---------|-----|------------|
| `qymyzlm` (import `kazllm`) | `src/` | THE ENGINE: Kazakh SLM — Engram n-gram memory, QLoRA continual pretraining, benchmark study |
| `qymyz-embed` | `embed/` | QymyzEmbed: best Kazakh text-embedding model (target: beat mE5-large MRR 0.909 on KazQAD) |
| `kazeval` | `evallab/` | Benchmarking lab: KazQAD retrieval/reranking (kk-MTEB), KazMMLU, Qorgau guardrails — ALL published numbers come from runners here |

Sibling project: [kazakh-nlp-atlas](https://github.com/kekeront/kazakh-nlp-atlas) — data-driven survey of Kazakh NLP + frontier knowledge base.

```bash
uv sync --all-packages --all-extras   # one env for the whole workspace
```

---

## Goal

Close the gap to [Sherkala-Chat-8B](https://arxiv.org/abs/2503.01493) (41.4% KazMMLU — reported, not yet re-run under our harness) with a **sub-1B active-parameter** model. First checkpoint: beat Qwen3-0.6B-Base (32.8%, measured here) by ≥3pp. Levers:

1. **QLoRA continual pretraining** on high-quality Kazakh data
2. **Engram** — sparse N-gram memory for agglutinative suffix patterns (0 extra FLOPs)
3. **Kazakh SFT** — instruction tuning with teacher-distilled reasoning
4. **Test-Time Scaling** — Process Reward Model for best-of-N inference

## Baseline Results (KazMMLU 3-shot, Kazakh subset)

Evaluated on 9,870 Kazakh-language questions from [MBZUAI/KazMMLU](https://huggingface.co/datasets/MBZUAI/KazMMLU). Non-italic rows measured on RTX 2070 in fp16, April 2026; *italic rows are reported numbers from their papers (different harness — not directly comparable)*. Shot count: the KazMMLU dev split holds only 3 exemplars per subject, so runs previously labeled "5-shot" were effectively **3-shot**.

| Model | Params | KazMMLU 3-shot | Tok/word | Tok/sec |
|-------|--------|---------------|----------|---------|
| **Qwen2.5-1.5B** | 1.54B | **34.3%** | 4.88 | 35.6 |
| **Qwen3-0.6B-Base** | 0.6B | **32.8%** | 4.88 | 31.3 |
| Qwen2.5-0.5B | 0.49B | 28.8% | 4.88 | 37.2 |
| Gemma3-1B-it | 1.0B | 28.7% | 3.27 | 19.2 |
| Llama-3.2-1B | 1.24B | 25.1% | 4.80 | 46.8 |
| Gemma3-270M | 0.27B | 24.4% | 3.27 | 28.5 |
| Ekitil-Qwen3-600M | 0.67B | 23.7% | 1.43 | 22.8 |
| Ekitil-Qwen3-300M | 0.25B | 23.5% | 1.43 | 48.7 |
| Random baseline | — | 25.0% | — | — |
| *Sherkala-Chat-8B* | *8B* | *41.4%* | *2.04* | *—* |
| *Llama-3.1-70B* | *70B* | *55.2%* | *4.73* | *—* |

**Key findings:**
- **Qwen dominates** at small scale — Qwen3-0.6B beats every other sub-2B model
- **Llama-3.2-1B is random-level** (25.1%) — zero Kazakh knowledge
- **From-scratch Kazakh models fail** — Ekitil has elite tokenizer (1.43 tok/word) but scores below random on knowledge tasks
- **Fine-tuning > training from scratch** — multilingual pretraining provides free world knowledge

**Primary target: Qwen3-0.6B-Base** — 32.8% baseline; nominal gap to Sherkala-8B is 8.6pp, but that ceiling is cross-protocol (their paper's harness) until Sherkala is re-run through our runner.

## Pipeline

```bash
# 1. Benchmark any model
python scripts/benchmark_baselines.py --models Qwen/Qwen3-0.6B-Base

# 2. QLoRA continual PT on Kazakh data (fits on RTX 2070)
python scripts/qlora_continual.py --tokens 100_000_000

# 3. Benchmark the fine-tuned model
python scripts/benchmark_baselines.py --models checkpoints/qlora_qwen3-0.6b-base_100m
```

## Roadmap

- [x] Baseline eval: 8 models on KazMMLU
- [ ] QLoRA continual PT: 100M / 200M / 400M Kazakh tokens
- [ ] Kazakh SFT: teacher-distilled instruction data
- [ ] Engram integration: suffix-pattern memory
- [ ] Test-Time Scaling: PRM + best-of-N inference

## References

- [Engram](https://arxiv.org/abs/2601.07372) — Conditional N-gram memory (DeepSeek-AI)
- [Sherkala](https://arxiv.org/abs/2503.01493) — 8B Kazakh continual PT (ISSAI)
- [SozKZ](https://arxiv.org/abs/2603.20854) — 600M Kazakh from scratch
- [KazMMLU](https://arxiv.org/abs/2502.12829) — ACL 2025, 23K questions

## License

Apache 2.0
