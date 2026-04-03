# QymyzLM

**Morphological memory makes agglutinative LMs punch above their weight.**

![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-EE4C2C?logo=pytorch&logoColor=white)
![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue)
![Kazakh](https://img.shields.io/badge/Language-Kazakh-green)
![Status: Active](https://img.shields.io/badge/Status-Active_Development-yellow)

---

## Problem

Kazakh is severely underserved by existing LLMs:

- **Tokenizer inefficiency**: Llama-3.1 needs **4.73 tokens/word** for Kazakh (vs ~1.3 for English) — wasting 3.6x compute on subword reconstruction
- **Morphological blindness**: Kazakh is agglutinative — a single word can carry 5+ suffixes (plural + case + possessive + tense + person). Standard Transformers waste 4-6 early layers reconstructing these stereotyped patterns
- **No dedicated small models**: Existing Kazakh-capable models are either 8B+ (Sherkala) or multilingual models that allocate minimal capacity to Kazakh

## Solution

QymyzLM grafts [Engram](https://arxiv.org/abs/2601.07372) conditional N-gram memory onto Qwen-2.5-1.5B via continual pretraining on 9B Kazakh tokens. Engram offloads stereotyped suffix-chain reconstruction to O(1) hash table lookups at **zero additional FLOPs**, freeing the backbone for semantic reasoning.

| Technique | What it does | Impact |
|-----------|-------------|--------|
| **Engram sparse memory** | O(1) lookup tables for {2,3}-gram suffix patterns | Offloads morphological reconstruction — **0 additional FLOPs** |
| **Vocab expansion** | Add ~8K Kazakh morphemes to Qwen's 151K vocab | **<2.0 tokens/word** (vs 4.73 for Llama-3.1) |
| **Continual PT** | 9B Kazakh + 1B English on Qwen-2.5-1.5B | Preserve multilingual capability while specializing |

**Target**: >45% KazMMLU — beating Sherkala-Chat-8B (41.4%) with **5x fewer parameters** and **~$120 compute**.

## Quick Start

```bash
git clone https://github.com/altairzhambyl/qymyzlm.git && cd qymyzlm
uv sync --all-extras
make test                  # verify everything works (69 tests)

# Continual PT pipeline (the headline model)
python scripts/build_continual_pt.py   # expand vocab + graft Engram onto Qwen-2.5-1.5B
python scripts/train.py model=qymyz1_5b training=pretrain_continual

# From-scratch ablation (research proof)
make train-500m            # 500M + Engram on A100
```

## Architecture

```
              QymyzLM-1.5B (Continual PT)
 ┌──────────────────────────────────────────────────────────┐
 │                     Input Tokens                          │
 │                          │                                │
 │                   ┌──────▼──────┐                         │
 │                   │  Embedding   │  Qwen 151K + 8K Kazakh │
 │                   └──────┬──────┘                         │
 │                          │                                │
 │  Layers 0-1  ┌───────────▼───────────┐                    │
 │              │  Qwen Decoder Layers  │                    │
 │              └───────────┬───────────┘                    │
 │                          │                                │
 │  Layer 2     ┌───────────▼───────────┐  ┌──────────────┐  │
 │              │  ◄── Engram Inject ───│◄─│ {2,3}-gram   │  │
 │              │  Qwen Decoder Layer   │  │ Hash Tables  │  │
 │              └───────────┬───────────┘  │ (512M sparse)│  │
 │                          │              │ 0 FLOPs      │  │
 │  Layers 3-6  ┌───────────▼───────────┐  └──────────────┘  │
 │              │  Qwen Decoder Layers  │                    │
 │              └───────────┬───────────┘                    │
 │                          │                                │
 │  Layer 7     ┌───────────▼───────────┐  ┌──────────────┐  │
 │              │  ◄── Engram Inject ───│◄─│ {2,3}-gram   │  │
 │              │  Qwen Decoder Layer   │  │ Hash Tables  │  │
 │              └───────────┬───────────┘  └──────────────┘  │
 │                          │                                │
 │  Layers 8-27 ┌───────────▼───────────┐                    │
 │              │  Qwen Decoder × 20    │                    │
 │              └───────────┬───────────┘                    │
 │                          │                                │
 │                   ┌──────▼──────┐                         │
 │                   │   LM Head   │                         │
 │                   └─────────────┘                         │
 └──────────────────────────────────────────────────────────┘
```

### Key Component

**Engram** (`src/kazllm/model/engram.py`): Conditional N-gram memory with context-aware gating. Hash tables store stereotyped Kazakh suffix patterns for O(1) lookup. Injected at layers 2 and 7 — adds memory contribution to hidden states before the block computes attention. Zero additional FLOPs; tables can be offloaded to host RAM at <3% throughput penalty.

## Model Configurations

| Config | Params | Base | Use Case |
|--------|--------|------|----------|
| **`qymyz1_5b`** | **1.5B + 512M sparse** | **Qwen-2.5-1.5B** | **Headline model (1x A100 40GB, ~$120)** |
| `kaz500m` | 530M + 512M sparse | From scratch | Ablation proof (1x A100 40GB) |
| `kaz50m_debug` | 50M | From scratch | Local debugging (RTX 2070) |
| `kaz_nano` | 43M | From scratch | Deep-and-thin research |

## Training Data

~9-10B Kazakh tokens from 7 sources:

| Source | Tokens | Weight |
|--------|--------|--------|
| CulturaX | ~2.0B | 1.0x |
| HPLT 2.0 | ~1.8B | 1.0x |
| mC4 | ~1.0B | 1.0x |
| MADLAD-400 | ~0.7B | 1.0x |
| mOSCAR | ~0.5B | 1.0x |
| Wikipedia (kk) | ~0.18B | 3.0x |
| Multidomain Kazakh | ~0.25B | 2.0x |

## Evaluation Targets

| Benchmark | Metric | Target | SOTA Reference |
|-----------|--------|--------|----------------|
| KazMMLU (5-shot) | accuracy | **>45%** | Sherkala-Chat-8B: 41.4%, Llama-3.1-70B: 55.2% |
| TUMLU-mini (5-shot) | accuracy | — | Turkic cross-lingual |
| FLORES-200 kaz↔eng/rus | chrF++ | — | Translation quality |

## Project Structure

```
src/kazllm/
├── model/
│   ├── config.py        # KazLLMConfig dataclass
│   ├── model.py         # KazLLMModel: mHC streams + Engram injection
│   ├── mhc.py           # MHCStreamManager, Sinkhorn-Knopp
│   ├── engram.py         # EngramModule: hash tables, gating, conv
│   ├── block.py          # TransformerBlock
│   ├── attention.py      # GQA + RoPE + Flash Attention 2
│   ├── mlp.py            # SwiGLU
│   ├── norm.py           # RMSNorm
│   └── rope.py           # RoPE frequencies
├── data/                 # Download → filter → dedup → pack pipeline
├── tokenizer/            # SentencePiece training, fertility benchmarks
├── training/             # Trainer, scheduler, FSDP, callbacks
├── eval/                 # lm-eval harness, benchmarks, metrics
└── utils/                # Config, logging, I/O, seed
configs/
├── model/                # kaz50m_debug, kaz_nano, kaz500m, kaz1b
├── training/             # pretrain_debug, pretrain_500m, sft_lora
└── data/                 # local_validate, full pipeline
```

## Development

```bash
uv sync --all-extras       # install all dependencies
make lint                  # ruff check + format
make test                  # pytest (no GPU needed)

# Full pipeline
make data                  # download + clean all sources
make tokenizer             # train Unigram 50K
make pack                  # tokenize + pack into uint16 shards
make train-500m            # pretrain 500M on A100
make eval                  # run all benchmarks
```

## Ablation Support (From-scratch 500M)

```bash
# Standard Llama baseline
python scripts/train.py model=kaz500m model.use_engram=false model.use_mhc=false

# + Engram only (the research hypothesis)
python scripts/train.py model=kaz500m model.use_engram=true model.use_mhc=false
```

## Competitive Context

| Model | Params | KazMMLU | Compute | Notes |
|-------|--------|---------|---------|-------|
| Random baseline | — | 25.0% | — | 4-way MC |
| SozKZ-600M | 600M | ?? | 8x H100 | Never ran KazMMLU |
| Qwen-2.5-7B | 7B | 35.1% | — | Multilingual |
| Llama-3.1-8B | 8B | 38.3% | — | Multilingual |
| Sherkala-Chat-8B | 8B | 41.4% | Cerebras CS-2 | NOT 47.6% (that's avg across 13 tasks) |
| **QymyzLM-1.5B** | **1.5B** | **>45%?** | **1x A100, ~$120** | **You are here** |
| Llama-3.1-70B | 70B | 55.2% | — | Upper bound |

## References

- [Engram](https://arxiv.org/abs/2601.07372) — Conditional N-gram memory (DeepSeek-AI)
- [TOBA-LM](https://arxiv.org/abs/2603.10006) — Engram on Austronesian agglutinative languages
- [Sherkala](https://arxiv.org/abs/2503.01493) — 8B Kazakh continual PT
- [SozKZ](https://arxiv.org/abs/2603.20854) — 600M Kazakh from scratch
- [KazMMLU](https://arxiv.org/abs/2502.12829) — ACL 2025, 23K questions

## License

Apache 2.0
