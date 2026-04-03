# KazLLM — Claude Code Instructions

## Project Overview

KazLLM is a Kazakh language small language model focused on token efficiency and morphological awareness.
Primary target: **KazLLM-500M** (backbone) + **~512M Engram sparse memory** = 0 additional FLOPs.

Core problems:
- Multilingual tokenizers hit **4.73 tokens/word** for Kazakh (Llama-3.1 baseline). Target: **<2.0**
- Standard Transformers waste early-layer depth reconstructing Kazakh morphological N-gram patterns through computation. Engram offloads this to O(1) lookup.

**Reference implementations:**
- SozKZ (arxiv 2603.20854): 600M from-scratch, 9B tokens, 50K BPE, ~2.0 fertility
- Sherkala (arxiv 2503.01493): 8B continual PT from Llama-3.1-8B, 159K vocab, 2.04 fertility
- Engram (arxiv 2601.07372): conditional N-gram memory as a complementary sparsity axis
- mHC (arxiv 2512.24880): manifold-constrained hyper-connections for stable n-stream residual

## Architecture (KazLLM-v2)

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Tokenizer | SentencePiece Unigram 50K | Best morpheme alignment for agglutinative Kazakh |
| Backbone | Llama-style decoder-only | Validated by SozKZ; from-scratch training |
| Residual | **mHC n=4 streams** | Stable training (doubly-stochastic H_res); richer morphological representations |
| N-gram memory | **Engram at layers {2, L/4}** | Offloads stereotyped Kazakh suffix patterns from backbone; zero FLOPs |
| Model size | **500M backbone** (primary) | Chinchilla-optimal for 9B tokens; fits on 1× A100 40GB |
| 1B variant | 1B (research target) | Needs 1× A100 80GB or 2× A100 40GB |
| Positional enc. | RoPE, theta=500K | Extended context for long agglutinative forms |
| Attention | GQA (2:1 ratio) | 2× KV cache reduction |
| Norm | RMSNorm (pre-norm) | Standard Llama; stable |
| Activation | SwiGLU | Standard Llama |
| Script | Cyrillic-primary, Latin-aware | Transition deadline 2031 |

### mHC Details (src/kazllm/model/mhc.py)
- 4 parallel residual streams (n=4); standard attention/FFN compute unchanged
- H_res projected onto Birkhoff polytope via Sinkhorn-Knopp (20 iters) → spectral norm ≤ 1 → no gradient explosion
- Dynamic + static mappings (α·tanh(xφ) + b); α initialised to 0 → pure standard residual at training start
- ~15-25% training overhead without TileLang kernel fusion (vs 6.7% in paper); acceptable for our budget

### Engram Details (src/kazllm/model/engram.py)
- {2,3}-gram suffix tables, K=4 hash heads, M=500K slots (500M), slot_dim=64
- Context-aware gating: α = σ(RMSNorm(h)·RMSNorm(W_K e)/√d) — suppresses hash collisions/polysemy
- Depthwise causal conv (kernel=4): Y = SiLU(Conv(RMSNorm(Ṽ))) + Ṽ
- Tokenizer compression buffer (canonical_map): NFKC+lowercase equivalent; reduces effective vocab ~23%
- Injected BEFORE the block at specified layer indices; adds memory contribution to hidden states
- At inference: can be offloaded to host RAM (<3% throughput penalty per paper)

### Why Engram is especially important for Kazakh:
Agglutinative suffix chains (plural, case, possessive, tense, person) form completely stereotyped {2,3}-gram
patterns. Standard Transformers waste 4-6 early layers reconstructing these (see Table 3 in Engram paper).
Engram offloads this lookup, freeing attention capacity for semantic reasoning.

## Project Structure

```
src/kazllm/
├── model/
│   ├── config.py      # KazLLMConfig with mhc_streams, engram_* params
│   ├── mhc.py         # MHCStreamManager, sinkhorn_knopp, expand/collapse_streams
│   ├── engram.py      # EngramModule: hash tables, gating, depthwise conv
│   ├── block.py       # TransformerBlock: standard or mHC forward
│   ├── model.py       # KazLLMModel: mHC stream management + Engram injection
│   ├── attention.py   # GQA + RoPE + Flash Attention 2
│   ├── mlp.py         # SwiGLU
│   ├── norm.py        # RMSNorm
│   └── rope.py        # RoPE frequencies
├── data/              # download → filter → dedup → pack pipeline
├── tokenizer/         # SentencePiece training, fertility benchmarks, HF wrapper
├── training/          # trainer, scheduler, FSDP utils, callbacks, loss
├── eval/              # lm-eval harness, benchmarks, metrics, results
└── utils/             # config dataclasses, logging, io, seed
configs/model/
├── kaz500m.yaml       # PRIMARY: 16L h=1536, mHC n=4, Engram layers {2,4}
├── kaz1b.yaml         # RESEARCH: 22L h=2048, mHC n=4, Engram layers {2,6}
└── kaz1b_continual.yaml  # continual PT variant from Llama-3.2-1B
```

## Development Workflow

```bash
uv sync --all-extras       # install deps
make lint                  # ruff check + format check
make test                  # pytest (no GPU required for most tests)

# Data pipeline
make data                  # download + clean all sources
make tokenizer             # train Unigram 50K tokenizer
make pack                  # tokenize + pack into uint16 shards

# Training (local debug, RTX 2070 OK for 50M debug config)
make train-debug           # pretrain_debug config (100M tokens, 1K steps)

# Training (GCP)
# 500M: a2-highgpu-1g (1× A100 40GB), pretrain_500m config, ~$264 spot
python scripts/train.py model=kaz500m training=pretrain_500m

# 1B: a2-highgpu-2g (2× A100 40GB), pretrain config
accelerate launch scripts/train.py model=kaz1b training=pretrain

make eval                  # run KazMMLU, TUMLU, KazQAD, FLORES-200
```

## Hardware Notes

### RTX 2070 (8GB VRAM) — what it can do:
- Data preprocessing, tokenizer training, debugging code: ✓
- Inference of quantized models (4-bit 1B: ~0.5GB): ✓
- QLoRA fine-tuning of 500M-3B (4-bit base): ✓
- Full pretraining of 50M debug model: ✓
- Full pretraining of 500M+: ✗ (need ~15-20GB for model + optimizer + mHC activations)

### GCP Instance Guide:
| Instance | GPUs | VRAM | Use For | Spot Price |
|----------|------|------|---------|------------|
| `a2-highgpu-1g` | 1× A100 40GB | 40GB | **500M pretraining** | ~$3.67/hr |
| `a2-highgpu-2g` | 2× A100 40GB | 80GB | 1B pretraining | ~$7.35/hr |
| `a2-highgpu-4g` | 4× A100 40GB | 160GB | 1B fast or multi-exp | ~$14.69/hr |
| `a2-ultragpu-1g` | 1× A100 80GB | 80GB | 1B pretraining | ~$4.63/hr |

Use preemptible (spot) instances — 60-70% cheaper. Save checkpoints every 2K steps.

**Recommended**: Start with `a2-highgpu-1g` spot for 500M. Set up Hydra to resume from checkpoint on preemption.

## Key Invariants

- **Never commit** `data/`, `checkpoints/`, `evals/`, `outputs/` (gitignored)
- **All hyperparameters in `configs/`** — no hardcoded values in Python
- **uint16 dtype** for packed token shards (vocab ≤65535, saves 50% disk vs int32)
- **Fertility target: <2.0** (SozKZ reference baseline)
- **KazMMLU 5-shot** is the primary evaluation metric
- **mHC alpha params initialised to 0** — do NOT change this init (model behaves as standard Llama at step 0)
- **Engram conv.weight initialised to 0** — identity init ensures stable training start
- **Engram sparse=True on nn.Embedding** — critical for performance with large tables

## Ablation Config Flags

Both mHC and Engram can be disabled independently for ablation:
- `model.use_mhc=false model.use_engram=false` → standard Llama baseline
- `model.use_mhc=true model.use_engram=false` → mHC only
- `model.use_mhc=false model.use_engram=true` → Engram only
- `model.use_mhc=true model.use_engram=true` → full KazLLM-v2 (default)

## Data Sources (~9-10B Kazakh tokens total)

| Source | HF repo | Est. tokens after filter | Weight |
|--------|---------|--------------------------|--------|
| CulturaX | `uonlp/CulturaX` (kz) | ~2.0B | 1.0x |
| HPLT 2.0 | `HPLT/HPLT2.0_cleaned` (kaz_Cyrl) | ~1.8B | 1.0x |
| mC4 | `mc4` (kk) | ~1.0B | 1.0x |
| MADLAD-400 | `google/madlad400` (kk) | ~0.7B | 1.0x |
| mOSCAR | `oscar-corpus/mOSCAR` (kaz) | ~0.5B | 1.0x |
| Wikipedia | `wikipedia` (kk) | ~0.18B | 3.0x |
| multidomain | `kz-transformers/multidomain-kazakh-dataset` | ~0.25B | 2.0x |

## Evaluation Benchmarks

| Benchmark | Metric | Shots | Target |
|-----------|--------|-------|--------|
| KazMMLU | accuracy | 5 | >35% (SozKZ 600M ~30%; aim to beat with 500M+Engram) |
| TUMLU-mini | accuracy | 5 | Turkic cross-lingual comparison |
| KazQAD | F1 | 0 | Reading comprehension |
| FLORES-200 kaz↔eng/rus | chrF++ | — | Translation quality |

**Current SOTA (KazMMLU):** Sherkala-Chat-8B at 41.4% (NOT 47.6% — that's avg across 13 tasks); Llama-3.1-70B at 55.2%

## Kazakh Language Notes

- Agglutinative: suffixes stack deterministically (stem + plural + case + possessive + personal = 1 word)
- Vowel harmony: suffix allomorphs double effective N-gram vocabulary (handled by Engram multi-head hashing)
- Script: 42-letter Cyrillic including Ә,Ғ,Қ,Ң,Ө,Ұ,Ү,Һ (absent from Russian Cyrillic)
- character_coverage=0.9999 required to cover all Kazakh-specific Unicode codepoints
- byte_fallback=True ensures no UNK on Latin-script Kazakh text (transition period 2031)
- Engram tokenizer compression: NFKC+lowercase canonical map reduces effective vocab ~23%
