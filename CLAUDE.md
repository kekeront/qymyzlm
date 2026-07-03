# QymyzLM (package `kazllm`) — Claude Code Instructions

## Project Overview

QymyzLM is a Kazakh-language SLM project. After the pivot (commit "Pivot to multi-model benchmark
study + QLoRA pipeline") the working approach is: benchmark existing small models on KazMMLU,
QLoRA-continual-pretrain the best one (Qwen3-0.6B-Base, 32.8% baseline) on Kazakh data, and graft
**Engram** sparse N-gram memory onto it. Canonical target: beat Qwen3-0.6B's 32.8% KazMMLU
3-shot (aim ≥36%); ceiling reference Sherkala-Chat-8B at 41.4% (reported, cross-protocol).
Shot count is **dev-limited**: KazMMLU's dev split has only 3 exemplars per subject, so 3-shot is
the maximum with subject-matched shots — runs previously labeled "5-shot" were effectively 3-shot.

Core problems:
- Multilingual tokenizers hit **4.73 tokens/word** for Kazakh (Llama-3.1 baseline). Target: **<2.0**
- Standard Transformers waste early-layer depth reconstructing Kazakh morphological N-gram patterns
  through computation. Engram offloads this to O(1) lookup.

**Monorepo note:** this repo is a uv workspace — `embed/` (qymyz-embed) and `evallab/` (kazeval)
are workspace members with their own CLAUDE.md files; this file covers only the engine (`src/`,
`scripts/`, `configs/`, `tests/`).

**Reference implementations:**
- SozKZ (arxiv 2603.20854): 600M from-scratch, 9B tokens, 50K BPE, ~2.0 fertility
- Sherkala (arxiv 2503.01493): 8B continual PT from Llama-3.1-8B, 159K vocab, 2.04 fertility
- Engram (arxiv 2601.07372): conditional N-gram memory as a complementary sparsity axis
- mHC (arxiv 2512.24880): manifold-constrained hyper-connections for stable n-stream residual

## Project Structure (implemented today — what is actually on disk)

```
src/kazllm/
├── model/
│   ├── engram.py               # EngramModule: hash tables, context-aware gating, causal conv
│   ├── norm.py                 # RMSNorm (used internally by Engram)
│   └── qwen_engram_wrapper.py  # QymyzForCausalLM: grafts Engram onto a pretrained HF causal LM
├── eval/
│   ├── benchmarks.py           # KazMMLU task def (lm-eval ships NO Kazakh tasks) + FLORES pairs
│   ├── harness.py              # run_benchmarks(): lm-eval wrapper, TaskManager(include_path=...)
│   ├── tasks/kazmmlu_kaz/      # bundled custom lm-eval task dir (12 subjects, 3-shot, group task)
│   ├── metrics.py              # accuracy, chrF++
│   └── results.py              # BenchmarkResult / EvalRun JSON serialisation
├── tokenizer/
│   └── fertility.py            # tokens/word fertility benchmark (Llama-3.1 kaz baseline 4.73)
├── training/
│   └── callbacks.py            # ThroughputCallback: tok/s + MFU logging
└── utils/                      # io.py (atomic JSON, shard paths), logging.py (rank-aware), seed.py
scripts/
├── benchmark_baselines.py      # KazMMLU 3-shot bench: accuracy + fertility + speed (PEFT-aware)
└── qlora_continual.py          # 4-bit QLoRA continual PT on streamed Kazakh data (RTX 2070 OK)
configs/
├── eval/default.yaml           # benchmark list + dtype defaults (mirrors eval/benchmarks.py)
└── training/sft_lora.yaml      # planned SFT stage — no consumer script yet
tests/                          # engram invariants, wrapper grafting, fertility, import smoke
```

There is NO from-scratch model code on disk (no config.py/mhc.py/block.py/model.py/attention.py,
no data pipeline, no train.py). Do not reference those paths from code — see the design spec below.

## Development Workflow

```bash
# uv workspace: one venv at the repo root, shared by embed/ and evallab/
uv sync --all-packages --all-extras   # or: make install

make lint          # .venv/bin/ruff check + format --check on src, scripts, tests
make test          # PYTHONPATH= .venv/bin/python -m pytest tests -q  (CPU-only, no downloads;
                   # PYTHONPATH cleared — ROS dist-packages break pytest plugin autoload)

# Real pipelines (these DO download models/datasets — never run just to "test")
make benchmark        # KazMMLU 3-shot baselines (scripts/benchmark_baselines.py)
make benchmark-quick  # same, 100 questions per model
make qlora            # QLoRA continual PT (scripts/qlora_continual.py), TOKENS=100_000_000 default
```

Ad-hoc invocations use the workspace venv directly: `.venv/bin/python scripts/benchmark_baselines.py
--models Qwen/Qwen3-0.6B-Base`.

## Key Invariants (enforced by tests/)

- **Never commit** `data/`, `checkpoints/`, `evals/`, `outputs/` (gitignored)
- **Engram conv.weight initialised to 0** — the depthwise causal conv stage is identity at step 0
- **Engram sparse=True on nn.Embedding** — critical for performance with large tables
- **Engram q/k/v projections small-init (std 0.01)** — early training is not disrupted
- **canonical_map defaults to identity** — updated only by tokenizer compression
- **Fertility target: <2.0** (SozKZ reference baseline)
- **KazMMLU 3-shot (dev-limited)** is the primary evaluation metric; all published numbers come
  from `evallab/` runners. Never label results "5-shot": the dev split has only 3 exemplars per
  subject, and lm-eval's first_n sampler hard-fails at num_fewshot=5
- Benchmark run artifacts default to `results/baselines/` (committed-friendly, with model id,
  revision, repo commit, date, shot count) — `evals/` is gitignored and NOT regenerable

---

# KazLLM-v2 design spec (pending design-panel — NOT implemented)

Everything below is design material for a future from-scratch KazLLM-v2. None of these files exist
on disk. Implementation is gated on a `/design-panel` spec; keep this section for that panel.

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

### mHC Details (planned module — does not exist yet)
- 4 parallel residual streams (n=4); standard attention/FFN compute unchanged
- H_res projected onto Birkhoff polytope via Sinkhorn-Knopp (20 iters) → spectral norm ≤ 1 → no gradient explosion
- Dynamic + static mappings (α·tanh(xφ) + b); α initialised to 0 → pure standard residual at training start
- ~15-25% training overhead without TileLang kernel fusion (vs 6.7% in paper); acceptable for our budget
- **mHC alpha params initialised to 0** — do NOT change this init (model behaves as standard Llama at step 0)

### Engram Details (src/kazllm/model/engram.py — this part IS implemented)
- {2,3}-gram suffix tables, K=4 hash heads, M=500K slots (500M), slot_dim=64
- Context-aware gating: α = σ(RMSNorm(h)·RMSNorm(W_K e)/√d) — suppresses hash collisions/polysemy
- Depthwise causal conv (kernel=4): Y = SiLU(Conv(RMSNorm(Ṽ))) + Ṽ
- Tokenizer compression buffer (canonical_map): NFKC+lowercase equivalent; reduces effective vocab ~23%
- Injected BEFORE the block at specified layer indices; adds memory contribution to hidden states
- At inference: can be offloaded to host RAM (<3% throughput penalty per paper)

### Why Engram is especially important for Kazakh
Agglutinative suffix chains (plural, case, possessive, tense, person) form completely stereotyped
{2,3}-gram patterns. Standard Transformers waste 4-6 early layers reconstructing these (see Table 3
in the Engram paper). Engram offloads this lookup, freeing attention capacity for semantic reasoning.

### Planned v2 layout (for the design panel; do not create without a spec)
```
src/kazllm/model/{config,mhc,block,model,attention,mlp,rope}.py
src/kazllm/data/            # download → filter → dedup → pack pipeline
configs/model/{kaz500m,kaz1b,kaz1b_continual}.yaml
```

### Ablation Config Flags (planned)
- `model.use_mhc=false model.use_engram=false` → standard Llama baseline
- `model.use_mhc=true model.use_engram=false` → mHC only
- `model.use_mhc=false model.use_engram=true` → Engram only
- `model.use_mhc=true model.use_engram=true` → full KazLLM-v2 (default)

Packed-shard convention when the data pipeline lands: **uint16 dtype** (vocab ≤65535, saves 50%
disk vs int32); all hyperparameters in `configs/` — no hardcoded values in Python.

## Data Sources (~9-10B Kazakh tokens total, for v2 pretraining)

| Source | HF repo | Est. tokens after filter | Weight |
|--------|---------|--------------------------|--------|
| CulturaX | `uonlp/CulturaX` (kz) | ~2.0B | 1.0x |
| HPLT 2.0 | `HPLT/HPLT2.0_cleaned` (kaz_Cyrl) | ~1.8B | 1.0x |
| mC4 | `mc4` (kk) | ~1.0B | 1.0x |
| MADLAD-400 | `google/madlad400` (kk) | ~0.7B | 1.0x |
| mOSCAR | `oscar-corpus/mOSCAR` (kaz) | ~0.5B | 1.0x |
| Wikipedia | `wikipedia` (kk) | ~0.18B | 3.0x |
| multidomain | `kz-transformers/multidomain-kazakh-dataset` | ~0.25B | 2.0x |

(The QLoRA pipeline today streams a smaller subset: Wikipedia, multidomain, HPLT 2.0, C4 —
see `SOURCES` in scripts/qlora_continual.py.)

## Hardware Notes

### RTX 2070 (8GB VRAM) — what it can do:
- Data preprocessing, tokenizer training, debugging code: ✓
- Inference of quantized models (4-bit 1B: ~0.5GB): ✓
- QLoRA fine-tuning of 500M-3B (4-bit base): ✓ ← **the current pipeline**
- Full pretraining of 50M debug model: ✓
- Full pretraining of 500M+: ✗ (need ~15-20GB for model + optimizer + mHC activations)

### GCP Instance Guide (for v2 pretraining):
| Instance | GPUs | VRAM | Use For | Spot Price |
|----------|------|------|---------|------------|
| `a2-highgpu-1g` | 1× A100 40GB | 40GB | **500M pretraining** | ~$3.67/hr |
| `a2-highgpu-2g` | 2× A100 40GB | 80GB | 1B pretraining | ~$7.35/hr |
| `a2-highgpu-4g` | 4× A100 40GB | 160GB | 1B fast or multi-exp | ~$14.69/hr |
| `a2-ultragpu-1g` | 1× A100 80GB | 80GB | 1B pretraining | ~$4.63/hr |

Use preemptible (spot) instances — 60-70% cheaper. Save checkpoints every 2K steps.

---

## Evaluation Benchmarks

| Benchmark | Metric | Shots | Runner today | Target |
|-----------|--------|-------|--------------|--------|
| KazMMLU (kaz subset) | accuracy | 3 (dev-limited) | this repo: `scripts/benchmark_baselines.py` + bundled lm-eval task `kazmmlu_kaz` | beat Qwen3-0.6B 32.8%; aim ≥36% |
| TUMLU-mini | accuracy | TBD | `evallab/` only (no lm-eval task exists in 0.4.11) | Turkic cross-lingual comparison |
| KazQAD | F1 | 0 | `evallab/` only (no lm-eval task exists in 0.4.11) | Reading comprehension |
| FLORES-200 kaz↔eng/rus | chrF++ | — | none yet (chrF++ helper in `kazllm.eval.metrics`) | Translation quality |

**Current SOTA (KazMMLU):** Sherkala-Chat-8B at 41.4% (NOT 47.6% — that's avg across 13 tasks);
Llama-3.1-70B at 55.2%. Full baseline table (8 models, RTX 2070): see README.md.

## Kazakh Language Notes

- Agglutinative: suffixes stack deterministically (stem + plural + case + possessive + personal = 1 word)
- Vowel harmony: suffix allomorphs double effective N-gram vocabulary (handled by Engram multi-head hashing)
- Script: 42-letter Cyrillic including Ә,Ғ,Қ,Ң,Ө,Ұ,Ү,Һ (absent from Russian Cyrillic)
- character_coverage=0.9999 required to cover all Kazakh-specific Unicode codepoints
- byte_fallback=True ensures no UNK on Latin-script Kazakh text (transition period 2031)
- Engram tokenizer compression: NFKC+lowercase canonical map reduces effective vocab ~23%
