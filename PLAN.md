# KazLLM — End-to-End Build Plan

**Two-pronged strategy (updated 2026-04-03):**
1. **Continual PT from Qwen-2.5-1.5B + Engram** → headline number (target 45%+ KazMMLU, beats Sherkala-Chat-8B at 41.4%)
2. **From-scratch 500M ± Engram ablation** → research proof (Engram delta on Turkic morphology)

**Note:** mHC dropped from primary experiment (sHC paper showed identity degeneration, 15-25% overhead, marginal benefit at 16 layers).
SozKZ NEVER ran KazMMLU — their 30.3% is kk-socio-cultural-bench. Sherkala's real KazMMLU is 41.4%, NOT 47.6% (that's avg across 13 tasks).

---

## Phase 0 — Local Validation (RTX 2070, no cloud cost)

Goal: verify the full pipeline runs end-to-end before spending money on GCP.

### 0.1 Unit tests pass
```bash
uv sync --all-extras
make lint
make test
```
All tests in `tests/` must pass:
- `test_mhc.py` — Sinkhorn-Knopp, stream roundtrip, gradient flow
- `test_engram.py` — hash determinism/range, output shape, gradient flow, canonical map
- `test_model_forward.py` — forward + backward for all 4 ablation modes

Status: [x] passing (33/33)

### 0.2 Missing module stubs
These modules are referenced but need implementation review:
- [ ] `src/kazllm/model/__init__.py` — export mhc, engram, block, model, config
- [ ] `src/kazllm/training/trainer.py` — verify ShardedMemmapDataset + KazLLMTrainer work with mHC shapes
- [ ] `src/kazllm/training/callbacks.py` — verify ThroughputCallback handles n-stream tensors

**Fixed during lint/test phase:**
- [x] `scripts/train.py` — now passes all mHC/Engram fields to KazLLMConfig
- [x] `configs/model/kaz50m_debug.yaml` — created (71.5M params, fits RTX 2070)
- [x] `configs/training/pretrain_debug.yaml` — added context_length=1024, gradient_checkpointing

### 0.3 Debug training run (50M model, local)
```bash
make train-debug   # pretrain_debug config: 100M tokens, 1K steps, no FSDP
```
- Config: `configs/training/pretrain_debug.yaml` with `model=kaz500m` (or smaller debug override)
- Verify: loss decreases, no NaN, checkpoint saves/resumes, W&B logs appear
- Hardware: RTX 2070 8GB — needs debug model ≤50M params (hidden=512, 6 layers, n_streams=2)

Add `configs/model/kaz50m_debug.yaml`:
```yaml
# 50M debug model — fits on RTX 2070 with gradient_checkpointing
vocab_size: 50000; hidden_size: 512; intermediate_size: 1376
num_hidden_layers: 6; num_attention_heads: 8; num_key_value_heads: 4
max_position_embeddings: 1024; use_flash_attention: false
use_mhc: true; mhc_streams: 2
use_engram: true; engram_layer_indices: [1, 3]; engram_table_size: 10007
engram_slot_dim: 32; engram_num_heads: 2
```

Status: [ ] debug run completes without error  ← NEXT STEP

---

## Phase 1 — Data Pipeline (Local + GCP prep)

### 1.1 Download and filter sources
```bash
make data   # runs scripts/download_data.py + scripts/clean_data.py
```
Sources (target ~9-10B tokens after filtering):
| Source | HF repo | Est. tokens |
|--------|---------|-------------|
| CulturaX | `uonlp/CulturaX` (kz) | ~2.0B |
| HPLT 2.0 | `HPLT/HPLT2.0_cleaned` (kaz_Cyrl) | ~1.8B |
| mC4 | `mc4` (kk) | ~1.0B |
| MADLAD-400 | `google/madlad400` (kk) | ~0.7B |
| mOSCAR | `oscar-corpus/mOSCAR` (kaz) | ~0.5B |
| Wikipedia | `wikipedia` (kk) | ~0.18B × 3× weight |
| multidomain | `kz-transformers/multidomain-kazakh-dataset` | ~0.25B × 2× weight |

Filters (`src/kazllm/data/filters.py`):
- Cyrillic ratio > 0.7
- Length: 50–100K chars
- Language detection: kk > 0.8
- Dedup: exact (SHA-256) + near (MinHash LSH threshold=0.85, 128 perms)

Status: [~] data downloading (HPLT active, 3 auth-free sources queued, CulturaX needs HF login)

**Source status after investigation:**
| Source | Repo | Status |
|--------|------|--------|
| hplt2_kaz | HPLT/HPLT2.0_cleaned (kaz_Cyrl) | Downloading ← |
| c4_kaz | allenai/c4 (kk) — Parquet mc4 replacement | Queued |
| moscar_kaz | oscar-corpus/mOSCAR (kaz_Cyrl) | Queued |
| kaz_wiki | wikimedia/wikipedia (20231101.kk) | Queued |
| multidomain_kaz | kz-transformers/multidomain-kazakh-dataset | Queued |
| culturax_kaz | uonlp/CulturaX (gated) | **Needs: `huggingface-cli login`** |

Auth-free sources alone = ~4.5B tokens (enough to validate pipeline; CulturaX adds ~2B for full 9B)

**To unlock CulturaX:**
1. Accept license: https://huggingface.co/datasets/uonlp/CulturaX
2. `! huggingface-cli login`
3. Re-run `make data` — already-downloaded sources are skipped

### 1.2 Train tokenizer
```bash
make tokenizer   # scripts/train_tokenizer.py
```
- SentencePiece Unigram 50K, `character_coverage=0.9999`, `byte_fallback=True`
- Must cover: Ә Ғ Қ Ң Ө Ұ Ү Һ (Kazakh-specific codepoints)
- Verify fertility: target <2.0 tokens/word on Kazakh validation set

```bash
python -c "from kazllm.tokenizer.fertility import benchmark_tokenizer; benchmark_tokenizer('data/tokenizer.model')"
```

Baselines to beat:
- Llama-3.1: 4.73 (baseline, terrible for Kazakh)
- SozKZ: ~2.0 (target)
- Sherkala: 2.04

Status: [ ] tokenizer trained  [ ] fertility <2.0 verified

### 1.3 Pack token shards
```bash
make pack   # scripts/pack_data.py
```
- Tokenize all filtered data with the trained tokenizer
- Pack into uint16 memmap shards (~500M tokens/shard)
- Output: `data/packed/train/shard_*.npy` + `manifest.json`
- Verify total packed tokens ≥ 9B

Status: [ ] shards written  [ ] manifest verified  [ ] uint16 dtype confirmed

---

## Phase 2 — GCP Infrastructure Setup

### 2.1 GCP project setup
- [ ] Create GCP project, enable billing, enable Compute Engine + GPU quota
- [ ] Request A100 quota for `a2-highgpu-1g` in preferred region (us-central1)
- [ ] Set up Cloud Storage bucket for: data shards, checkpoints, tokenizer
- [ ] Configure preemption handler: auto-resume from latest checkpoint on restart

### 2.2 Instance configuration
Instance: `a2-highgpu-1g` (1× A100 40GB, ~$3.67/hr spot)
```bash
# Setup script for fresh instance
git clone <repo>; cd kazllm0x1
uv sync --all-extras
# Copy data from GCS
gsutil -m cp -r gs://<bucket>/packed/ data/packed/
gsutil cp gs://<bucket>/tokenizer.model data/
```

### 2.3 Checkpoint strategy
- Save every 2K steps (`save_steps: 2000` in `pretrain_500m.yaml`)
- Upload checkpoint to GCS immediately after save
- On preemption: Hydra auto-resumes from `checkpoint_dir` latest checkpoint
- Keep last 3 checkpoints locally; all in GCS

Status: [ ] GCS bucket created  [ ] quota approved  [ ] resume logic tested

---

## Phase 3 — Pre-training (GCP a2-highgpu-1g)

### 3.1 500M pre-training run
```bash
python scripts/train.py model=kaz500m training=pretrain_500m
```

Config summary (`configs/training/pretrain_500m.yaml`):
- 9B tokens total, 138K steps
- Batch: 8 × 4 grad_accum × 1 GPU = 32 seqs × 2048 tokens = 65K tokens/step
- LR: 3e-4, cosine decay, min_lr_ratio=0.1, warmup 2K steps
- bf16, gradient_checkpointing, max_grad_norm=1.0
- Estimated: ~3 days, ~$264 spot

### 3.2 Training health monitoring
W&B dashboard checks every ~12 hours:
- [ ] Loss decreasing smoothly (no spikes >2× baseline)
- [ ] grad_norm < 1.0 at all steps (mHC Sinkhorn guarantee)
- [ ] throughput: target ~100K tokens/s on A100 40GB
- [ ] GPU memory: target <35GB with gradient_checkpointing

Key invariants to watch:
- mHC alpha params should grow slowly from 0 (controlled plasticity)
- Engram table sparsity: sparse grad norms should be nonzero
- No NaN in any metric

### 3.3 Ablation runs (parallel, cheap)
To understand each component's contribution, run 3 ablation variants at 1B token budget (~2% cost):
```bash
# Baseline Llama
python scripts/train.py model=kaz500m training=pretrain_500m \
  model.use_mhc=false model.use_engram=false training.total_tokens=1_000_000_000

# mHC only
python scripts/train.py model=kaz500m training=pretrain_500m \
  model.use_mhc=true model.use_engram=false training.total_tokens=1_000_000_000

# Engram only
python scripts/train.py model=kaz500m training=pretrain_500m \
  model.use_mhc=false model.use_engram=true training.total_tokens=1_000_000_000
```
Compare validation loss at 1B tokens: expected ranking Engram-only ≈ mHC-only < full v2.

Status: [ ] main run started  [ ] 1B checkpoint saved  [ ] 9B training complete

---

## Phase 4 — Evaluation

### 4.1 Primary benchmarks
```bash
make eval   # scripts/evaluate.py
```

| Benchmark | Metric | Shots | KazLLM Target | Baseline |
|-----------|--------|-------|---------------|----------|
| KazMMLU | accuracy | 5 | **>35%** (stretch: >40%) | SozKZ-600M ~30% |
| TUMLU-mini | accuracy | 5 | competitive | — |
| KazQAD | F1 | 0 | competitive | — |
| FLORES-200 kaz→eng | chrF++ | — | competitive | — |
| FLORES-200 kaz→rus | chrF++ | — | competitive | — |

### 4.2 Comparison baseline results (reference)
These are the numbers KazLLM must beat on KazMMLU 5-shot:

| Model | Params | KazMMLU | Notes |
|-------|--------|---------|-------|
| Random baseline | — | ~25% | 4-way MC |
| SozKZ | 600M | ?? (never ran KazMMLU) | Closest size comp; from-scratch |
| Llama-3.1-8B | 8B | ~38% | 16× larger |
| Sherkala-Chat-8B | 8B | 41.4% | SOTA open; continual PT (NOT 47.6% — that's avg across 13 tasks) |
| Llama-3.1-70B | 70B | **55.2%** | SOTA overall; 140× larger |

**Win condition (from-scratch 500M):** >35% KazMMLU — beating SozKZ with fewer params via Engram.
**Win condition (continual PT Qwen-1.5B):** >45% KazMMLU — beating Sherkala-Chat-8B (41.4%) with 5x fewer params.
**Research win:** Engram delta on Kazakh > Engram delta on English (proves morphological memory thesis).

### 4.3 Efficiency comparison
Beyond accuracy, document the efficiency story:

| Metric | Llama-3.1-70B | Sherkala-8B | KazLLM-500M |
|--------|--------------|-------------|-------------|
| Parameters | 70B | 8B | ~500M + ~512M sparse |
| Active FLOPs/token | ~70B | ~8B | ~500M (Engram = 0) |
| Kazakh fertility | 4.73 | 2.04 | <2.0 target |
| KazMMLU (target) | 55.2% | 47.6% | >35% |

Key narrative: KazLLM-500M achieves competitive KazMMLU with 140× fewer active FLOPs than the best
available model, primarily through (1) dedicated tokenizer and (2) Engram offloading morphological
N-gram computation.

### 4.4 Ablation results table
After ablation runs (Phase 3.3), produce:

| Model variant | Val loss @1B | KazMMLU (extrapolated) |
|--------------|-------------|----------------------|
| Baseline (Llama only) | — | — |
| + mHC | — | — |
| + Engram | — | — |
| + mHC + Engram (full v2) | — | — |

Status: [ ] eval harness verified  [ ] KazMMLU results  [ ] efficiency table written

---

## Phase 5 — Post-training (Optional)

### 5.1 Instruction fine-tuning (QLoRA)
```bash
make sft   # scripts/train_sft.py with configs/training/sft_lora.yaml
```
- LoRA rank=16, alpha=32, target all projection layers
- Dataset: Kazakh instruction pairs (to be sourced/translated)
- Can run locally on RTX 2070 (QLoRA 4-bit base)

### 5.2 1B research variant
If 500M results are promising:
- Provision `a2-highgpu-2g` (2× A100 40GB) or `a2-ultragpu-1g` (1× A100 80GB)
- Config: `configs/model/kaz1b.yaml` + `configs/training/pretrain.yaml`
- 9B tokens on 1B model (under-trained by Chinchilla, but informative)

---

## Phase 6 — Documentation and Release

- [ ] Training report: methodology, ablation results, efficiency analysis
- [ ] Model card: intended use, limitations, Kazakh script coverage, fertility numbers
- [ ] Release artifacts: model weights, tokenizer, eval scripts
- [ ] Benchmark reproducibility: exact commands to replicate KazMMLU score

---

## Current Status

| Phase | Status | Blocker |
|-------|--------|---------|
| 0. Local validation | IN PROGRESS | debug model config missing |
| 1. Data pipeline | NOT STARTED | data download |
| 2. GCP setup | NOT STARTED | quota request |
| 3. Pre-training | NOT STARTED | phases 1+2 |
| 4. Evaluation | NOT STARTED | trained checkpoint |
| 5. Post-training | NOT STARTED | phase 3 |

**Next immediate actions:**
1. `make lint && make test` — verify all existing tests pass
2. Create `configs/model/kaz50m_debug.yaml` — RTX 2070-compatible debug config
3. Run `make train-debug` — prove the pipeline end-to-end locally
4. Start data download in parallel: `make data`
5. Begin GCP quota request (can take 24-48h to approve)
