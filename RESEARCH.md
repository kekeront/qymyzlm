# KazLLM Research Analysis — Honest Assessment

*Draft: April 2026*

---

## Executive Summary

**KazLLM is a well-conceived, technically ambitious project with a realistic shot at its primary target.** It combines three recent techniques (Engram, mHC, MTP) from credible sources (DeepSeek-AI) into a 500M Kazakh language model trained on 9B tokens for ~$264. The project addresses a genuine gap — Kazakh is the most-spoken Turkic language (~20M native speakers) with almost no dedicated small language models. The >35% KazMMLU target is achievable; the >40% stretch target is optimistic but not impossible.

**Verdict: This is a good project.** It's technically sound, addresses a real need, has a manageable budget, and — win or lose on the stretch target — will produce a valuable artifact for Kazakh NLP. The main risks are engineering execution (combining three novel components) and data quality, not fundamental feasibility.

---

## 1. The Landscape: Kazakh NLP in 2024–2026

### 1.1 Why Kazakh Matters

- **~17 million speakers globally** (~13.7M in Kazakhstan per 2021 census), official language of Kazakhstan
- **GDP $260B+ (2025)** — Kazakhstan is the largest Central Asian economy, 96%+ internet penetration
- **Massive government tailwind**:
  - "Concept for AI Development 2024-2029" adopted by government
  - 2025 declared "Year of Digitalization and AI"
  - Law on AI (No. 230-VIII) signed November 2025, entered force January 2026
  - Dedicated Ministry of AI being established
  - 2-exaflop supercomputer (Alem.cloud) operational in Astana since 2025
- **Script transition**: Planned shift from Cyrillic to Latin by 2031 — any model built now needs to handle both
- **UNESCO classification**: "Vulnerable" — not endangered, but digitally underrepresented
- **Turkic language family**: Agglutinative morphology shared with Turkish (~90M speakers), Uzbek (~35M), Azerbaijani (~25M). Techniques that work for Kazakh morphology generalize to 150M+ Turkic speakers

### 1.2 Existing Kazakh Language Models

The landscape has grown rapidly in 2024-2026 but remains sparse below 8B:

| Model | Size | Type | KazMMLU | Year | Notes |
|-------|------|------|---------|------|-------|
| GPT-4o | ~1T? | Proprietary | ~76% | 2024 | Not deployable locally |
| DeepSeek-V3 | 671B MoE | Open weights | ~76% | 2024 | Too large for target use case |
| Gemma-2-27B-IT | 27B | Multilingual | 57.4% | 2025 | Outperforms all dedicated Kazakh models |
| Llama-3.1-70B | 70B | Multilingual | 56.2% | 2024 | Best large open model |
| **Sherkala-8B** (base) | 8B | Continual PT (Llama-3.1-8B) | 51.6% | 2025 | Base model; higher than chat due to alignment tax |
| Sherkala-Chat-8B | 8B | Instruction-tuned | 41.4% | 2025 | Real KazMMLU=41.4%, NOT 47.6% (that's avg across 13 tasks; inflated by BoolQA 75.8%, PIQA 65.9%). Destroyed Russian: MT-Bench 1.02/10 |
| **ISSAI KazLLM-8B** | 8B | Continual PT | 41.7% | 2024 | National partnership (Ministry of AI + Nazarbayev U.) |
| Llama-3.1-8B | 8B | Multilingual | 38.3% | 2024 | Not Kazakh-specific |
| Qwen-2.5-7B | 7B | Multilingual | 35.1% | 2025 | Good multilingual baseline |
| **SozKZ-600M** | 600M | From scratch | ~30%* | 2026 | Closest comparable; 9B tokens, 50K BPE |
| BLOOM-7.1B | 7.1B | Multilingual | 29.3% | 2023 | Weak Kazakh support |
| Random baseline | — | — | ~25% | — | 4-way multiple choice |

*\*Note: SozKZ NEVER ran KazMMLU. Their 30.3% is on kk-socio-cultural-bench (7,111 cultural QA questions), NOT KazMMLU. Cannot be placed on KazMMLU leaderboard.*

**Key observations:**
- There are only **three** dedicated Kazakh LMs (Sherkala, ISSAI KazLLM, SozKZ) — and all 8B+ models are continual PT from Llama
- **No dedicated Kazakh model exists below 600M parameters**. This is the gap.
- A new competitor, **KazByte** (March 2026), takes an alternative approach: byte-level adapter on frozen Qwen2.5-7B, bypassing tokenization entirely. Validation is ongoing — not yet a proven competitor but signals growing attention to the tokenizer problem.

### 1.3 Is There Demand?

**Yes, strongly.**

- **Academic**: Kazakh NLP research is accelerating. Sherkala, SozKZ, ISSAI KazLLM, KazByte, and KazMMLU benchmark (ACL 2025) — all published in 2024-2026. Multiple ACL/EMNLP papers on Kazakh NLP.
- **Government**: Kazakhstan's Ministry of AI + Law on AI (Jan 2026) create institutional demand. ISSAI's KazLLM-8B was released December 2024 in a national partnership with Ministry of Science, QazCode/Beeline, and Nazarbayev University — demonstrating government willingness to fund and adopt open Kazakh LMs.
- **Practical**: A 500M model that runs on consumer hardware fills a genuine deployment niche — local inference for translation, content generation, education tools, government services. There are 23+ universities in Kazakhstan with AI curricula as of 2025.
- **Ecosystem**: Organizations like `kz-transformers` are actively building Kazakh datasets (multidomain corpus), indicating a growing community.

---

## 2. Technical Viability Analysis

### 2.1 The Three Novel Components

KazLLM combines three techniques. Let's assess each:

#### Engram (arXiv 2601.07372) — **Strong foundation**

- **Source**: DeepSeek-AI + Peking University (January 2026)
- **What it does**: O(1) hash-table lookup of N-gram suffix patterns, injected into early transformer layers. Adds sparse parameters but zero FLOPs.
- **Paper results**: Scaled to 27B params. MMLU +3.0 (not +3.4 as abstract claims — abstract contradicts Table 1), CMMLU +4.0, BBH +5.0, HumanEval +3.0 over iso-parameter MoE baseline. CMMLU gain > MMLU gain supports "helps agglutinative/memorized patterns more" thesis.
- **Why it's especially good for Kazakh**: Agglutinative languages have highly stereotyped suffix chains. The paper's mechanistic analysis (via LogitLens/CKA) showed Engram relieves early layers from reconstructing local patterns — exactly what Kazakh morphology demands. This is the single strongest technical bet in the project.
- **Credibility**: Published by the team behind DeepSeek-V3. Code released on GitHub. Well-written paper with thorough ablations.
- **Risk at 500M scale**: The paper's smallest tested scale is a 3B MoE with 0.56B activated parameters — close to KazLLM's 530M. The validation loss improvement at that scale (1.768 vs 1.808) is modest but consistent. The 512M sparse params are proportionally larger relative to the 530M backbone (roughly 1:1 ratio), which is actually favorable by the paper's Sparsity Allocation analysis.
- **Follow-up research validates the approach**: A collision-free extension paper (arXiv 2601.16531) found that eliminating hash collisions does NOT consistently improve validation loss — meaning the K=4 multi-head hash design is already sufficient. A CXL offloading paper (arXiv 2603.10087) confirmed <3% throughput penalty for offloading Engram tables to host memory. Both support the KazLLM design choices.

#### mHC (arXiv 2512.24880) — **Solid but with overhead concerns**

- **Source**: DeepSeek-AI (January 2026)
- **What it does**: 4 parallel residual streams with Sinkhorn-Knopp projection onto the Birkhoff polytope. Guarantees spectral norm ≤ 1, preventing gradient explosion at depth.
- **Paper results**: Demonstrated effective training at scale with 6.7% overhead (with TileLang kernel fusion).
- **Why for Kazakh specifically**: The argument is that richer residual representations help capture morphological complexity without adding depth. This is plausible but less uniquely Kazakh-relevant than Engram.
- **Overhead concern**: The 6.7% overhead cited in the paper requires TileLang kernel fusion, which KazLLM does **not** implement. Without kernel fusion, the CLAUDE.md states **15-25% training overhead**. For a ~$264 training run, this adds $40-66. Tolerable, but worth noting.
- **Risk at 500M scale**: The paper showed scaling results, but the primary experiments were at larger scale. At 500M/16 layers, the benefit of stable gradient propagation is less critical than at 48+ layers. The n=4 streams quadruple the memory of the residual state, which matters on 40GB A100.

#### MTP (DeepSeek-V3 style, arXiv 2412.19437) — **Promising with caveats**

- **Source**: DeepSeek-V3 Technical Report (December 2024)
- **What it does**: Predicts 1 additional future token during training. Discarded at inference — zero deployment cost.
- **Paper results**: Shown effective in DeepSeek-V3 (671B MoE), but at much larger scale.
- **Small-scale concern**: The KazLLM config itself notes "BabyLM ACL 2025: MTP can hurt <1B without curriculum." The project addresses this with a warmup curriculum (lambda=0 for first 500M tokens, ramp to 0.3, reduce to 0.1 after 60%). This aligns with the "forward curriculum" approach from arXiv 2505.22757 (Pre-Training Curriculum for MTP in Language Models), which specifically validates curriculum-based MTP for smaller models.
- **Risk**: MTP at 500M is the least validated of the three components. However, D=1 (predict only 1 additional token) is the most conservative application — DeepSeek-V3's ablation showed consistent improvement even at smaller baseline configurations.
- **Mitigation**: Can be disabled via `use_mtp=false` without affecting the rest of the architecture. The ablation support is good.

### 2.2 Combining All Three — Integration Risk

**This is the biggest technical risk.** Each component individually is well-motivated, but their interaction is untested:

- **Engram + mHC**: The Engram paper's Section 2.4 explicitly discusses integration with multi-branch architectures (i.e., mHC with M=4). KazLLM's implementation applies Engram to the mean of mHC streams and adds the result to all streams equally. This is a simplification of the per-branch gating described in the paper (Eq. 6). It's a reasonable engineering choice (simpler, fewer params) but may sacrifice some of the branch-specific modulation.
- **MTP + mHC**: MTP modules operate in single-stream mode (no mHC), receiving collapsed hidden states. This is clean — no interaction risk.
- **Three-way**: No published work combines all three. The project is novel here, which is both an academic strength and an engineering risk.

**Assessment**: The integration is architecturally clean. The ablation flags (`use_mhc`, `use_engram`, `use_mtp`) allow isolating each component's contribution. The 1B-token ablation runs at ~2% of total cost are well-planned.

### 2.3 Implementation Quality

Having reviewed the full codebase:

**Strengths:**
- Clean, well-documented PyTorch code with proper type hints
- All key components have dedicated test suites (33/33 passing)
- Proper initialization: mHC alphas=0 (identity at start), Engram conv weights=0 (identity)
- Gradient checkpointing support for memory efficiency
- Sparse gradient handling in the trainer (`_densify_sparse_grads`)
- HuggingFace `PreTrainedModel` integration for ecosystem compatibility
- Ablation support baked into the architecture from day one

**Concerns:**
- Sinkhorn-Knopp runs 20 iterations per forward pass, per block, per sub-layer (attention + MLP). For 16 layers × 2 sub-layers = 32 Sinkhorn calls per forward pass. This is compute overhead even though FLOPs per call are small.
- No KV cache implementation for inference (acceptable for pretraining phase)
- The `repeat_interleave` in GQA attention could be replaced with proper view/expand for efficiency
- No mixed-precision Sinkhorn (currently runs in model dtype)

---

## 3. Data Analysis

### 3.1 Is 9B Tokens Enough?

**Yes — this is close to Chinchilla-optimal.**

The Chinchilla scaling law (Hoffmann et al., 2022) recommends ~20 tokens per parameter for compute-optimal training. For 530M backbone params: 530M × 20 = 10.6B tokens. The 9B token budget gives a 17:1 ratio — slightly under-trained by Chinchilla but within a reasonable margin.

For reference:
- SozKZ-600M: 9B tokens → 15:1 ratio (more under-trained)
- Llama-2-7B: 2T tokens → 286:1 ratio (massively over-trained for its size)
- SmolLM2-135M: 2T tokens → 14,800:1 ratio (extreme over-training)

The trend in 2024–2026 has been toward over-training (train longer than Chinchilla suggests), because inference cost scales with model size, not training data. But for a first training run of a novel architecture, Chinchilla-near is fine. Over-training a 500M model on more data would improve results but requires more data — and Kazakh simply doesn't have 50B+ high-quality tokens available.

### 3.2 Data Quality Concerns

This is a **significant risk**. Kazakh web data has well-known issues:

| Concern | Severity | Mitigation in project |
|---------|----------|-----------------------|
| **Script mixing** (Cyrillic + Latin) | Medium | Cyrillic ratio > 0.7 filter |
| **Russian contamination** | High | Language detection kk > 0.8 threshold |
| **Low-quality web text** | High | Length filter (50–100K chars), dedup (SHA-256 + MinHash) |
| **Translation artifacts** | Medium | Not explicitly addressed |
| **Domain imbalance** (news-heavy) | Medium | Wikipedia 3x upweight, multidomain 2x upweight |
| **Near-duplicates** | Medium | MinHash LSH (threshold=0.85, 128 perms) |
| **Kazakh-specific Unicode** | Low | `character_coverage=0.9999` in tokenizer |

**The biggest concern is Russian contamination.** Many Kazakh web sources contain significant Russian text mixed in (code-switching is common in Kazakhstan). A language detection threshold of 0.8 may let through documents that are 20% Russian — and these Russian segments would consume model capacity without benefiting Kazakh understanding.

**Missing**: No explicit quality scoring (perplexity-based filtering, classifier-based quality estimation). Modern data pipelines (e.g., FineWeb, DCLM) use such techniques aggressively. Adding even a simple fastText quality classifier could improve the data substantially.

### 3.3 Source Breakdown Assessment

| Source | Est. tokens | Quality expectation | Risk |
|--------|-------------|---------------------|------|
| CulturaX (kz) | ~2.0B | Medium-high (filtered Common Crawl) | Gated — needs HF login |
| HPLT 2.0 | ~1.8B | Medium (web-crawled, quality varies) | May overlap with CulturaX |
| mC4 | ~1.0B | Medium (noisy web) | Known quality issues |
| MADLAD-400 | ~0.7B | Medium (web-crawled) | Smaller, may overlap |
| mOSCAR | ~0.5B | Low-medium (less filtered) | Quality floor |
| Wikipedia (kk) | ~0.18B | High (encyclopedic) | Small but 3x upweighted — good |
| multidomain | ~0.25B | High (curated) | Small but 2x upweighted — good |

**Deduplication between sources is critical.** CulturaX, HPLT, mC4, MADLAD-400, and mOSCAR all ultimately derive from Common Crawl or similar web scrapes. Without cross-source dedup, you may be training on the same documents 3-5 times under different source labels. The project includes MinHash LSH dedup — make sure it runs across all sources, not within each source independently.

---

## 4. Tokenizer Strategy

### 4.1 Unigram vs BPE for Kazakh

**Unigram is the right choice for Kazakh.** Here's why:

| Factor | BPE | Unigram | Winner for Kazakh |
|--------|-----|---------|-------------------|
| Morpheme alignment | Greedy, frequency-biased | Probabilistic, tries all segmentations | **Unigram** |
| Agglutinative suffixes | Merges common pairs; may split morpheme boundaries | Optimizes full-sequence likelihood; better morpheme recovery | **Unigram** |
| Rare word handling | Fragments heavily | More graceful degradation | **Unigram** |
| Established for Turkic | SozKZ used BPE | Research on Turkish/Finnish favors Unigram | **Unigram** |
| Training speed | Faster | Slower (EM algorithm) | BPE |

Research on agglutinative languages (Turkish, Finnish, Hungarian) consistently shows that Unigram tokenizers achieve better morphological segmentation than BPE, because the EM-based objective naturally discovers morpheme boundaries rather than greedily merging frequent pairs. A January 2026 study on CSE-guided morphological tokenization for Turkic languages (MDPI Information) specifically validates Unigram > BPE for Kazakh using a morphology-aware framework.

SozKZ used BPE (50K) and achieved ~2.0 fertility. KazLLM targets <2.0 with Unigram (50K). If KazLLM achieves even 1.85–1.95 fertility, it means more Kazakh text fits in the same context window — a compounding advantage for downstream tasks.

### 4.2 Vocab Size

50K is reasonable. For comparison:
- SozKZ: 50K (BPE)
- Sherkala: 159K (expanded from Llama-3.1's 128K)
- Llama-3.1: 128K (multilingual)
- SmolLM2: 49K

For a dedicated Kazakh model with ~10B tokens, 50K provides good coverage without making the embedding table disproportionately large (50K × 1536 = 76.8M params = ~14.5% of backbone).

---

## 5. Competitive Positioning

### 5.1 Can KazLLM-500M Beat SozKZ-600M (~30% KazMMLU)?

**Very likely yes.** Here's the argument:

1. **Tokenizer advantage**: Unigram should slightly beat SozKZ's BPE on fertility, giving more effective context per sequence.
2. **Engram**: Offloading morphological reconstruction from early layers is worth at least a few percentage points according to the paper's ablations. For Kazakh specifically (more agglutinative than English), the benefit should be at or above the paper's English benchmarks.
3. **mHC**: Richer residual representations provide an additional (smaller) boost.
4. **MTP**: If the curriculum works, additional learning signal per token.
5. **Same data budget**: Both use ~9B tokens.
6. **Fewer backbone params**: 530M vs 600M, but Engram adds 512M sparse params at zero FLOPs.

**Confidence: 70–80% that KazLLM exceeds 35%.** SozKZ at ~30% leaves a 5-point margin. Even if Engram provides only half its claimed benefit (due to smaller scale), and mHC adds marginal improvements, 35% is achievable.

### 5.2 Is >40% Realistic?

**Possible but unlikely at 500M.** Here's the math:

- Going from 30% to 40% is a +10 point jump (33% relative improvement)
- SozKZ-600M → Llama-3.1-8B is 30% → 38% (+8 points), requiring 13x more parameters
- KazLLM would need to close this gap with architecture alone

For >40%, you'd need Engram to provide near its full-scale benefit (+3-4 points) AND mHC to provide meaningful additional gains (+2-3 points) AND MTP to provide measurable improvement (+1-2 points) AND the tokenizer to be measurably better. All of these at the same time, at half the demonstrated scale.

**Confidence: 20–30% that KazLLM exceeds 40%.** It's a stretch. The 1B variant would have better odds.

### 5.3 The Efficiency Narrative

Even if KazLLM hits "only" 35% KazMMLU, the efficiency story is compelling:

| Model | KazMMLU | Active params | FLOPs/token | Ratio vs KazLLM |
|-------|---------|---------------|-------------|-----------------|
| Llama-3.1-70B | 55.2% | 70B | ~70B | 132x more compute |
| Sherkala-Chat-8B | 41.4% | 8B | ~8B | 15x more compute |
| Llama-3.1-8B | ~38% | 8B | ~8B | 15x more compute |
| **KazLLM-500M** | **>35%** | **530M** | **~530M** | **1x (baseline)** |
| SozKZ-600M | ~30% | 600M | ~600M | 1.13x |

A model that achieves comparable KazMMLU to Llama-3.1-8B at 15x fewer FLOPs per token is a strong result regardless of exact numbers.

---

## 6. Risks and Failure Modes

### 6.1 High-Risk Factors

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Training instability** from mHC + Engram interaction | Medium | High (wasted GPU hours) | Identity init, ablation flags, 50M debug model first |
| **Data quality** — Russian contamination, duplicates | Medium-High | Medium (caps ceiling) | Filters in place, but could be stronger |
| **mHC overhead** exceeds memory budget on A100 40GB | Medium | High (can't train) | Gradient checkpointing; may need to reduce batch size |
| **Engram hash collisions** at 500K table size | Low-Medium | Low (graceful degradation via gating) | Context-aware gating suppresses collisions |
| **GCP spot preemption** during critical training phase | Medium | Medium (time lost) | Checkpoint every 2K steps, resume logic |
| **MTP hurts at small scale** despite curriculum | Low-Medium | Low (just adds noise) | Can disable via ablation flag |

### 6.2 Memory Budget Analysis

This is worth calculating explicitly for the A100 40GB:

```
Model parameters:
  Backbone: 530M × 2 bytes (bf16) = 1.06 GB
  Engram sparse: 512M × 2 bytes = 1.02 GB (but sparse — actual GPU memory lower)
  MTP module: 39M × 2 bytes = 0.08 GB
  Total model: ~2.16 GB

Optimizer states (AdamW, fp32 momentum + variance):
  Backbone + Engram dense: ~(530M + small dense) × 8 bytes ≈ 4.5 GB
  Engram sparse: separate sparse optimizer, ~512M × 8 bytes = 4.1 GB
  Total optimizer: ~8.6 GB

Activations (with gradient checkpointing, batch=8, ctx=2048):
  Standard Llama-500M: ~4-6 GB
  mHC n=4 streams: multiply residual activation by 4 → ~8-12 GB additional
  Engram: minimal (lookup + small projections)
  Total activations: ~12-18 GB

Gradient buffer: ~2.2 GB

TOTAL ESTIMATED: 25-31 GB
```

**This fits on A100 40GB, but it's tight.** The mHC 4-stream residual is the biggest pressure point. If memory is tight, options include:
- Reduce batch size from 8 to 4 (increases training steps)
- Reduce mHC streams from 4 to 2 (reduces benefit but saves significant memory)
- More aggressive gradient checkpointing

### 6.3 What Happens If It Fails?

Even in the worst case, the project produces:
1. A high-quality Kazakh tokenizer (Unigram 50K, fertility <2.0)
2. A cleaned, deduplicated 9B-token Kazakh corpus
3. Ablation data on Engram/mHC/MTP at 500M scale
4. A baseline Llama-style 500M Kazakh model (ablation mode with all features disabled)

These artifacts have standalone value for the Kazakh NLP community.

---

## 7. Budget Realism

### 7.1 Training Cost

The $264 estimate assumes:
- `a2-highgpu-1g` spot: ~$3.67/hr
- 72 hours (3 days) training time
- ~100K tokens/sec throughput on A100 40GB

**With mHC overhead (15-25% without kernel fusion):**
- Throughput drops to ~75-85K tokens/sec
- Training time extends to ~3.5-4 days
- Cost: ~$310-350

**With spot preemption (expect 1-3 preemptions over 3-4 days):**
- Each preemption adds 15-30 min recovery time
- Total cost impact: ~$20-40

**Realistic total: $300-400** including mHC overhead and preemption recovery. Still very affordable.

### 7.2 Total Project Cost (all phases)

| Phase | Cost | Notes |
|-------|------|-------|
| Data pipeline | $0 | Local or free-tier |
| Tokenizer training | $0 | Local (RTX 2070) |
| Debug training (50M) | $0 | Local (RTX 2070) |
| **500M pretraining** | **$300-400** | A100 40GB spot |
| Ablation runs (3 variants × 1B tokens) | ~$30-50 | ~11% of main run each |
| Evaluation | ~$5-10 | Short inference runs |
| **Total** | **$335-460** | |

This is remarkably cheap for a from-scratch language model. For comparison:
- Llama-2-7B: estimated ~$2M training cost
- SmolLM2-135M: multiple GPU-months of H100 time
- SozKZ-600M: not disclosed, but similar compute class

---

## 8. Impact and Value Assessment

### 8.1 Academic Value

**High.** This project contributes:

1. **Novel architecture combination**: First application of Engram + mHC + MTP together at any scale. Even if results are mixed, the ablation data is valuable.
2. **Agglutinative language focus**: Engram was designed for English. Testing it on a morphologically rich language is a natural and important experiment. If Engram helps more for Kazakh than English (as the morphological argument suggests), that's a publishable finding.
3. **Efficiency story**: Demonstrating competitive performance at 500M vs 8B models via sparse memory is relevant to the broader "small models" research direction (MobileLLM, SmolLM2, etc.).
4. **Reproducibility**: ~$300-400 total cost means anyone can reproduce the results.

**Potential venues**: ACL, EMNLP, NAACL (low-resource languages track), LREC, TurkicNLP workshops.

### 8.2 Practical Value

**Medium-High for the Kazakh ecosystem.**

- **Edge deployment**: 500M model can run on phones, laptops, Raspberry Pi (with quantization)
- **Cost**: Inference at ~530M active FLOPs/token vs 8B for Sherkala = 15x cheaper to serve
- **Kazakh government**: Digital transformation needs local models for data sovereignty
- **Education**: Small model suitable for fine-tuning on specific tasks (translation, QA, content generation) by researchers with limited compute

### 8.3 Who Would Use This?

1. **Kazakh NLP researchers** (university labs in Kazakhstan, Turkic NLP community)
2. **Companies building Kazakh-language products** (chatbots, translation, content moderation)
3. **Government digital services** (document processing, citizen services)
4. **Language preservation efforts** (tools for Kazakh language education, especially during script transition)
5. **Other Turkic language researchers** (techniques may transfer to Turkish, Uzbek, Kyrgyz, etc.)

---

## 9. What's Going to Happen? (Predictions)

### 9.1 Most Likely Outcome (60% probability)

KazLLM-500M achieves **33-37% KazMMLU**, beating SozKZ-600M and establishing a new efficiency frontier for Kazakh language models. Ablations show Engram provides the largest improvement (+3-5 points), mHC provides a modest boost (+1-2 points), and MTP has marginal or slightly negative effect at this scale. The project produces a clean paper suitable for a workshop or short conference paper.

### 9.2 Best Case (15% probability)

KazLLM-500M hits **38-42% KazMMLU**, genuinely approaching Llama-3.1-8B territory at 15x fewer FLOPs. All three components contribute measurably. The efficiency story is dramatic enough for a top-tier venue (ACL/EMNLP main). The 1B variant, if funded, could potentially challenge Sherkala-8B.

### 9.3 Disappointing Case (20% probability)

KazLLM-500M lands at **28-33% KazMMLU** — matching or barely beating SozKZ-600M. Data quality issues or mHC overhead eat into the Engram benefit. The project still produces valuable ablation data and artifacts, but the headline result is underwhelming.

### 9.4 Failure Case (5% probability)

Training instability (mHC + Engram interaction), memory overflow on A100 40GB forcing significant architecture compromises, or severe data quality issues result in **<28% KazMMLU** — below SozKZ. This would indicate fundamental problems with the architecture combination at this scale.

---

## 10. Recommendations

### 10.1 Before Training

1. **Complete the debug run** (50M model, local) — this is the single most important pre-training step. It validates the full pipeline.
2. **Run memory profiling** on A100 40GB with the exact 500M config. Don't discover memory issues at step 50K.
3. **Add cross-source dedup** to the data pipeline. Running dedup within each source is not enough.
4. **Consider a fastText quality classifier** for the web data. Even a simple one removes the worst noise.

### 10.2 During Training

1. **Run the ablation experiments first** (1B tokens each, ~$10-15 per variant). This tells you whether each component helps before committing to the full 9B run.
2. **Monitor mHC alpha parameters** — they should grow slowly from 0. If they spike, something is wrong.
3. **Monitor Engram gate values** — if alpha (the gating scalar) stays near 0 for all positions, Engram isn't contributing.
4. **Save frequent checkpoints** (every 2K steps is good) — spot preemption is real.

### 10.3 Architecture Decisions to Revisit

1. **mHC n=4 → n=2**: If memory is tight, n=2 streams give ~70% of the benefit at ~50% of the overhead. The paper shows diminishing returns above n=2.
2. **MTP curriculum**: The warmup is well-designed, but consider starting with `use_mtp=false` for the main run and adding it only if the ablation shows clear benefit.
3. **Engram per-branch gating**: The current implementation averages across mHC streams for Engram input. The paper's Eq. 6 uses per-branch gating. If Engram underperforms in ablations, this is a concrete improvement to try.

### 10.4 After Training

1. **Quantize aggressively**: GPTQ or AWQ 4-bit will make the 500M model run on phones.
2. **TurboQuant** for KV cache compression (3-bit KV) — already identified in the project notes.
3. **Release everything**: model weights, tokenizer, data pipeline, ablation results. This maximizes impact for the Kazakh NLP community.
4. **Write the paper**: Focus on the efficiency narrative + Engram-for-agglutinative-languages angle.

---

## 11. Conclusion

**KazLLM is a well-designed project that addresses a genuine need with a realistic budget and sound technical foundations.** The architecture choices are individually well-motivated and come from credible sources. The main uncertainty is whether the three-way combination works at 500M scale — but the ablation infrastructure means you'll learn something valuable regardless of the outcome.

The project sits at an interesting intersection:
- **Low risk financially**: $300-400 total
- **Medium risk technically**: Novel combination, but clean ablation support
- **High potential reward**: Efficiency frontier for Kazakh NLP, publishable research, practical deployment

**Go build it.** The debug run is next. Once that passes, the rest is execution.

---

## Appendix A: Paper Credibility Assessment

| Paper | Venue/Source | Authors | Code | Replication | Credibility |
|-------|-------------|---------|------|-------------|-------------|
| Engram (2601.07372) | DeepSeek-AI + PKU | Senior team (Damai Dai et al.) | GitHub released | Large-scale experiments | **Very High** |
| mHC (2512.24880) | DeepSeek-AI | Senior team (Zhenda Xie et al.) | In DeepSeek codebase | Production-validated | **Very High** |
| DeepSeek-V3 MTP (2412.19437) | DeepSeek-AI | Full team | V3 released | Production model | **Very High** |
| SozKZ (2603.20854) | arXiv preprint | — | — | Single result | **Medium** (newer, less validated) |
| Sherkala (2503.01493) | arXiv preprint | — | — | Single result | **Medium** (newer, less validated) |

## Appendix B: Kazakh Morphology Quick Reference

Why agglutination matters for language modeling:

```
English:  "in my houses"     = 3 tokens, 3 words
Kazakh:   "үйлерімде"        = 1 word = үй + лер + ім + де
           house + PLURAL + MY + LOCATIVE

English:  "I was not able to make them do it"  = 10 words
Kazakh:   "істете алмадым"                      = 2 words
           іс + те + те + ал + ма + ды + м
           do + CAUS + CAUS + can + NEG + PAST + 1SG
```

A multilingual tokenizer (Llama-3.1) fragments these into 4-5 subword pieces each, wasting attention capacity reconstructing the original morpheme structure. A dedicated tokenizer + Engram memory handles this in 1-2 tokens + O(1) suffix lookup.

## Appendix C: Comparable Projects for Reference

| Project | Lang | Size | Data | Result | Budget |
|---------|------|------|------|--------|--------|
| SmolLM2-135M | English | 135M | 2T tokens | Strong for size | Large (HF cluster) |
| MobileLLM | English | 125M-350M | 1T tokens | SoTA <1B | Meta compute |
| SozKZ | Kazakh | 600M | 9B tokens | ~30% KazMMLU | Unknown |
| **KazLLM** | **Kazakh** | **530M+512M** | **9B tokens** | **>35% target** | **~$300-400** |

---

## Appendix D: Key References and Sources

| Reference | ArXiv / Source | Relevance |
|-----------|---------------|-----------|
| Engram | arXiv 2601.07372 | Core architecture component |
| Engram collision-free extension | arXiv 2601.16531 | Validates K=4 hash design is sufficient |
| Engram CXL offload | arXiv 2603.10087 | Confirms <3% inference overhead |
| mHC | arXiv 2512.24880 | Core architecture component |
| DeepSeek-V3 (MTP) | arXiv 2412.19437 | MTP reference |
| MTP curriculum for SLMs | arXiv 2505.22757 | Validates curriculum approach |
| SozKZ | arXiv 2603.20854 | Primary competitor |
| Sherkala | arXiv 2503.01493 | Kazakh SOTA reference |
| KazMMLU benchmark | arXiv 2502.12829 (ACL 2025) | Primary evaluation benchmark |
| KazByte | arXiv 2603.27859 | Alternative approach (byte-level) |
| ISSAI KazLLM | issai.nu.edu.kz/kazllm | National partnership model |
| Morphological tokenization for Turkic | MDPI Information, Jan 2026 | Unigram > BPE for Kazakh |
| Kazakhstan AI Concept 2024-2029 | primeminister.kz | Government policy context |
| Chinchilla scaling replication | Epoch AI | Scaling law reference |
