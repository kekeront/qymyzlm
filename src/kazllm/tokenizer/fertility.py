"""Measure tokenizer fertility (tokens per whitespace-delimited word)."""

import json
import logging
from pathlib import Path

import sentencepiece as spm

log = logging.getLogger(__name__)

# Llama-3.1 baseline fertility for Kazakh (measured on FLORES-200 kaz dev)
LLAMA31_KAZ_FERTILITY = 4.73
SOZKZ_50K_BPE_FERTILITY = 2.0  # SozKZ reference
SHERKALA_EXPANDED_FERTILITY = 2.04  # Sherkala reference


def compute_fertility(
    sp_model_path: str | Path,
    texts: list[str],
) -> float:
    """Compute average fertility (tokens/word) on a list of texts."""
    sp = spm.SentencePieceProcessor(model_file=str(sp_model_path))
    total_tokens = 0
    total_words = 0
    for text in texts:
        words = text.split()
        if not words:
            continue
        tokens = sp.encode(text)
        total_tokens += len(tokens)
        total_words += len(words)
    return total_tokens / max(total_words, 1)


def benchmark_tokenizer(
    sp_model_path: str | Path,
    test_texts: list[str],
    output_path: str | Path | None = None,
) -> dict:
    """Run full fertility benchmark and return results dict."""
    fertility = compute_fertility(sp_model_path, test_texts)

    results = {
        "model_path": str(sp_model_path),
        "num_test_texts": len(test_texts),
        "kazakh_fertility": fertility,
        "vs_llama31_baseline": fertility / LLAMA31_KAZ_FERTILITY,
        "reduction_vs_llama31_pct": (1 - fertility / LLAMA31_KAZ_FERTILITY) * 100,
        "target_met": fertility < 2.5,
        "excellent": fertility < 2.0,
    }

    log.info(f"Fertility: {fertility:.3f} tokens/word")
    log.info(f"Reduction vs Llama-3.1: {results['reduction_vs_llama31_pct']:.1f}%")
    log.info(
        f"Baselines — SozKZ: {SOZKZ_50K_BPE_FERTILITY}, Sherkala: {SHERKALA_EXPANDED_FERTILITY}"
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    return results
