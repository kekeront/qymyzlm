#!/usr/bin/env python3
"""Benchmark baseline models on KazMMLU (Kazakh subset) 5-shot.

Measures: accuracy, tokenizer fertility (tok/word), generation speed (tok/s).

Usage:
    python scripts/benchmark_baselines.py
    python scripts/benchmark_baselines.py --models Qwen/Qwen2.5-0.5B meta-llama/Llama-3.2-1B
    python scripts/benchmark_baselines.py --quick   # 100 questions per model (fast test)
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

# Force unbuffered output for piped commands
os.environ["PYTHONUNBUFFERED"] = "1"
from pathlib import Path

import torch
from datasets import get_dataset_config_names, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Models ────────────────────────────────────────────────────────────────────

DEFAULT_MODELS = [
    # ~0.5B class
    ("Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B"),
    ("Qwen3-0.6B", "Qwen/Qwen3-0.6B"),
    # ~1B class
    ("Gemma3-1B", "google/gemma-3-1b-pt"),
    ("Llama-3.2-1B", "meta-llama/Llama-3.2-1B"),
    ("Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B"),
]

# ── Kazakh text for fertility measurement ─────────────────────────────────────

KAZ_TEXT = (
    "Қазақстан Республикасы — Орталық Азиядағы мемлекет. "
    "Солтүстігінде және батысында Ресеймен, шығысында Қытаймен, "
    "оңтүстігінде Қырғызстанмен, Өзбекстанмен және Түркіменстанмен шектеседі. "
    "Астанасы — Астана қаласы. Қазақстан аумағы жағынан әлемдегі тоғызыншы ірі мемлекет. "
    "Халық саны 20 миллионнан асады. Мемлекеттік тілі — қазақ тілі. "
    "Қазақ тілі — түркі тілдер тобына жатады. Ол агглютинативті тіл болып табылады, "
    "яғни сөздерге жұрнақтар мен жалғаулар қосылып, жаңа мағыналар жасалады."
)


# ── Dataset loading ───────────────────────────────────────────────────────────


def load_kazmmlu_kazakh():
    """Load all KazMMLU Kazakh-language configs (dev + test splits)."""
    configs = get_dataset_config_names("MBZUAI/KazMMLU")
    kaz_configs = [c for c in configs if "in kaz)" in c.lower()]

    all_test, all_dev = [], []

    for config in tqdm(kaz_configs, desc="Loading KazMMLU configs"):
        try:
            ds = load_dataset("MBZUAI/KazMMLU", config)
            for split_name, target in [("test", all_test), ("dev", all_dev)]:
                if split_name in ds:
                    for row in ds[split_name]:
                        row["_config"] = config
                        target.append(row)
        except Exception as e:
            print(f"  ⚠ Skipping {config}: {e}")

    print(f"  → {len(all_test)} test + {len(all_dev)} dev examples loaded")
    return all_test, all_dev


# ── Prompt formatting ─────────────────────────────────────────────────────────


def get_valid_options(row):
    """Return list of valid option letters for a row."""
    options = []
    for letter in "ABCDE":
        val = row.get(f"Option {letter}")
        if val is not None and str(val).strip() not in ("", "nan", "None"):
            options.append(letter)
    return options


def format_example(row, with_answer=False):
    """Format one multiple-choice example."""
    parts = [row["Question"]]
    for letter in get_valid_options(row):
        parts.append(f"{letter}. {row[f'Option {letter}']}")
    parts.append("Жауап:")
    prompt = "\n".join(parts)
    if with_answer:
        prompt += f" {row['Answer Key']}"
    return prompt


def build_fewshot_prompt(dev_examples, test_row, n_shot=5):
    """Build n-shot prompt: dev examples + test question."""
    shots = [format_example(ex, with_answer=True) for ex in dev_examples[:n_shot]]
    shots.append(format_example(test_row, with_answer=False))
    return "\n\n".join(shots)


# ── Evaluation ────────────────────────────────────────────────────────────────


def get_option_token_ids(tokenizer):
    """Pre-compute token IDs for answer letters A–E."""
    token_map = {}
    for letter in "ABCDE":
        ids = tokenizer.encode(f" {letter}", add_special_tokens=False)
        token_map[letter] = ids[-1]
    return token_map


@torch.inference_mode()
def evaluate_kazmmlu(model, tokenizer, test_data, dev_data, n_shot=5, quick=False):
    """Run KazMMLU 5-shot log-likelihood evaluation."""
    dev_by_config = defaultdict(list)
    for row in dev_data:
        dev_by_config[row["_config"]].append(row)

    option_tokens = get_option_token_ids(tokenizer)
    device = next(model.parameters()).device

    correct = 0
    total = 0
    subset = test_data[:100] if quick else test_data

    for row in tqdm(subset, desc="  KazMMLU eval"):
        dev_examples = dev_by_config.get(row["_config"], dev_data[:n_shot])
        prompt = build_fewshot_prompt(dev_examples, row, n_shot=n_shot)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        logits = model(**inputs).logits[0, -1]  # next-token logits

        valid = get_valid_options(row)
        pred = max(valid, key=lambda l: logits[option_tokens[l]].item())

        if pred == row["Answer Key"]:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0.0
    return acc, total


# ── Fertility & speed ─────────────────────────────────────────────────────────


def measure_fertility(tokenizer, text=KAZ_TEXT):
    """Tokens per word on sample Kazakh text."""
    words = text.split()
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens) / len(words)


@torch.inference_mode()
def measure_speed(model, tokenizer, n_tokens=128):
    """Generation speed in tokens/sec."""
    device = next(model.parameters()).device
    prompt = "Қазақстан — Орталық Азиядағы ел."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # warmup
    model.generate(**inputs, max_new_tokens=10, do_sample=False)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(**inputs, max_new_tokens=n_tokens, do_sample=False)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    generated = out.shape[1] - inputs["input_ids"].shape[1]
    return generated / elapsed if elapsed > 0 else 0.0


# ── Output formatting ─────────────────────────────────────────────────────────


def markdown_table(results):
    """Render results as a markdown table."""
    lines = [
        "| Model | Params | KazMMLU 5-shot | Tok/word | Tok/sec |",
        "|-------|--------|---------------|----------|---------|",
    ]

    for r in sorted(results, key=lambda x: x.get("kazmmlu_5shot", 0), reverse=True):
        if "error" in r:
            lines.append(f"| {r['name']} | — | FAILED | — | — |")
        else:
            lines.append(
                f"| **{r['name']}** | {r['params_b']}B "
                f"| {r['kazmmlu_5shot']}% "
                f"| {r['fertility']} "
                f"| {r['speed_tok_s']} |"
            )

    lines.append("| Random baseline | — | 25.0% | — | — |")
    lines.append("| *Sherkala-Chat-8B* | *8B* | *41.4%* | *2.04* | *—* |")
    lines.append("| *Llama-3.1-70B* | *70B* | *55.2%* | *4.73* | *—* |")
    return "\n".join(lines) + "\n"


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Benchmark models on KazMMLU (Kazakh)")
    parser.add_argument("--models", nargs="+", help="HF model IDs (overrides defaults)")
    parser.add_argument("--output", default="evals/baselines", help="Output directory")
    parser.add_argument("--n-shot", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="100 questions only (fast test)")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "auto"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load dataset
    print("\nLoading KazMMLU (Kazakh subset)...")
    test_data, dev_data = load_kazmmlu_kazakh()

    # Determine models
    if args.models:
        models = [(m.split("/")[-1], m) for m in args.models]
    else:
        models = DEFAULT_MODELS

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for name, model_id in models:
        print(f"\n{'=' * 60}")
        print(f" {name}  ({model_id})")
        print(f"{'=' * 60}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Fertility (no model needed)
            fertility = measure_fertility(tokenizer)
            print(f"  Fertility: {fertility:.2f} tok/word")

            # Load model (supports PEFT/LoRA checkpoints)
            dtype = getattr(torch, args.dtype) if args.dtype != "auto" else torch.float16
            adapter_config = Path(model_id) / "adapter_config.json"
            if adapter_config.exists():
                from peft import AutoPeftModelForCausalLM
                model = AutoPeftModelForCausalLM.from_pretrained(
                    model_id, trust_remote_code=True,
                ).to(dtype=dtype, device=device)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, trust_remote_code=True,
                ).to(dtype=dtype, device=device)
            model.eval()
            params_b = sum(p.numel() for p in model.parameters()) / 1e9

            # KazMMLU
            acc, n_q = evaluate_kazmmlu(
                model, tokenizer, test_data, dev_data,
                n_shot=args.n_shot, quick=args.quick,
            )
            print(f"  KazMMLU {args.n_shot}-shot: {acc * 100:.1f}% ({n_q} questions)")

            # Speed
            speed = measure_speed(model, tokenizer)
            print(f"  Speed: {speed:.1f} tok/s")

            result = {
                "name": name,
                "model_id": model_id,
                "params_b": round(params_b, 2),
                "kazmmlu_5shot": round(acc * 100, 1),
                "n_questions": n_q,
                "fertility": round(fertility, 2),
                "speed_tok_s": round(speed, 1),
            }
            results.append(result)

            # Save incrementally (resume-friendly)
            with open(output_dir / "baselines.json", "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"name": name, "model_id": model_id, "error": str(e)})

    # Print results
    print(f"\n{'=' * 60}")
    print(" BASELINE RESULTS")
    print(f"{'=' * 60}\n")
    table = markdown_table(results)
    print(table)

    # Save markdown
    with open(output_dir / "baselines.md", "w") as f:
        f.write("# KazMMLU Baseline Results\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d')}\n")
        f.write(f"Device: {device}")
        if device == "cuda":
            f.write(f" ({torch.cuda.get_device_name()})")
        f.write(f"\nEval: {args.n_shot}-shot, Kazakh subset only\n\n")
        f.write(table)

    print(f"Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
