#!/usr/bin/env python3
"""Benchmark baseline models on KazMMLU (Kazakh subset), 3-shot by default.

Measures: accuracy, tokenizer fertility (tok/word), generation speed (tok/s).

Shot count: the KazMMLU dev split holds only 3 exemplars per subject, so 3-shot
is the honest maximum with subject-matched shots (requesting more silently
reuses the same 3 — the effective count is logged and stored in the results).

Prompts that exceed --max-length are truncated from the LEFT by dropping the
earliest few-shot exemplars; the test question and the "Жауап:" answer cue are
never cut. Truncation events are counted and reported.

Usage:
    python scripts/benchmark_baselines.py
    python scripts/benchmark_baselines.py --models Qwen/Qwen2.5-0.5B meta-llama/Llama-3.2-1B
    python scripts/benchmark_baselines.py --quick   # 100 questions per model (fast test)
"""

import argparse
import json
import os
import subprocess
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


def build_fewshot_prompt(dev_examples, test_row, n_shot=3):
    """Build n-shot prompt: dev examples + test question."""
    shots = [format_example(ex, with_answer=True) for ex in dev_examples[:n_shot]]
    shots.append(format_example(test_row, with_answer=False))
    return "\n\n".join(shots)


def encode_prompt_left_truncated(tokenizer, dev_examples, test_row, n_shot, max_length):
    """Encode the few-shot prompt, dropping the EARLIEST shots if it is too long.

    Never right-truncates: the test question and the "Жауап:" answer cue are
    always preserved. If even the zero-shot prompt exceeds max_length, the
    token TAIL (question end + cue) is kept.

    Returns:
        (inputs dict, shots_used, tail_truncated)
    """
    shots_available = min(n_shot, len(dev_examples))
    inputs = None
    for k in range(shots_available, -1, -1):
        prompt = build_fewshot_prompt(dev_examples, test_row, n_shot=k)
        inputs = tokenizer(prompt, return_tensors="pt")
        if inputs["input_ids"].shape[1] <= max_length:
            return inputs, k, False
    # Even zero-shot is too long: keep the last max_length tokens (question + cue).
    inputs = {key: val[:, -max_length:] for key, val in inputs.items()}
    return inputs, 0, True


# ── Evaluation ────────────────────────────────────────────────────────────────


def get_option_token_ids(tokenizer):
    """Pre-compute token IDs for answer letters A–E."""
    token_map = {}
    for letter in "ABCDE":
        ids = tokenizer.encode(f" {letter}", add_special_tokens=False)
        token_map[letter] = ids[-1]
    return token_map


@torch.inference_mode()
def evaluate_kazmmlu(model, tokenizer, test_data, dev_data, n_shot=3, quick=False, max_length=2048):
    """Run KazMMLU n-shot next-token letter-logit evaluation.

    Returns:
        (accuracy, num_questions, stats dict with shot/truncation accounting)
    """
    dev_by_config = defaultdict(list)
    for row in dev_data:
        dev_by_config[row["_config"]].append(row)

    max_dev = max((len(v) for v in dev_by_config.values()), default=0)
    if n_shot > max_dev:
        print(
            f"  ⚠ Requested {n_shot}-shot but KazMMLU dev has at most {max_dev} exemplars "
            f"per subject — effective shot count is capped at {max_dev} (dev-limited)."
        )

    option_tokens = get_option_token_ids(tokenizer)
    device = next(model.parameters()).device

    correct = 0
    total = 0
    min_shots_used = n_shot
    n_shots_dropped = 0
    n_tail_truncated = 0
    subset = test_data[:100] if quick else test_data

    for row in tqdm(subset, desc="  KazMMLU eval"):
        dev_examples = dev_by_config.get(row["_config"], dev_data[:n_shot])
        inputs, shots_used, tail_truncated = encode_prompt_left_truncated(
            tokenizer, dev_examples, row, n_shot=n_shot, max_length=max_length
        )
        if shots_used < min(n_shot, len(dev_examples)):
            n_shots_dropped += 1
        if tail_truncated:
            n_tail_truncated += 1
        min_shots_used = min(min_shots_used, shots_used)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        logits = model(**inputs).logits[0, -1]  # next-token logits

        valid = get_valid_options(row)
        pred = max(valid, key=lambda letter: logits[option_tokens[letter]].item())

        if pred == row["Answer Key"]:
            correct += 1
        total += 1

    stats = {
        "shots_requested": n_shot,
        "shots_available_per_subject": max_dev,
        "min_shots_used": min_shots_used,
        "n_prompts_shots_dropped": n_shots_dropped,
        "n_prompts_tail_truncated": n_tail_truncated,
    }
    if n_shots_dropped or n_tail_truncated:
        print(
            f"  ⚠ Truncation fired: {n_shots_dropped} prompts lost early shots, "
            f"{n_tail_truncated} prompts were tail-truncated (max_length={max_length})."
        )

    acc = correct / total if total > 0 else 0.0
    return acc, total, stats


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

# Reported numbers from other papers — different harness, NOT measured by this
# script. Rendered in a separate table, never mixed into measured rows.
REPORTED_REFERENCE_ROWS = [
    ("Sherkala-Chat-8B", "8B", "41.4%", "2.04"),
    ("Llama-3.1-70B", "70B", "55.2%", "4.73"),
]


def get_repo_commit():
    """Git commit of this repo (for regenerable result provenance), or None."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() or None


def markdown_table(results, n_shot=3):
    """Render measured results, then reported reference numbers in a separate table."""
    lines = [
        "Measured by this script:",
        "",
        f"| Model | Params | KazMMLU {n_shot}-shot | Tok/word | Tok/sec |",
        "|-------|--------|---------------|----------|---------|",
    ]

    for r in sorted(results, key=lambda x: x.get("kazmmlu_acc", 0), reverse=True):
        if "error" in r:
            lines.append(f"| {r['name']} | — | FAILED | — | — |")
        else:
            lines.append(
                f"| **{r['name']}** | {r['params_b']}B "
                f"| {r['kazmmlu_acc']}% "
                f"| {r['fertility']} "
                f"| {r['speed_tok_s']} |"
            )
    lines.append("| Random baseline | — | ~25% (4-5 options) | — | — |")

    lines += [
        "",
        "Reported in their papers (different harness — not comparable, not measured here):",
        "",
        "| Model | Params | KazMMLU (reported) | Tok/word |",
        "|-------|--------|--------------------|----------|",
    ]
    for name, params, acc, fertility in REPORTED_REFERENCE_ROWS:
        lines.append(f"| {name} | {params} | {acc} | {fertility} |")
    return "\n".join(lines) + "\n"


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Benchmark models on KazMMLU (Kazakh)")
    parser.add_argument("--models", nargs="+", help="HF model IDs (overrides defaults)")
    parser.add_argument(
        "--output",
        default="results/baselines",
        help="Output directory (default is committed-friendly; evals/ is gitignored)",
    )
    parser.add_argument(
        "--n-shot",
        type=int,
        default=3,
        help="Few-shot count; 3 is the max with subject-matched shots (dev has 3/subject)",
    )
    parser.add_argument("--quick", action="store_true", help="100 questions only (fast test)")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "auto"])
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Prompt token budget; longer prompts drop earliest shots (never the question)",
    )
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

    metadata = {
        "date": time.strftime("%Y-%m-%d"),
        "repo_commit": get_repo_commit(),
        "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        "dtype": args.dtype,
        "n_shot": args.n_shot,
        "max_length": args.max_length,
        "quick": args.quick,
        "dataset": "MBZUAI/KazMMLU (Kazakh subset)",
    }

    def save_results():
        with open(output_dir / "baselines.json", "w") as f:
            json.dump({"metadata": metadata, "results": results}, f, indent=2, ensure_ascii=False)

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
                    model_id,
                    trust_remote_code=True,
                ).to(dtype=dtype, device=device)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                ).to(dtype=dtype, device=device)
            model.eval()
            params_b = sum(p.numel() for p in model.parameters()) / 1e9

            # KazMMLU
            acc, n_q, shot_stats = evaluate_kazmmlu(
                model,
                tokenizer,
                test_data,
                dev_data,
                n_shot=args.n_shot,
                quick=args.quick,
                max_length=args.max_length,
            )
            print(f"  KazMMLU {args.n_shot}-shot: {acc * 100:.1f}% ({n_q} questions)")

            # Speed
            speed = measure_speed(model, tokenizer)
            print(f"  Speed: {speed:.1f} tok/s")

            result = {
                "name": name,
                "model_id": model_id,
                "revision": getattr(model.config, "_commit_hash", None),
                "params_b": round(params_b, 2),
                "kazmmlu_acc": round(acc * 100, 1),
                "n_shot": args.n_shot,
                "n_questions": n_q,
                "shot_stats": shot_stats,
                "fertility": round(fertility, 2),
                "speed_tok_s": round(speed, 1),
            }
            results.append(result)

            # Save incrementally (resume-friendly)
            save_results()

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"name": name, "model_id": model_id, "error": str(e)})
            save_results()

    # Print results
    print(f"\n{'=' * 60}")
    print(" BASELINE RESULTS")
    print(f"{'=' * 60}\n")
    table = markdown_table(results, n_shot=args.n_shot)
    print(table)

    # Save markdown
    with open(output_dir / "baselines.md", "w") as f:
        f.write("# KazMMLU Baseline Results\n\n")
        f.write(f"Date: {metadata['date']}\n")
        f.write(f"Device: {metadata['device']}\n")
        f.write(f"Repo commit: {metadata['repo_commit']}\n")
        f.write(
            f"Eval: {args.n_shot}-shot (dev-limited: 3 exemplars/subject), "
            f"Kazakh subset only, dtype {args.dtype}\n\n"
        )
        f.write(table)

    print(f"Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
