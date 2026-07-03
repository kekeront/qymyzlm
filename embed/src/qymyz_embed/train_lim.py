"""Less-is-More fine-tune of intfloat/multilingual-e5-base (arXiv 2603.22290).

Protocol (paper defaults): ~10,000 pairs, FULL fine-tune, effective batch 512, lr 7e-5,
5 epochs, linear scheduler with 0.2 warmup ratio, then a 0.5/0.5 weight average with the
base model (qymyz_embed.merge).

Effective batch 512 on the RTX 2070 (8 GB): CachedMultipleNegativesRankingLoss (GradCache)
with per_device_train_batch_size=512 — memory is bounded by mini_batch_size, NOT the batch
size. gradient_accumulation_steps is pinned to 1 because accumulation does NOT enlarge the
in-batch-negative pool for (Cached)MNRL.

Turing GPU: fp16 yes, bf16 NO. warmup is passed as warmup_steps=0.2 (float == ratio;
warmup_ratio is deprecated under transformers v5).

Usage:
    python -m qymyz_embed.train_lim --data pairs.jsonl
    python -m qymyz_embed.train_lim --data pairs_ds --output embed/checkpoints/lim-v0

Checkpoints land in embed/checkpoints/ by default (gitignored). Evaluate ONLY via
qymyz_embed.evaluate (evallab runners).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import torch
from datasets import Dataset, load_from_disk
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.training_args import BatchSamplers

from qymyz_embed.prefixes import E5_PROMPTS, has_query_prefix, training_prompts_for_columns

ME5_BASE = "intfloat/multilingual-e5-base"  # XLM-R base, 278M params, dim 768
DEFAULT_MAX_PAIRS = 10_000  # the paper's "less is more" cap
# embed/src/qymyz_embed/train_lim.py -> parents[2] == embed/
DEFAULT_CHECKPOINT_ROOT = Path(__file__).resolve().parents[2] / "checkpoints"


def load_pairs(path: Path) -> Dataset:
    """Load a pairs dataset from a JSONL file or a save_to_disk() directory."""
    if path.is_dir():
        dataset = load_from_disk(str(path))
        if not isinstance(dataset, Dataset):
            raise ValueError(f"{path} is a DatasetDict; point --data at a single split")
        return dataset
    with path.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    if not rows:
        raise ValueError(f"no rows in {path}")
    return Dataset.from_list(rows)


def _ordered_columns(columns: Sequence[str]) -> list[str]:
    """Contrastive column order: anchor, positive, negative(_1..n). Extra columns dropped."""
    negatives = sorted((c for c in columns if c.startswith("negative")), key=lambda c: (len(c), c))
    return ["anchor", "positive", *negatives]


def prepare_pairs(dataset: Dataset, max_pairs: int = DEFAULT_MAX_PAIRS, seed: int = 42) -> Dataset:
    """Validate columns, enforce role order (column ORDER defines roles for MNRL), cap size.

    max_pairs=0 disables the cap; otherwise a seeded shuffle picks the subset.
    """
    if not {"anchor", "positive"} <= set(dataset.column_names):
        raise ValueError(
            f"pairs dataset needs 'anchor' and 'positive' columns, got {dataset.column_names}"
        )
    dataset = dataset.select_columns(_ordered_columns(dataset.column_names))
    if max_pairs and len(dataset) > max_pairs:
        dataset = dataset.shuffle(seed=seed).select(range(max_pairs))
    return dataset


def build_loss(
    model: SentenceTransformer, *, scale: float = 20.0, mini_batch_size: int = 8
) -> CachedMultipleNegativesRankingLoss:
    """scale=20.0 == temperature 0.05 (ST default). mini_batch_size bounds GPU memory;
    8 is the reasoned-safe value for mE5-base on 8 GB — try 16 if headroom allows."""
    return CachedMultipleNegativesRankingLoss(model, scale=scale, mini_batch_size=mini_batch_size)


def build_args(
    output_dir: str | Path,
    *,
    batch_size: int = 512,
    lr: float = 7e-5,
    epochs: float = 5,
    warmup_ratio: float = 0.2,
    seed: int = 42,
    prompts: dict[str, str] | None = None,
    fp16: bool | None = None,
) -> SentenceTransformerTrainingArguments:
    """Less-is-More training arguments. fp16=None auto-detects CUDA (Turing: never bf16)."""
    if fp16 is None:
        fp16 = torch.cuda.is_available()
    return SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,  # the FULL in-batch-negative pool
        gradient_accumulation_steps=1,  # accumulation does NOT grow the negative pool
        learning_rate=lr,
        num_train_epochs=epochs,
        fp16=fp16,
        bf16=False,  # Turing has no bf16
        warmup_steps=warmup_ratio,  # float == ratio; warmup_ratio deprecated (transformers v5)
        # lr_scheduler_type default is already "linear" — matches the paper
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # duplicate positives poison in-batch negs
        prompts=prompts,
        save_strategy="no",
        logging_steps=10,
        seed=seed,
        report_to="none",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Less-is-More fine-tune (arXiv 2603.22290)")
    parser.add_argument("--data", type=Path, required=True, help="pairs JSONL or HF dataset dir")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="checkpoint dir (default: embed/checkpoints/lim-<timestamp>)",
    )
    parser.add_argument("--base-model", default=ME5_BASE)
    parser.add_argument(
        "--max-pairs", type=int, default=DEFAULT_MAX_PAIRS, help="0 disables the cap"
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--mini-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=7e-5)
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--warmup-ratio", type=float, default=0.2)
    parser.add_argument("--scale", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args(argv)

    dataset = prepare_pairs(load_pairs(args.data), max_pairs=args.max_pairs, seed=args.seed)

    # kazparc_pairs.py emits prefixed text by default; raw text gets prefixes via prompts=.
    already_prefixed = has_query_prefix(str(dataset[0]["anchor"]))
    prompts = None if already_prefixed else training_prompts_for_columns(dataset.column_names)
    print(
        f"{len(dataset)} pairs, columns {dataset.column_names}, "
        f"prefixes {'in data' if already_prefixed else 'via prompts='}",
        file=sys.stderr,
    )

    output_dir = args.output or DEFAULT_CHECKPOINT_ROOT / f"lim-{time.strftime('%Y%m%d-%H%M%S')}"
    # Register e5 prompts so the SAVED model's encode_query()/encode_document() work.
    model = SentenceTransformer(args.base_model, prompts=E5_PROMPTS)
    loss = build_loss(model, scale=args.scale, mini_batch_size=args.mini_batch_size)
    training_args = build_args(
        output_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        prompts=prompts,
        fp16=False if args.no_fp16 else None,
    )
    trainer = SentenceTransformerTrainer(
        model=model, args=training_args, train_dataset=dataset, loss=loss
    )
    trainer.train()

    final_dir = Path(output_dir) / "final"
    model.save(str(final_dir))
    print(
        f"saved -> {final_dir}\n"
        f"next: python -m qymyz_embed.merge {final_dir} {args.base_model} "
        f"--output {output_dir}/souped  # 0.5/0.5 soup with the base",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
