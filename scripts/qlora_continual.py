#!/usr/bin/env python3
"""QLoRA continual pretraining for any HF model on Kazakh data.

Streams Kazakh text from HuggingFace, trains with 4-bit QLoRA.
Fits on RTX 2070 (8GB) for models up to ~3B.

Usage:
    # Quick test (1M tokens, ~2 min)
    python scripts/qlora_continual.py --tokens 1_000_000

    # 100M tokens (~1-2 hours on RTX 2070)
    python scripts/qlora_continual.py --tokens 100_000_000

    # 400M tokens (~4-6 hours)
    python scripts/qlora_continual.py --tokens 400_000_000

    # Custom model
    python scripts/qlora_continual.py --model Qwen/Qwen2.5-1.5B --tokens 100_000_000
"""

import argparse
import logging
import os

os.environ["PYTHONUNBUFFERED"] = "1"

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Data sources (priority order) ─────────────────────────────────────────────

SOURCES = [
    {"repo": "wikimedia/wikipedia", "name": "20231101.kk", "text_col": "text", "label": "wiki"},
    {"repo": "kz-transformers/multidomain-kazakh-dataset", "name": None, "text_col": "text", "label": "multidomain"},
    {"repo": "HPLT/HPLT2.0_cleaned", "name": "kaz_Cyrl", "text_col": "text", "label": "hplt2"},
    {"repo": "allenai/c4", "name": "kk", "text_col": "text", "label": "c4"},
]

MIN_CHARS = 100
MIN_CYRILLIC_RATIO = 0.4


def is_kazakh(text: str) -> bool:
    if len(text) < MIN_CHARS:
        return False
    cyrillic = sum(1 for c in text if "\u0400" <= c <= "\u04ff")
    return cyrillic / max(len(text), 1) >= MIN_CYRILLIC_RATIO


def stream_kazakh_texts(max_tokens: int, tokenizer, seq_length: int):
    """Stream tokenized Kazakh text as packed sequences."""
    total_tokens = 0
    buffer = []

    for src in SOURCES:
        if total_tokens >= max_tokens:
            break

        log.info(f"Streaming from {src['label']} ({src['repo']})...")
        load_kw = {"split": "train", "streaming": True}
        if src["name"]:
            load_kw["name"] = src["name"]

        try:
            ds = load_dataset(src["repo"], **load_kw)
        except Exception as e:
            log.warning(f"  Skipping {src['label']}: {e}")
            continue

        docs_kept = 0
        for ex in ds:
            if total_tokens >= max_tokens:
                break

            text = ex.get(src["text_col"], "")
            if not is_kazakh(text):
                continue

            ids = tokenizer.encode(text, add_special_tokens=False)
            ids.append(tokenizer.eos_token_id)
            buffer.extend(ids)
            docs_kept += 1

            while len(buffer) >= seq_length:
                chunk = buffer[:seq_length]
                buffer = buffer[seq_length:]
                total_tokens += seq_length
                yield {"input_ids": chunk, "labels": chunk}

                if total_tokens % 1_000_000 < seq_length:
                    log.info(f"  {total_tokens / 1e6:.1f}M tokens ({docs_kept} docs from {src['label']})")

        log.info(f"  {src['label']}: {docs_kept} docs, running total: {total_tokens / 1e6:.1f}M tokens")

    log.info(f"Total: {total_tokens / 1e6:.1f}M tokens streamed")


class StreamingPackedDataset(torch.utils.data.Dataset):
    """Pre-collect streamed data into memory for Trainer compatibility."""

    def __init__(self, data: list[dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
        }


def main():
    parser = argparse.ArgumentParser(description="QLoRA continual PT on Kazakh data")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B-Base", help="Base model HF ID")
    parser.add_argument("--tokens", type=int, default=100_000_000, help="Total tokens to train on")
    parser.add_argument("--seq-length", type=int, default=512, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--output", default=None, help="Output dir (default: auto)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--save-steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    args = parser.parse_args()

    model_short = args.model.split("/")[-1].lower()
    tokens_m = args.tokens // 1_000_000
    output_dir = args.output or f"checkpoints/qlora_{model_short}_{tokens_m}m"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    if device == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name()}")
        log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    log.info(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Stream and collect data ───────────────────────────────────────────────
    log.info(f"Collecting {tokens_m}M tokens of Kazakh data...")
    data = list(stream_kazakh_texts(args.tokens, tokenizer, args.seq_length))
    log.info(f"Collected {len(data)} sequences ({len(data) * args.seq_length / 1e6:.1f}M tokens)")

    if not data:
        log.error("No data collected! Check internet connection and data sources.")
        return

    dataset = StreamingPackedDataset(data)

    # ── Model (4-bit quantized) ───────────────────────────────────────────────
    log.info(f"Loading model: {args.model} (4-bit quantized)")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # ── LoRA ──────────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Training ──────────────────────────────────────────────────────────────
    effective_batch = args.batch_size * args.grad_accum
    tokens_per_step = effective_batch * args.seq_length
    total_steps = (len(data) * args.epochs) // effective_batch
    warmup_steps = min(100, total_steps // 10)

    log.info(f"Training config:")
    log.info(f"  Sequences: {len(data)}")
    log.info(f"  Effective batch: {effective_batch} seqs ({tokens_per_step / 1e3:.0f}K tokens/step)")
    log.info(f"  Total steps: {total_steps}")
    log.info(f"  Warmup: {warmup_steps} steps")
    log.info(f"  Output: {output_dir}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        max_steps=total_steps,
        num_train_epochs=args.epochs,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_steps=10,
        report_to=["wandb"] if args.wandb else ["none"],
        run_name=f"qlora_{model_short}_{tokens_m}m",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    log.info("Starting QLoRA continual pretraining...")
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    log.info(f"Saving LoRA adapter to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    log.info("Done! Run benchmark:")
    log.info(f"  python scripts/benchmark_baselines.py --models {output_dir}")


if __name__ == "__main__":
    main()
