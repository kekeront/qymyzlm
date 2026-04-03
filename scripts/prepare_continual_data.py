"""Download, tokenize, and pack Kazakh data for QymyzLM continual pretraining.

Uses the EXPANDED Qwen tokenizer (from checkpoints/qymyz1_5b_init/tokenizer)
so token IDs match the model's vocabulary. Outputs uint32 shards.

Auth-free sources (~4.5B tokens raw, ~3-4B after filtering):
  - HPLT 2.0 (kaz_Cyrl)
  - Wikipedia (kk)
  - multidomain-kazakh-dataset
  - mOSCAR (kaz_Cyrl)
  - C4 (kk)

Usage:
    PYTHONPATH=src python scripts/prepare_continual_data.py
    PYTHONPATH=src python scripts/prepare_continual_data.py --sources wiki multidomain  # subset
    PYTHONPATH=src python scripts/prepare_continual_data.py --skip-download  # tokenize only
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from kazllm.data.packer import pack_sequences
from kazllm.data.shard_writer import ShardWriter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# Sources in priority order (largest/most useful first)
SOURCES = {
    "hplt2": {
        "hf_repo": "HPLT/HPLT2.0_cleaned",
        "lang": "kaz_Cyrl",
        "split": "train",
        "text_col": "text",
        "est_tokens": "1.8B",
    },
    "c4": {
        "hf_repo": "allenai/c4",
        "lang": "kk",
        "split": "train",
        "text_col": "text",
        "est_tokens": "1.0B",
    },
    "moscar": {
        "hf_repo": "oscar-corpus/mOSCAR",
        "lang": "kaz_Cyrl",
        "split": "train",
        "text_col": "text",
        "est_tokens": "0.5B",
    },
    "wiki": {
        "hf_repo": "wikimedia/wikipedia",
        "lang": "20231101.kk",
        "split": "train",
        "text_col": "text",
        "est_tokens": "0.18B",
    },
    "multidomain": {
        "hf_repo": "kz-transformers/multidomain-kazakh-dataset",
        "lang": None,
        "split": "train",
        "text_col": "text",
        "est_tokens": "0.25B",
    },
}

# Minimum document quality thresholds
MIN_CHARS = 50
MIN_CYRILLIC_RATIO = 0.4


def is_kazakh_enough(text: str) -> bool:
    """Quick filter: enough Cyrillic chars and long enough."""
    if len(text) < MIN_CHARS:
        return False
    cyrillic = sum(1 for c in text if "\u0400" <= c <= "\u04ff")
    return cyrillic / max(len(text), 1) >= MIN_CYRILLIC_RATIO


def stream_source(src_name: str, src_cfg: dict):
    """Stream text from a HuggingFace source with basic filtering."""
    load_kwargs = {"split": src_cfg["split"], "streaming": True}
    if src_cfg["lang"]:
        load_kwargs["name"] = src_cfg["lang"]

    log.info(f"[{src_name}] Streaming from {src_cfg['hf_repo']} (est. {src_cfg['est_tokens']} tokens)...")
    ds = load_dataset(src_cfg["hf_repo"], **load_kwargs, trust_remote_code=True)

    text_col = src_cfg["text_col"]
    kept, skipped = 0, 0
    for ex in ds:
        text = ex.get(text_col, "")
        if not text or not is_kazakh_enough(text):
            skipped += 1
            continue
        kept += 1
        if kept % 100_000 == 0:
            log.info(f"  [{src_name}] {kept:,} kept, {skipped:,} filtered")
        yield text

    log.info(f"  [{src_name}] DONE: {kept:,} kept, {skipped:,} filtered")


def main():
    parser = argparse.ArgumentParser(description="Prepare continual PT data for QymyzLM")
    parser.add_argument(
        "--tokenizer-dir",
        default="checkpoints/qymyz1_5b_init/tokenizer",
        help="Path to expanded Qwen tokenizer",
    )
    parser.add_argument(
        "--output-dir",
        default="data/tokenized/continual",
        help="Output directory for packed shards",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=2048,
        help="Sequence length for packing",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=list(SOURCES.keys()),
        default=list(SOURCES.keys()),
        help="Which sources to process (default: all)",
    )
    parser.add_argument(
        "--tokens-per-shard",
        type=int,
        default=100_000_000,  # 100M tokens per shard (~400MB uint32)
        help="Tokens per shard file",
    )
    args = parser.parse_args()

    # Load tokenizer
    tokenizer_dir = Path(args.tokenizer_dir)
    if not tokenizer_dir.exists():
        log.error(f"Tokenizer not found at {tokenizer_dir}. Run build_continual_pt.py first.")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), trust_remote_code=True)
    vocab_size = len(tokenizer)
    log.info(f"Loaded tokenizer: vocab_size={vocab_size}")

    # Setup writer
    output_dir = Path(args.output_dir)
    writer = ShardWriter(output_dir, tokens_per_shard=args.tokens_per_shard)

    total_docs = 0
    total_tokens_est = 0

    def token_iterator():
        nonlocal total_docs, total_tokens_est
        for src_name in args.sources:
            src_cfg = SOURCES[src_name]
            try:
                for text in stream_source(src_name, src_cfg):
                    ids = tokenizer.encode(text, add_special_tokens=False)
                    if ids:
                        total_docs += 1
                        total_tokens_est += len(ids)
                        yield ids
            except Exception as e:
                log.error(f"[{src_name}] Failed: {e}")
                log.error(f"[{src_name}] Continuing with next source...")
                continue

    for sequence in pack_sequences(
        token_iterator(),
        context_length=args.context_length,
        eos_token_id=tokenizer.eos_token_id or 151643,  # Qwen EOS
        vocab_size=vocab_size,
    ):
        writer.write(sequence)

    manifest = writer.finalize()
    log.info(f"COMPLETE: {manifest['total_tokens']:,} tokens in {manifest['num_shards']} shards")
    log.info(f"Documents processed: {total_docs:,}")
    log.info(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
