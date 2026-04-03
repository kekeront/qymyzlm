"""Download, clean, tokenize, pack Kazakh data and upload to GCS.

Can run locally or on a cheap GCE VM. Does NOT need a GPU.

Usage:
    # Full pipeline (~6-12 hours depending on bandwidth):
    PYTHONPATH=src python scripts/prepare_gcs_data.py \
        --gcs-bucket gs://kazllm-training-v3 \
        --skip-gated   # omit CulturaX if no HF token

    # Upload existing local tokenized data (fast):
    PYTHONPATH=src python scripts/prepare_gcs_data.py \
        --gcs-bucket gs://kazllm-training-v3 \
        --upload-only
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def gcs_upload(local_path: str, gcs_path: str) -> None:
    log.info(f"Uploading {local_path} -> {gcs_path}")
    subprocess.run(
        ["gcloud", "storage", "cp", "-r", local_path, gcs_path],
        check=True,
    )


def run_step(cmd: list[str], description: str, env=None) -> None:
    log.info(f"=== {description} ===")
    log.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        log.error(f"Failed: {description} (exit {result.returncode})")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Prepare training data and upload to GCS")
    parser.add_argument("--gcs-bucket", default="gs://kazllm-training-v3")
    parser.add_argument("--upload-only", action="store_true", help="Skip processing, just upload existing data")
    parser.add_argument("--skip-gated", action="store_true", help="Skip gated datasets (CulturaX)")
    parser.add_argument("--data-config", default="default", help="Hydra data config name")
    args = parser.parse_args()

    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    tokenizer_local = Path("data/tokenizer/kaz_sp_unigram_50k/hf_tokenizer")
    tokenized_local = Path("data/tokenized")

    if not args.upload_only:
        # Step 1: Download raw data from HuggingFace
        run_step(
            [sys.executable, "scripts/download_data.py", f"data={args.data_config}"],
            "Downloading raw data from HuggingFace",
            env=env,
        )

        # Step 2: Clean and deduplicate
        run_step(
            [sys.executable, "scripts/clean_data.py", f"data={args.data_config}"],
            "Cleaning and deduplicating data",
            env=env,
        )

        # Step 3: Tokenize and pack into shards
        if not tokenizer_local.exists():
            log.error(f"Tokenizer not found at {tokenizer_local}. Run 'make tokenizer' first.")
            sys.exit(1)

        run_step(
            [sys.executable, "scripts/pack_data.py", f"data={args.data_config}"],
            "Tokenizing and packing into shards",
            env=env,
        )

    # Step 4: Upload to GCS
    gcs_data = f"{args.gcs_bucket}/data/tokenized"
    gcs_tokenizer = f"{args.gcs_bucket}/tokenizer"

    if not tokenized_local.exists() or not (tokenized_local / "manifest.json").exists():
        log.error(f"No tokenized data found at {tokenized_local}")
        sys.exit(1)

    # Upload tokenized shards
    log.info("=== Uploading tokenized data to GCS ===")
    manifest = json.loads((tokenized_local / "manifest.json").read_text())
    log.info(f"Uploading {manifest['num_shards']} shards ({manifest['total_tokens']:,} tokens)")

    # Upload manifest
    gcs_upload(str(tokenized_local / "manifest.json"), f"{gcs_data}/manifest.json")

    # Upload each shard
    for shard in manifest["shards"]:
        shard_file = Path(shard["shard_path"]).name
        local_shard = tokenized_local / shard_file
        if local_shard.exists():
            gcs_upload(str(local_shard), f"{gcs_data}/{shard_file}")
        else:
            log.warning(f"Shard not found: {local_shard}")

    # Upload tokenizer
    log.info("=== Uploading tokenizer to GCS ===")
    if tokenizer_local.exists():
        for f in tokenizer_local.iterdir():
            gcs_upload(str(f), f"{gcs_tokenizer}/{f.name}")
    else:
        log.warning(f"Tokenizer not found at {tokenizer_local}")

    log.info("=== Done ===")
    log.info(f"Data: {gcs_data}")
    log.info(f"Tokenizer: {gcs_tokenizer}")
    log.info(f"Total tokens: {manifest['total_tokens']:,}")


if __name__ == "__main__":
    main()
