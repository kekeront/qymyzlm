"""Vertex AI training entry-point for KazLLM-500M.

Handles:
- Downloading tokenized data from GCS to local SSD
- Training with checkpoint saving to local + GCS sync
- Automatic resume from latest GCS checkpoint on preemption

Environment variables (set by Vertex AI or manually):
    GCS_DATA_DIR:       gs://bucket/path/to/tokenized/  (contains manifest.json + shards)
    GCS_CHECKPOINT_DIR: gs://bucket/path/to/checkpoints/
    GCS_TOKENIZER_DIR:  gs://bucket/path/to/tokenizer/
    WANDB_API_KEY:      (optional) for W&B logging
    MODEL_CONFIG:       model config name (default: kaz500m)
    TRAINING_CONFIG:    training config name (default: pretrain_500m)
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def sync_from_gcs(gcs_path: str, local_path: str) -> None:
    """Download from GCS to local path."""
    Path(local_path).mkdir(parents=True, exist_ok=True)
    log.info(f"Syncing {gcs_path} -> {local_path}")
    subprocess.run(
        ["gcloud", "storage", "cp", "-r", f"{gcs_path}/*", local_path],
        check=True,
    )


def sync_to_gcs(local_path: str, gcs_path: str) -> None:
    """Upload local path to GCS."""
    log.info(f"Syncing {local_path} -> {gcs_path}")
    subprocess.run(
        ["gcloud", "storage", "cp", "-r", local_path, gcs_path],
        check=True,
    )


def find_latest_gcs_checkpoint(gcs_dir: str) -> str | None:
    """Find latest checkpoint-N in GCS directory."""
    result = subprocess.run(
        ["gcloud", "storage", "ls", gcs_dir],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None

    ckpt_dirs = [
        line.strip().rstrip("/")
        for line in result.stdout.strip().split("\n")
        if "checkpoint-" in line
    ]
    if not ckpt_dirs:
        return None

    latest = sorted(ckpt_dirs, key=lambda x: int(x.split("checkpoint-")[1].split("/")[0]))[-1]
    return latest


def main():
    # Read env config
    gcs_data = os.environ.get("GCS_DATA_DIR", "gs://kazllm-training-v3/data/tokenized")
    gcs_ckpt = os.environ.get("GCS_CHECKPOINT_DIR", "gs://kazllm-training-v3/checkpoints/kaz500m")
    gcs_tokenizer = os.environ.get("GCS_TOKENIZER_DIR", "gs://kazllm-training-v3/tokenizer")
    model_config = os.environ.get("MODEL_CONFIG", "kaz500m")
    training_config = os.environ.get("TRAINING_CONFIG", "pretrain_500m")

    local_data = "/tmp/data/tokenized"
    local_ckpt = "/tmp/checkpoints"
    local_tokenizer = "/tmp/tokenizer"

    # Log GPU info
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        log.error("No CUDA device found!")
        sys.exit(1)

    # 1. Sync data from GCS
    log.info("=== Step 1: Downloading training data from GCS ===")
    sync_from_gcs(gcs_data, local_data)

    # 2. Sync tokenizer from GCS
    log.info("=== Step 2: Downloading tokenizer from GCS ===")
    sync_from_gcs(gcs_tokenizer, local_tokenizer)

    # 3. Check for existing checkpoint to resume from
    log.info("=== Step 3: Checking for existing checkpoints ===")
    resume_from = None
    latest_gcs_ckpt = find_latest_gcs_checkpoint(gcs_ckpt)
    if latest_gcs_ckpt:
        log.info(f"Found GCS checkpoint: {latest_gcs_ckpt}")
        sync_from_gcs(latest_gcs_ckpt, local_ckpt)
        # Find local checkpoint dir
        local_ckpt_dirs = sorted(Path(local_ckpt).glob("checkpoint-*"))
        if local_ckpt_dirs:
            resume_from = str(local_ckpt_dirs[-1])
            log.info(f"Will resume from: {resume_from}")
    else:
        log.info("No existing checkpoint found, training from scratch")

    # 4. Fix manifest shard paths to use local paths
    import json

    manifest_path = Path(local_data) / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    for shard in manifest["shards"]:
        shard_name = Path(shard["shard_path"]).name
        shard["shard_path"] = str(Path(local_data) / shard_name)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"Updated manifest: {manifest['num_shards']} shards, {manifest['total_tokens']:,} tokens")

    # 5. Run training via Hydra
    log.info(f"=== Step 4: Starting training (model={model_config}, training={training_config}) ===")

    train_cmd = [
        sys.executable,
        "scripts/train.py",
        f"model={model_config}",
        f"training={training_config}",
        f"training.tokenized_dir={local_data}",
        f"training.checkpoint_dir={local_ckpt}",
    ]
    if resume_from:
        train_cmd.append(f"training.resume_from_checkpoint={resume_from}")

    # Set wandb mode
    env = os.environ.copy()
    if "WANDB_API_KEY" not in env:
        env["WANDB_MODE"] = "disabled"

    result = subprocess.run(train_cmd, env=env)

    if result.returncode != 0:
        log.error(f"Training failed with return code {result.returncode}")
        # Still sync whatever checkpoints we have
        log.info("Syncing partial checkpoints to GCS...")
    else:
        log.info("Training completed successfully!")

    # 6. Sync checkpoints to GCS
    log.info("=== Step 5: Uploading checkpoints to GCS ===")
    if Path(local_ckpt).exists():
        sync_to_gcs(f"{local_ckpt}/", gcs_ckpt)
        log.info("Checkpoints synced to GCS")
    else:
        log.warning("No checkpoints to upload")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
