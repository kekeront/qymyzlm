"""Export a training checkpoint to HF-compatible format for evaluation and serving.

Strips MTP modules (training-only) and saves model + tokenizer in a single directory.

Usage:
    PYTHONPATH=src python scripts/export_checkpoint.py \
        --checkpoint checkpoints/kaz_nano \
        --output checkpoints/kaz_nano/hf_export
"""

import argparse
import logging
import shutil
from pathlib import Path

import torch
import yaml

from kazllm.model.config import KazLLMConfig
from kazllm.model.model import KazLLMForCausalLM

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def find_latest_checkpoint(checkpoint_dir: Path) -> Path:
    """Find the latest checkpoint-N directory."""
    ckpt_dirs = sorted(
        checkpoint_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]),
    )
    if not ckpt_dirs:
        raise FileNotFoundError(f"No checkpoint-* directories found in {checkpoint_dir}")
    return ckpt_dirs[-1]


def export(checkpoint_dir: str, output_dir: str, model_config: str | None = None):
    checkpoint_path = Path(checkpoint_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find latest checkpoint
    ckpt = find_latest_checkpoint(checkpoint_path)
    log.info(f"Exporting from: {ckpt}")

    # Load config
    config_json = ckpt / "config.json"
    if config_json.exists():
        config = KazLLMConfig.from_pretrained(str(ckpt))
    elif model_config:
        with open(model_config) as f:
            cfg = yaml.safe_load(f)
        config = KazLLMConfig(**cfg)
    else:
        raise FileNotFoundError(f"No config.json in {ckpt}. Provide --model-config path to yaml.")

    # Disable MTP for inference
    config.use_mtp = False

    # Load model
    model = KazLLMForCausalLM(config)

    # Load weights, stripping MTP modules
    model_file = ckpt / "model.safetensors"
    if not model_file.exists():
        model_file = ckpt / "pytorch_model.bin"
    if not model_file.exists():
        raise FileNotFoundError(f"No model weights found in {ckpt}")

    state_dict = torch.load(str(model_file), map_location="cpu", weights_only=True)
    # Strip MTP modules
    mtp_keys = [k for k in state_dict if "mtp_modules" in k]
    if mtp_keys:
        log.info(f"Stripping {len(mtp_keys)} MTP keys (training-only)")
        for k in mtp_keys:
            del state_dict[k]

    model.load_state_dict(state_dict, strict=False)
    log.info(f"Loaded weights ({len(state_dict)} tensors)")

    # Save model
    model.save_pretrained(str(output_path))

    # Copy tokenizer
    tokenizer_src = Path("data/tokenizer/kaz_sp_unigram_50k/hf_tokenizer")
    if tokenizer_src.exists():
        for f in tokenizer_src.iterdir():
            shutil.copy2(f, output_path / f.name)
        log.info(f"Copied tokenizer from {tokenizer_src}")
    else:
        log.warning(f"Tokenizer not found at {tokenizer_src}")

    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Exported to {output_path}: {total_params / 1e6:.1f}M params (MTP stripped)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Training checkpoint directory")
    parser.add_argument(
        "--output", default=None, help="Output dir (default: <checkpoint>/hf_export)"
    )
    parser.add_argument(
        "--model-config", default=None, help="Model config YAML (if no config.json)"
    )
    args = parser.parse_args()

    output = args.output or str(Path(args.checkpoint) / "hf_export")
    export(args.checkpoint, output, args.model_config)
