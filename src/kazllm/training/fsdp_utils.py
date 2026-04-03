"""FSDP wrapping policy and checkpoint consolidation utilities."""

import logging
from functools import partial
from pathlib import Path

import torch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from kazllm.model.block import TransformerBlock

log = logging.getLogger(__name__)


def get_fsdp_wrap_policy():
    """Auto-wrap policy that shards at TransformerBlock granularity."""
    return partial(transformer_auto_wrap_policy, transformer_layer_cls={TransformerBlock})


def consolidate_checkpoint(sharded_dir: str | Path, output_hf_dir: str | Path) -> None:
    """Consolidate FSDP sharded checkpoints into a single HuggingFace-format checkpoint.

    Must be called on rank 0 after all ranks have saved their shards.

    Args:
        sharded_dir: Directory containing FSDP sharded state dicts.
        output_hf_dir: Directory to write the merged HF checkpoint.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    sharded_dir = Path(sharded_dir)
    output_hf_dir = Path(output_hf_dir)
    output_hf_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Consolidating checkpoint from {sharded_dir} to {output_hf_dir}")

    # Load config from sharded dir
    config = AutoConfig.from_pretrained(sharded_dir)
    model = AutoModelForCausalLM.from_config(config)

    # Load consolidated state dict
    state_dict_path = sharded_dir / "pytorch_model.bin"
    if state_dict_path.exists():
        state_dict = torch.load(state_dict_path, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        log.warning(f"No pytorch_model.bin found in {sharded_dir}; skipping weight loading")

    model.save_pretrained(output_hf_dir)
    log.info(f"Consolidated HF checkpoint saved to {output_hf_dir}")
