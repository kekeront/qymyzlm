"""Custom Trainer for KazLLM pretraining."""

import logging
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import Trainer

from kazllm.training.loss import compute_lm_loss

log = logging.getLogger(__name__)


class ShardedMemmapDataset(Dataset):
    """Dataset that reads packed token sequences from numpy memmap shards.

    Supports both uint16 (vocab <= 65535) and uint32 (large vocabs like Qwen's 151K).
    The dtype is read from the manifest's "dtype" field, defaulting to uint16.
    """

    def __init__(self, manifest_path: str, context_length: int = 2048):
        import json

        import numpy as np

        with open(manifest_path) as f:
            manifest = json.load(f)

        self.context_length = context_length
        dtype_str = manifest.get("dtype", "uint16")
        dtype = np.dtype(dtype_str)
        self._shards: list[np.memmap] = []
        self._offsets: list[int] = [0]

        for shard_info in manifest["shards"]:
            arr = np.memmap(shard_info["shard_path"], dtype=dtype, mode="r")
            self._shards.append(arr)
            self._offsets.append(self._offsets[-1] + len(arr) // context_length)

        self._total_sequences = self._offsets[-1]

    def __len__(self) -> int:
        return self._total_sequences

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        import bisect

        import numpy as np

        shard_idx = bisect.bisect_right(self._offsets, idx) - 1
        local_idx = idx - self._offsets[shard_idx]
        start = local_idx * self.context_length
        tokens = self._shards[shard_idx][start : start + self.context_length].astype(np.int64)
        input_ids = torch.from_numpy(tokens)
        return {"input_ids": input_ids, "labels": input_ids.clone()}


def _mtp_lambda_schedule(step: int, total_steps: int, peak_lambda: float = 0.3) -> float:
    """MTP lambda curriculum: NTP warmup → ramp → peak → decay.

    Schedule (following Aynetdinov & Akbik, ACL 2025 "Pre-Training Curriculum
    for Multi-Token Prediction in Language Models", arxiv 2505.22757):
      - Steps  0-20%: lambda=0 (NTP only — model learns stable language prior)
      - Steps 20-50%: linear ramp 0 → peak_lambda
      - Steps 50-60%: peak_lambda (full MTP pressure)
      - Steps 60-100%: peak_lambda/3 (reduced, matches DeepSeek-V3 late schedule)

    The 20% NTP warmup is critical: without it, MTP degrades small models (<1B)
    because they lack capacity to handle multi-token prediction from step 0.
    """
    frac = step / max(total_steps, 1)
    if frac < 0.20:
        return 0.0
    elif frac < 0.50:
        return peak_lambda * (frac - 0.20) / 0.30
    elif frac < 0.60:
        return peak_lambda
    else:
        return peak_lambda / 3.0


class KazLLMTrainer(Trainer):
    """Trainer subclass with cross-document loss masking and structured metrics logging.

    Overrides create_optimizer to give Engram embedding tables a separate
    parameter group with 5x LR and zero weight decay (per Engram paper's
    training recipe — sparse tables need higher LR to learn from sparse updates).
    """

    # Engram tables need 5x the backbone LR (Engram paper training recipe)
    ENGRAM_LR_MULT = 5.0

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        engram_table_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "engram_layers" in name and ".tables." in name:
                engram_table_params.append(param)
            else:
                other_params.append(param)

        base_lr = self.args.learning_rate
        param_groups = [
            {"params": other_params, "lr": base_lr, "weight_decay": self.args.weight_decay},
        ]
        if engram_table_params:
            param_groups.append({
                "params": engram_table_params,
                "lr": base_lr * self.ENGRAM_LR_MULT,
                "weight_decay": 0.0,
            })
            log.info(
                f"Engram optimizer: {len(engram_table_params)} table params at "
                f"{base_lr * self.ENGRAM_LR_MULT:.1e} LR (5x backbone), wd=0"
            )

        self.optimizer = torch.optim.AdamW(param_groups)
        return self.optimizer

    def _densify_sparse_grads(self) -> None:
        """Convert sparse gradients (from Engram nn.Embedding(sparse=True)) to dense
        so that clip_grad_norm_ works."""
        for p in self.model.parameters():
            if p.grad is not None and p.grad.is_sparse:
                p.grad = p.grad.to_dense()

    def _clip_grad_norm(self, *args, **kwargs):
        self._densify_sparse_grads()
        return super()._clip_grad_norm(*args, **kwargs)

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs,
    ):
        # MTP curriculum: step-dependent lambda
        if getattr(model.config, "use_mtp", False) and self.state.max_steps > 0:
            lam = _mtp_lambda_schedule(
                self.state.global_step,
                self.state.max_steps,
                model.config.mtp_lambda,
            )
            inputs["mtp_lambda"] = lam

        outputs = model(**inputs)
        loss = (
            outputs.loss
            if outputs.loss is not None
            else compute_lm_loss(outputs.logits, inputs["labels"], model.config.vocab_size)
        )
        return (loss, outputs) if return_outputs else loss
