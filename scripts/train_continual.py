"""Continual pretraining entry-point for QymyzLM (Qwen + Engram).

Unlike train.py (which builds KazLLMForCausalLM from scratch), this script
loads a pre-built QymyzLM checkpoint (from build_continual_pt.py) and trains
it with English data mixing to prevent catastrophic forgetting.

Usage:
    python scripts/train_continual.py training=pretrain_continual
"""

import json
import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from kazllm.training.callbacks import ThroughputCallback
from kazllm.utils.logging import setup_logging
from kazllm.utils.seed import set_seed

log = logging.getLogger(__name__)


class MixedLanguageDataset(Dataset):
    """Dataset that samples from two shard pools (primary + secondary) with a configurable ratio.

    Used for continual PT: primary = Kazakh (90%), secondary = English (10%).
    Both use the same expanded tokenizer so vocab is consistent.
    """

    def __init__(
        self,
        primary_manifest: str,
        context_length: int = 2048,
        secondary_manifest: str | None = None,
        secondary_ratio: float = 0.1,
    ):
        self.context_length = context_length
        self.secondary_ratio = secondary_ratio if secondary_manifest else 0.0

        self._primary_shards, self._primary_offsets, self._primary_total = self._load_manifest(
            primary_manifest
        )
        self._secondary_shards = []
        self._secondary_offsets = [0]
        self._secondary_total = 0

        if secondary_manifest and Path(secondary_manifest).exists():
            self._secondary_shards, self._secondary_offsets, self._secondary_total = (
                self._load_manifest(secondary_manifest)
            )
            log.info(
                f"Mixed dataset: {self._primary_total} primary + "
                f"{self._secondary_total} secondary seqs "
                f"(ratio={self.secondary_ratio:.0%} secondary)"
            )
        else:
            self.secondary_ratio = 0.0
            log.info(f"Primary-only dataset: {self._primary_total} sequences")

    def _load_manifest(self, manifest_path: str):
        with open(manifest_path) as f:
            manifest = json.load(f)

        dtype_str = manifest.get("dtype", "uint16")
        dtype = np.dtype(dtype_str)
        shards = []
        offsets = [0]

        for shard_info in manifest["shards"]:
            arr = np.memmap(shard_info["shard_path"], dtype=dtype, mode="r")
            shards.append(arr)
            offsets.append(offsets[-1] + len(arr) // self.context_length)

        return shards, offsets, offsets[-1]

    def __len__(self) -> int:
        return self._primary_total

    def _get_from_pool(self, idx: int, shards, offsets) -> dict[str, torch.Tensor]:
        import bisect

        shard_idx = bisect.bisect_right(offsets, idx) - 1
        local_idx = idx - offsets[shard_idx]
        start = local_idx * self.context_length
        tokens = shards[shard_idx][start : start + self.context_length].astype(np.int64)
        input_ids = torch.from_numpy(tokens)
        return {"input_ids": input_ids, "labels": input_ids.clone()}

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Probabilistically sample from secondary pool
        if self._secondary_total > 0 and torch.rand(1).item() < self.secondary_ratio:
            sec_idx = torch.randint(0, self._secondary_total, (1,)).item()
            return self._get_from_pool(sec_idx, self._secondary_shards, self._secondary_offsets)

        idx = idx % self._primary_total
        return self._get_from_pool(idx, self._primary_shards, self._primary_offsets)


class QymyzTrainer(Trainer):
    """Trainer for QymyzLM continual pretraining.

    Handles Engram parameter groups (5x LR for hash tables, wd=0) and
    sparse gradient densification.
    """

    ENGRAM_LR_MULT = 5.0

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        from kazllm.model.qwen_engram_wrapper import QymyzForCausalLM

        engram_table_params = []
        engram_other_params = []
        base_params = []

        model = self.model
        if isinstance(model, QymyzForCausalLM):
            engram_table_params = list(model.engram_table_parameters())
            engram_other_params = list(model.engram_non_table_parameters())
            # Base model params = everything except Engram
            engram_param_ids = {id(p) for p in engram_table_params} | {
                id(p) for p in engram_other_params
            }
            base_params = [
                p for p in model.parameters()
                if id(p) not in engram_param_ids and p.requires_grad
            ]
        else:
            base_params = [p for p in model.parameters() if p.requires_grad]

        base_lr = self.args.learning_rate
        param_groups = [
            {"params": base_params, "lr": base_lr, "weight_decay": self.args.weight_decay},
        ]
        if engram_other_params:
            param_groups.append({
                "params": engram_other_params,
                "lr": base_lr,
                "weight_decay": self.args.weight_decay,
            })
        if engram_table_params:
            param_groups.append({
                "params": engram_table_params,
                "lr": base_lr * self.ENGRAM_LR_MULT,
                "weight_decay": 0.0,
            })
            log.info(
                f"Engram optimizer: {len(engram_table_params)} table params at "
                f"{base_lr * self.ENGRAM_LR_MULT:.1e} LR (5x), wd=0"
            )

        self.optimizer = torch.optim.AdamW(param_groups)
        return self.optimizer

    def _densify_sparse_grads(self) -> None:
        for p in self.model.parameters():
            if p.grad is not None and p.grad.is_sparse:
                p.grad = p.grad.to_dense()

    def _clip_grad_norm(self, *args, **kwargs):
        self._densify_sparse_grads()
        return super()._clip_grad_norm(*args, **kwargs)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    set_seed(cfg.seed)

    # Load pre-built QymyzLM checkpoint
    checkpoint_dir = Path(cfg.training.continual_pt_checkpoint)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"QymyzLM checkpoint not found at {checkpoint_dir}. "
            "Run scripts/build_continual_pt.py first."
        )

    from scripts.build_continual_pt import load_checkpoint

    model, tokenizer = load_checkpoint(checkpoint_dir)
    num_params = sum(p.numel() for p in model.parameters())
    log.info(f"QymyzLM loaded: {num_params / 1e9:.2f}B params")

    # Dataset
    primary_manifest = Path(cfg.training.tokenized_dir) / "manifest.json"
    secondary_manifest = None
    if hasattr(cfg.training, "english_tokenized_dir") and cfg.training.english_tokenized_dir:
        secondary_manifest = str(Path(cfg.training.english_tokenized_dir) / "manifest.json")

    train_dataset = MixedLanguageDataset(
        primary_manifest=str(primary_manifest),
        context_length=cfg.training.context_length,
        secondary_manifest=secondary_manifest,
        secondary_ratio=cfg.training.get("english_ratio", 0.1),
    )
    total_seqs = len(train_dataset)
    total_btokens = total_seqs * cfg.training.context_length / 1e9
    log.info(f"Training on {total_seqs:,} sequences ({total_btokens:.2f}B tokens)")

    # Compute steps
    num_gpus = max(1, torch.cuda.device_count())
    effective_batch = (
        cfg.training.per_device_batch_size
        * cfg.training.gradient_accumulation_steps
        * num_gpus
    )
    tokens_per_step = effective_batch * cfg.training.context_length
    total_steps = cfg.training.total_tokens // tokens_per_step

    log.info(
        f"Effective batch: {effective_batch} seqs, {tokens_per_step:,} tokens/step, "
        f"{total_steps:,} total steps"
    )

    training_args = TrainingArguments(
        output_dir=cfg.training.checkpoint_dir,
        per_device_train_batch_size=cfg.training.per_device_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        max_grad_norm=cfg.training.max_grad_norm,
        warmup_steps=cfg.training.warmup_steps,
        max_steps=total_steps,
        bf16=cfg.training.bf16,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        fsdp=cfg.training.fsdp if cfg.training.fsdp else "",
        save_steps=cfg.training.save_steps,
        logging_steps=cfg.training.logging_steps,
        eval_strategy="no",
        report_to=["wandb"],
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        run_name="qymyz1_5b_continual",
    )

    # Enable gradient checkpointing
    if cfg.training.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    trainer = QymyzTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[
            ThroughputCallback(
                context_length=cfg.training.context_length,
                model_params=num_params,
            )
        ],
    )

    trainer.train(resume_from_checkpoint=cfg.training.get("resume_from_checkpoint"))
    log.info("QymyzLM continual pretraining complete.")


if __name__ == "__main__":
    main()
