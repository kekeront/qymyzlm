"""Main pretraining entry-point for KazLLM."""

import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from transformers import TrainingArguments

from kazllm.model.config import KazLLMConfig
from kazllm.model.model import KazLLMForCausalLM
from kazllm.training.callbacks import ThroughputCallback
from kazllm.training.trainer import KazLLMTrainer, ShardedMemmapDataset
from kazllm.utils.logging import setup_logging
from kazllm.utils.seed import set_seed

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    set_seed(cfg.seed)

    # Build model
    model_cfg = KazLLMConfig(
        vocab_size=cfg.model.vocab_size,
        hidden_size=cfg.model.hidden_size,
        intermediate_size=cfg.model.intermediate_size,
        num_hidden_layers=cfg.model.num_hidden_layers,
        num_attention_heads=cfg.model.num_attention_heads,
        num_key_value_heads=cfg.model.num_key_value_heads,
        max_position_embeddings=cfg.model.max_position_embeddings,
        rope_theta=cfg.model.rope_theta,
        rms_norm_eps=cfg.model.rms_norm_eps,
        tie_word_embeddings=cfg.model.get("tie_word_embeddings", False),
        use_flash_attention=cfg.model.use_flash_attention,
        # mHC
        use_mhc=cfg.model.use_mhc,
        mhc_streams=cfg.model.mhc_streams,
        # Engram
        use_engram=cfg.model.use_engram,
        engram_layer_indices=list(cfg.model.engram_layer_indices),
        engram_ngram_orders=list(cfg.model.engram_ngram_orders),
        engram_num_heads=cfg.model.engram_num_heads,
        engram_table_size=cfg.model.engram_table_size,
        engram_slot_dim=cfg.model.engram_slot_dim,
        engram_conv_kernel_size=cfg.model.engram_conv_kernel_size,
        # MTP
        use_mtp=cfg.model.get("use_mtp", False),
        mtp_depth=cfg.model.get("mtp_depth", 1),
        mtp_lambda=cfg.model.get("mtp_lambda", 0.3),
    )

    if cfg.training.from_pretrained:
        log.info(f"Loading base model from {cfg.training.from_pretrained}")
        model = KazLLMForCausalLM.from_pretrained(cfg.training.from_pretrained, config=model_cfg)
    else:
        log.info("Initializing model from scratch")
        model = KazLLMForCausalLM(model_cfg)

    num_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters: {num_params / 1e9:.2f}B")

    # Load dataset
    manifest = Path(cfg.training.tokenized_dir) / "manifest.json"
    train_dataset = ShardedMemmapDataset(
        manifest_path=str(manifest),
        context_length=cfg.training.context_length,
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
    max_steps = cfg.training.get("max_steps", 0)
    if max_steps > 0:
        total_steps = min(total_steps, max_steps)

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
        fp16=cfg.training.get("fp16", False),
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
    )

    trainer = KazLLMTrainer(
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

    trainer.train(resume_from_checkpoint=cfg.training.resume_from_checkpoint)
    log.info("Training complete.")


if __name__ == "__main__":
    main()
