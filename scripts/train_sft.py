"""LoRA instruction fine-tuning for KazLLM chat capability."""

import logging

import hydra
import torch
from datasets import load_dataset
from omegaconf import DictConfig
from peft import LoraConfig, TaskType, get_peft_model
from transformers import PreTrainedTokenizerFast, TrainingArguments

from kazllm.model.model import KazLLMForCausalLM
from kazllm.training.trainer import KazLLMTrainer
from kazllm.utils.logging import setup_logging
from kazllm.utils.seed import set_seed

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    set_seed(cfg.seed)

    model = KazLLMForCausalLM.from_pretrained(
        cfg.training.base_model,
        torch_dtype=torch.bfloat16 if cfg.training.bf16 else torch.float32,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.training.lora_rank,
        lora_alpha=cfg.training.lora_alpha,
        lora_dropout=cfg.training.lora_dropout,
        target_modules=list(cfg.training.target_modules),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.training.base_model)

    # Load instruction data (jsonl with "instruction", "input", "output" fields)
    dataset = load_dataset("json", data_files=cfg.training.data_path, split="train")

    def format_example(ex):
        if ex.get("input"):
            prompt = (
                f"<|system|>Сен пайдалы көмекшісің."
                f"<|user|>{ex['instruction']}\n{ex['input']}<|assistant|>"
            )
        else:
            prompt = f"<|system|>Сен пайдалы көмекшісің.<|user|>{ex['instruction']}<|assistant|>"
        full = prompt + ex["output"] + tokenizer.eos_token
        ids = tokenizer(full, max_length=cfg.training.max_seq_length, truncation=True)
        ids["labels"] = ids["input_ids"].copy()
        return ids

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        per_device_train_batch_size=cfg.training.per_device_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        num_train_epochs=cfg.training.num_epochs,
        bf16=cfg.training.bf16,
        save_steps=cfg.training.save_steps,
        logging_steps=10,
        report_to=["wandb"],
    )

    trainer = KazLLMTrainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()
    log.info("SFT complete.")


if __name__ == "__main__":
    main()
