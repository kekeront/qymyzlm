"""Structured config dataclasses for Hydra."""

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    cleaned_dir: str = "data/cleaned"
    deduped_dir: str = "data/deduped"
    lang_id_threshold: float = 0.75
    min_chars: int = 50
    max_chars: int = 100_000
    min_cyrillic_ratio: float = 0.50


@dataclass
class TokenizerConfig:
    model_type: str = "unigram"
    vocab_size: int = 50_000
    character_coverage: float = 0.9999
    byte_fallback: bool = True
    add_dummy_prefix: bool = False
    max_sentence_length: int = 8192
    num_threads: int = 16
    sampling_sentences: int = 10_000_000
    output_dir: str = "data/tokenizer/kaz_sp_unigram_50k"


@dataclass
class ModelConfig:
    vocab_size: int = 50_000
    hidden_size: int = 2048
    intermediate_size: int = 5504
    num_hidden_layers: int = 22
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    max_position_embeddings: int = 4096
    rope_theta: float = 500_000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False
    use_flash_attention: bool = True


@dataclass
class TrainingConfig:
    total_tokens: int = 9_000_000_000
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    context_length: int = 2048
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    warmup_steps: int = 2000
    bf16: bool = True
    gradient_checkpointing: bool = True
    fsdp: str = "full_shard auto_wrap"
    save_steps: int = 1000
    logging_steps: int = 10
    eval_steps: int = 500
    seed: int = 42
    tokenized_dir: str = "data/tokenized"
    checkpoint_dir: str = "checkpoints/kaz1b_pretrain"
    from_pretrained: str | None = None
    resume_from_checkpoint: str | None = None
    max_steps: int = -1


@dataclass
class EvalConfig:
    benchmarks: list = field(default_factory=list)
    model_dtype: str = "bfloat16"
    results_dir: str = "evals"


@dataclass
class KazLLMConfig:
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    output_dir: str = "outputs"
    seed: int = 42
