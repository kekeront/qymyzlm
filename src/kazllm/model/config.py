"""KazLLM model configuration."""

from transformers import PretrainedConfig


class KazLLMConfig(PretrainedConfig):
    model_type = "kazllm"

    def __init__(
        self,
        # --- Core transformer dims ---
        vocab_size: int = 50_000,
        hidden_size: int = 2048,
        intermediate_size: int = 5504,
        num_hidden_layers: int = 22,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        max_position_embeddings: int = 4096,
        rope_theta: float = 500_000.0,
        rms_norm_eps: float = 1e-5,
        tie_word_embeddings: bool = False,
        use_flash_attention: bool = True,
        # --- mHC (Manifold-Constrained Hyper-Connections) ---
        use_mhc: bool = True,
        mhc_streams: int = 4,  # n: number of parallel residual streams
        # --- Engram N-gram memory ---
        use_engram: bool = True,
        engram_layer_indices: list | None = None,  # layers where Engram is injected
        engram_ngram_orders: list | None = None,  # default: [2, 3]
        engram_num_heads: int = 4,  # K hash heads per N-gram order
        engram_table_size: int = 500_003,  # M slots per table (prime)
        engram_slot_dim: int = 64,  # d per slot; total d_mem = orders*heads*slot_dim
        engram_conv_kernel_size: int = 4,  # depthwise causal conv kernel
        # --- MTP (Multi-Token Prediction, DeepSeek-V3 style) ---
        use_mtp: bool = False,
        mtp_depth: int = 1,  # D: number of additional tokens to predict
        mtp_lambda: float = 0.3,  # MTP loss weight
        # --- Special tokens ---
        pad_token_id: int = 0,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.use_flash_attention = use_flash_attention
        # mHC
        self.use_mhc = use_mhc
        self.mhc_streams = mhc_streams
        # Engram
        self.use_engram = use_engram
        self.engram_layer_indices = engram_layer_indices or [2, num_hidden_layers // 4]
        self.engram_ngram_orders = engram_ngram_orders or [2, 3]
        self.engram_num_heads = engram_num_heads
        self.engram_table_size = engram_table_size
        self.engram_slot_dim = engram_slot_dim
        self.engram_conv_kernel_size = engram_conv_kernel_size
        # MTP
        self.use_mtp = use_mtp
        self.mtp_depth = mtp_depth
        self.mtp_lambda = mtp_lambda
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def engram_d_mem(self) -> int:
        """Total retrieved memory dimension per Engram module."""
        return len(self.engram_ngram_orders) * self.engram_num_heads * self.engram_slot_dim

    @property
    def engram_params_per_module(self) -> int:
        """Approximate parameter count per Engram module."""
        num_tables = len(self.engram_ngram_orders) * self.engram_num_heads
        return num_tables * self.engram_table_size * self.engram_slot_dim
