"""QymyzLM: Engram wrapper for Qwen-2.5 (or any HuggingFace causal LM).

Grafts EngramModule onto a pretrained Qwen model by wrapping specified decoder
layers. The base model is NOT modified — Engram modules are attached as separate
nn.Module instances that inject memory before each wrapped layer's computation.

Usage:
    from transformers import AutoModelForCausalLM
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B")
    model = QymyzForCausalLM.from_base(base, engram_layer_indices=[2, 7])
"""

import logging
from dataclasses import dataclass

import torch.nn as nn
from torch import Tensor

from kazllm.model.engram import EngramModule

log = logging.getLogger(__name__)


@dataclass
class EngramConfig:
    """Engram hyperparameters for grafting onto a pretrained model."""

    layer_indices: list[int] = None
    ngram_orders: list[int] = None
    num_heads: int = 4
    table_size: int = 500_003
    slot_dim: int = 64
    conv_kernel_size: int = 4

    def __post_init__(self):
        if self.layer_indices is None:
            self.layer_indices = [2, 7]
        if self.ngram_orders is None:
            self.ngram_orders = [2, 3]


class EngramWrappedLayer(nn.Module):
    """Wraps a single decoder layer to inject Engram memory before its forward pass."""

    def __init__(self, original_layer: nn.Module, engram: EngramModule):
        super().__init__()
        self.layer = original_layer
        self.engram = engram

    def __getattr__(self, name: str):
        """Proxy attribute lookups to the original layer for compatibility.

        HuggingFace may access layer attributes like `attention_type` during
        forward — delegate anything not found on this wrapper to the original.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.layer, name)

    def forward(self, hidden_states: Tensor, **kwargs) -> tuple:
        # Retrieve input_ids stashed by the top-level model
        input_ids = getattr(self, "_current_input_ids", None)
        if input_ids is not None:
            hidden_states = hidden_states + self.engram(hidden_states, input_ids)
        return self.layer(hidden_states, **kwargs)


class QymyzForCausalLM(nn.Module):
    """Qwen + Engram wrapper for continual pretraining.

    Wraps a HuggingFace Qwen2ForCausalLM (or compatible) model with Engram
    memory injection at specified layers. The base model's weights are preserved;
    only Engram modules are added as new parameters.

    The wrapper threads `input_ids` to wrapped layers via a stashed attribute,
    since HuggingFace decoder layers don't receive input_ids in their forward.
    """

    def __init__(self, base_model: nn.Module, engram_config: EngramConfig):
        super().__init__()
        self.base_model = base_model
        self.engram_config = engram_config

        # Access base model's config for architecture details
        self.config = base_model.config
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size

        # Create Engram modules and wrap specified layers
        self._wrapped_indices: list[int] = []
        layers = self._get_decoder_layers()

        for idx in engram_config.layer_indices:
            if idx >= len(layers):
                log.warning(f"Layer index {idx} >= num_layers {len(layers)}, skipping")
                continue

            engram = EngramModule(
                hidden_size=hidden_size,
                ngram_orders=engram_config.ngram_orders,
                num_heads=engram_config.num_heads,
                table_size=engram_config.table_size,
                slot_dim=engram_config.slot_dim,
                vocab_size=vocab_size,
                conv_kernel_size=engram_config.conv_kernel_size,
                rms_norm_eps=getattr(self.config, "rms_norm_eps", 1e-6),
            )

            wrapped = EngramWrappedLayer(layers[idx], engram)
            layers[idx] = wrapped
            self._wrapped_indices.append(idx)

        log.info(
            f"QymyzLM: grafted Engram at layers {self._wrapped_indices} "
            f"({sum(p.numel() for p in self.engram_parameters())/1e6:.1f}M new params)"
        )

    def _get_decoder_layers(self) -> nn.ModuleList:
        """Get the decoder layer list from the base model.

        Supports Qwen2 (model.layers) and Llama (model.layers) architectures.
        """
        # Qwen2ForCausalLM: self.model.layers
        if hasattr(self.base_model, "model") and hasattr(self.base_model.model, "layers"):
            return self.base_model.model.layers
        raise ValueError(
            f"Cannot find decoder layers in {type(self.base_model).__name__}. "
            "Expected .model.layers attribute."
        )

    def _stash_input_ids(self, input_ids: Tensor) -> None:
        """Stash input_ids on wrapped layers so they can access them during forward."""
        layers = self._get_decoder_layers()
        for idx in self._wrapped_indices:
            layers[idx]._current_input_ids = input_ids

    def _clear_input_ids(self) -> None:
        """Clear stashed input_ids to avoid memory leaks."""
        layers = self._get_decoder_layers()
        for idx in self._wrapped_indices:
            layers[idx]._current_input_ids = None

    def forward(
        self,
        input_ids: Tensor | None = None,
        labels: Tensor | None = None,
        attention_mask: Tensor | None = None,
        **kwargs,
    ):
        """Forward pass: stash input_ids, run base model, clear."""
        if input_ids is not None:
            self._stash_input_ids(input_ids)

        try:
            outputs = self.base_model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                **kwargs,
            )
        finally:
            self._clear_input_ids()

        return outputs

    def engram_parameters(self):
        """Yield only the Engram parameters (for separate optimizer group)."""
        layers = self._get_decoder_layers()
        for idx in self._wrapped_indices:
            wrapped = layers[idx]
            if isinstance(wrapped, EngramWrappedLayer):
                yield from wrapped.engram.parameters()

    def engram_table_parameters(self):
        """Yield only the Engram hash table parameters (for 5x LR group)."""
        layers = self._get_decoder_layers()
        for idx in self._wrapped_indices:
            wrapped = layers[idx]
            if isinstance(wrapped, EngramWrappedLayer):
                for name, param in wrapped.engram.named_parameters():
                    if "tables." in name:
                        yield param

    def engram_non_table_parameters(self):
        """Yield Engram parameters that are NOT hash tables (projections, norms)."""
        layers = self._get_decoder_layers()
        for idx in self._wrapped_indices:
            wrapped = layers[idx]
            if isinstance(wrapped, EngramWrappedLayer):
                for name, param in wrapped.engram.named_parameters():
                    if "tables." not in name:
                        yield param

    @classmethod
    def from_base(
        cls,
        base_model: nn.Module,
        engram_layer_indices: list[int] | None = None,
        engram_config: EngramConfig | None = None,
    ) -> "QymyzForCausalLM":
        """Create QymyzLM from a pretrained base model.

        Args:
            base_model: A HuggingFace causal LM (e.g. Qwen2ForCausalLM).
            engram_layer_indices: Layer indices to inject Engram. Overrides engram_config.
            engram_config: Full Engram configuration. If None, uses defaults.
        """
        if engram_config is None:
            engram_config = EngramConfig()
        if engram_layer_indices is not None:
            engram_config.layer_indices = engram_layer_indices
        return cls(base_model, engram_config)

    def gradient_checkpointing_enable(self, **kwargs):
        """Delegate gradient checkpointing to base model."""
        self.base_model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        """Delegate gradient checkpointing disable to base model."""
        self.base_model.gradient_checkpointing_disable()

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
