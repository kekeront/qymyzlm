"""Transformer block with optional mHC n-stream residual management.

Standard mode (use_mhc=False):
    x = x + attn(norm1(x))
    x = x + mlp(norm2(x))

mHC mode (use_mhc=True):
    The block receives and returns (B, T, n, C) n-stream tensors.
    Two MHCStreamManagers govern the attn and MLP sub-layers independently,
    each applying: x_streams = H_res @ x_streams + H_post ⊗ F(H_pre · x_streams)
"""

import torch.nn as nn
from torch import Tensor

from kazllm.model.attention import GroupedQueryAttention
from kazllm.model.config import KazLLMConfig
from kazllm.model.mhc import MHCStreamManager
from kazllm.model.mlp import SwiGLUMLP
from kazllm.model.norm import RMSNorm


class TransformerBlock(nn.Module):
    """Transformer block: supports both standard residual and mHC n-stream residual.

    When use_mhc=True, forward() expects and returns (B, T, n, C) tensors.
    The attention and MLP computations are unchanged (operate on (B, T, C));
    mHC manages how the C-dim outputs are mixed back into the n-stream residual.
    """

    def __init__(self, config: KazLLMConfig):
        super().__init__()
        self.use_mhc = config.use_mhc

        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config)

        if config.use_mhc:
            self.mhc_attn = MHCStreamManager(config.hidden_size, config.mhc_streams)
            self.mhc_mlp = MHCStreamManager(config.hidden_size, config.mhc_streams)

    def _forward_standard(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        x = x + self.self_attn(self.input_layernorm(x), attention_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

    def _forward_mhc(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """x: (B, T, n, C) → (B, T, n, C)"""

        # Attention sub-layer via mHC
        def attn_fn(h: Tensor, attention_mask=None) -> Tensor:
            return self.self_attn(self.input_layernorm(h), attention_mask)

        x = self.mhc_attn(x, attn_fn, attention_mask=attention_mask)

        # MLP sub-layer via mHC
        def mlp_fn(h: Tensor) -> Tensor:
            return self.mlp(self.post_attention_layernorm(h))

        x = self.mhc_mlp(x, mlp_fn)
        return x

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        if self.use_mhc:
            return self._forward_mhc(x, attention_mask)
        return self._forward_standard(x, attention_mask)
