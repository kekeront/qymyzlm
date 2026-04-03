"""Grouped Query Attention with RoPE and Flash Attention 2."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from kazllm.model.config import KazLLMConfig
from kazllm.model.rope import apply_rotary, precompute_freqs


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: KazLLMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        # Register RoPE freq cache as buffer
        cos, sin = precompute_freqs(
            self.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )
        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)

        self.use_flash_attention = config.use_flash_attention

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary(q, k, self.cos_cache, self.sin_cache)

        # Expand KV heads to match Q heads (GQA)
        k = k.repeat_interleave(self.kv_groups, dim=1)
        v = v.repeat_interleave(self.kv_groups, dim=1)

        if self.use_flash_attention:
            # PyTorch 2.2+ dispatches to Flash Attention 2 kernel automatically.
            # Pass attn_mask for padding support (4-D additive float mask or 2-D bool).
            # When training on packed sequences (attention_mask=None) is_causal=True
            # alone is sufficient.  For padded evaluation batches the mask must be
            # provided; without it, padding tokens would attend to each other and
            # corrupt perplexity measurements.
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask, is_causal=(attention_mask is None)
            )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            # Causal mask
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
            if attention_mask is not None:
                scores = scores + attention_mask
            out = F.softmax(scores, dim=-1) @ v

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)
