"""Rotary Position Embeddings (RoPE)."""

import torch
from torch import Tensor


def precompute_freqs(
    head_dim: int,
    max_seq_len: int,
    theta: float = 500_000.0,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    """Precompute cos/sin frequency cache."""
    assert head_dim % 2 == 0
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)  # (seq_len, head_dim/2)
    freqs = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
    return freqs.cos(), freqs.sin()


def rotate_half(x: Tensor) -> Tensor:
    """Rotate interleaved halves for RoPE application."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Apply rotary embeddings to query and key tensors.

    Args:
        q: (batch, heads, seq_len, head_dim)
        k: (batch, kv_heads, seq_len, head_dim)
        cos: (seq_len, head_dim)
        sin: (seq_len, head_dim)
    """
    cos = cos[: q.shape[2]].unsqueeze(0).unsqueeze(0)  # (1, 1, seq, head_dim)
    sin = sin[: q.shape[2]].unsqueeze(0).unsqueeze(0)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot
