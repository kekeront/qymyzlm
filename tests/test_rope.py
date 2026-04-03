"""Numerical correctness of RoPE implementation."""

import torch

from kazllm.model.rope import apply_rotary, precompute_freqs, rotate_half


def test_rotate_half():
    x = torch.arange(8, dtype=torch.float).reshape(1, 1, 1, 8)
    rotated = rotate_half(x)
    assert rotated.shape == x.shape
    # First half should be negated second half
    assert torch.allclose(rotated[..., :4], -x[..., 4:])
    assert torch.allclose(rotated[..., 4:], x[..., :4])


def test_precompute_freqs_shape():
    cos, sin = precompute_freqs(head_dim=64, max_seq_len=128, theta=10000.0)
    assert cos.shape == (128, 64)
    assert sin.shape == (128, 64)


def test_apply_rotary_shape():
    cos, sin = precompute_freqs(head_dim=16, max_seq_len=32, theta=10000.0)
    q = torch.randn(2, 4, 8, 16)  # batch, heads, seq, head_dim
    k = torch.randn(2, 2, 8, 16)  # batch, kv_heads, seq, head_dim
    q_rot, k_rot = apply_rotary(q, k, cos, sin)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_rope_norm_preservation():
    """RoPE should preserve the norm of query/key vectors."""
    cos, sin = precompute_freqs(head_dim=32, max_seq_len=16)
    q = torch.randn(1, 1, 4, 32)
    k = torch.randn(1, 1, 4, 32)
    q_rot, k_rot = apply_rotary(q, k, cos, sin)
    assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5)
    assert torch.allclose(k.norm(dim=-1), k_rot.norm(dim=-1), atol=1e-5)
