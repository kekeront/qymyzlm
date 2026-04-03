"""Tests for mHC: stream management, Sinkhorn-Knopp, signal stability."""

import torch

from kazllm.model.mhc import (
    MHCStreamManager,
    collapse_streams,
    expand_to_streams,
    sinkhorn_knopp,
)


def test_sinkhorn_knopp_doubly_stochastic():
    """Output should have row sums ≈ 1 and column sums ≈ 1."""
    logits = torch.randn(2, 4, 4, 4)
    result = sinkhorn_knopp(logits, num_iters=50)
    assert torch.allclose(result.sum(dim=-1), torch.ones_like(result.sum(dim=-1)), atol=1e-4)
    assert torch.allclose(result.sum(dim=-2), torch.ones_like(result.sum(dim=-2)), atol=1e-4)


def test_sinkhorn_knopp_non_negative():
    """Output should be non-negative."""
    logits = torch.randn(1, 1, 3, 3)
    result = sinkhorn_knopp(logits)
    assert (result >= 0).all()


def test_expand_collapse_roundtrip():
    x = torch.randn(2, 8, 32)
    expanded = expand_to_streams(x, n_streams=4)
    assert expanded.shape == (2, 8, 4, 32)
    # All n copies should be identical
    assert torch.allclose(expanded[:, :, 0, :], expanded[:, :, 1, :])
    # Collapse = mean → same as original (since all streams are copies)
    collapsed = collapse_streams(expanded)
    assert torch.allclose(collapsed, x)


def test_mhc_stream_manager_output_shape():
    B, T, n, C = 2, 16, 4, 64
    manager = MHCStreamManager(hidden_size=C, n_streams=n)
    x = torch.randn(B, T, n, C)

    def identity_fn(h, **kwargs):
        return h

    out = manager(x, identity_fn)
    assert out.shape == (B, T, n, C)


def test_mhc_near_identity_at_init():
    """At initialisation (alpha ≈ 0), the output should be close to input.

    With alpha=0: H_pre ≈ [1/n,...,1/n], H_post ≈ 1, H_res ≈ I.
    Then: x_out = I @ x + 1 ⊗ F(avg(x))
    For identity F: x_out[k] = x_mean + 1 * x_mean ≠ x[k] exactly, but bounded.

    More practically: the sum of outputs should match approximately.
    """
    B, T, n, C = 1, 4, 4, 32
    manager = MHCStreamManager(hidden_size=C, n_streams=n)
    manager.eval()

    x = torch.ones(B, T, n, C)
    with torch.no_grad():
        out = manager(x, lambda h, **kw: h)

    # Formula: x_{l+1} = H_res @ x + H_post ⊗ F(H_pre · x)
    # At init: H_res ≈ I, H_post ≈ 1, H_pre ≈ uniform → out ≈ x + F(mean(x)) ≈ 2x
    assert out.shape == (B, T, n, C)
    assert not out.isnan().any()
    assert not out.isinf().any()
    # Output ≈ 2× input for all-ones with identity F (standard mHC residual structure)
    assert torch.allclose(out.mean(), 2.0 * x.mean(), atol=0.2)


def test_mhc_gradient_flow():
    """Gradients should flow back through the mHC block without NaN."""
    B, T, n, C = 2, 8, 4, 32
    manager = MHCStreamManager(hidden_size=C, n_streams=n)
    x = torch.randn(B, T, n, C, requires_grad=True)

    linear = torch.nn.Linear(C, C, bias=False)

    def layer_fn(h, **kwargs):
        return linear(h)

    out = manager(x, layer_fn)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert not x.grad.isnan().any(), "NaN gradients in mHC backward pass"
