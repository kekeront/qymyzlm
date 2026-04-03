"""
Manifold-Constrained Hyper-Connections (mHC).

Expands the residual stream from 1×C to n×C streams with a doubly stochastic
residual mixing matrix H_res, ensuring gradient norm stays bounded at any depth.

The key equations per block (treating attn+FFN as a single layer function F):

    H_pre  = σ(α_pre  · tanh(x̃ φ_pre)  + b_pre)          # (B,T,n), non-negative
    H_post = 2σ(α_post · tanh(x̃ φ_post) + b_post)          # (B,T,n), non-negative
    H_res  = SinkhornKnopp(α_res · tanh(x̃ φ_res) + b_res)  # (B,T,n,n), doubly stochastic

    x_in  = sum_k H_pre[k] * x_streams[k]       # (B,T,C)  — aggregate n→1
    out   = F(x_in)                               # (B,T,C)  — standard Transformer block
    x_{l+1} = H_res @ x_streams + H_post ⊗ out  # (B,T,n,C) — n-stream output

Sinkhorn-Knopp on H_res guarantees the composite mapping ∏ H_res_l is also doubly
stochastic → spectral norm ≤ 1 → signal preserved at any depth.

Reference: "mHC: Manifold-Constrained Hyper-Connections" (arXiv 2512.24880v2)
"""

import torch
import torch.nn as nn
from torch import Tensor


def sinkhorn_knopp(logits: Tensor, num_iters: int = 20) -> Tensor:
    """Project matrix onto doubly stochastic manifold (Birkhoff polytope).

    Args:
        logits: (..., n, n) unnormalized log-scale values
        num_iters: Sinkhorn iterations (20 is the value used in the paper)
    Returns:
        (..., n, n) doubly stochastic matrix (all rows and columns sum to 1)
    """
    # Subtract per-row max before exp for numerical stability (log-sum-exp trick).
    # Without this, logits > ~88 overflow to inf in float32/bf16, which turns
    # the entire Sinkhorn result — and gradients — to NaN.
    m = (logits - logits.amax(dim=-1, keepdim=True)).exp()
    for _ in range(num_iters):
        m = m / m.sum(dim=-1, keepdim=True).clamp_min(1e-9)  # row normalize
        m = m / m.sum(dim=-2, keepdim=True).clamp_min(1e-9)  # column normalize
    return m


class MHCStreamManager(nn.Module):
    """Single mHC residual update for one transformer sub-layer.

    Manages the n-stream residual given a layer function F: R^C → R^C.
    The input x_streams has shape (B, T, n, C); the output has the same shape.

    All mapping parameters are dynamic (input-dependent) + static (learned bias).
    Dynamic contributions start near zero (α parameters initialised to 0), so the
    model trains identically to a standard transformer at initialisation.
    """

    def __init__(self, hidden_size: int, n_streams: int = 4):
        super().__init__()
        self.n = n_streams
        self.C = hidden_size
        nC = n_streams * hidden_size

        # Dynamic projections (x̃ → mapping logits)
        self.phi_pre = nn.Linear(nC, n_streams, bias=False)
        self.phi_post = nn.Linear(nC, n_streams, bias=False)
        self.phi_res = nn.Linear(nC, n_streams * n_streams, bias=False)

        # Static bias terms
        # b_pre: init so H_pre ≈ [1/n, ..., 1/n] → sigmoid^{-1}(1/n)
        logit_1_over_n = torch.log(torch.tensor(1.0 / (n_streams - 1.0)))
        self.b_pre = nn.Parameter(logit_1_over_n.expand(n_streams).clone())
        # b_post: init so H_post ≈ 1 (2σ = 1 → σ^{-1}(0.5) = 0)
        self.b_post = nn.Parameter(torch.zeros(n_streams))
        # b_res: init so H_res ≈ I (large positive on diagonal, 0 off)
        b_res_init = torch.zeros(n_streams, n_streams)
        b_res_init.fill_diagonal_(5.0)  # exp(5)≈148 >> exp(0)=1 → near-identity after Sinkhorn
        self.b_res = nn.Parameter(b_res_init.flatten())

        # Scale factors for dynamic components (init 0 = pure static at training start)
        self.alpha_pre = nn.Parameter(torch.zeros(1))
        self.alpha_post = nn.Parameter(torch.zeros(1))
        self.alpha_res = nn.Parameter(torch.zeros(1))

        # RMSNorm on flattened n-stream (no learnable params — just normalisation)
        self.norm = nn.RMSNorm(nC, elementwise_affine=False)

        # Init dynamic projection weights to near-zero
        nn.init.normal_(self.phi_pre.weight, std=0.01)
        nn.init.normal_(self.phi_post.weight, std=0.01)
        nn.init.normal_(self.phi_res.weight, std=0.01)

    def _compute_mappings(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute H_pre, H_post, H_res from n-stream input.

        Args:
            x: (B, T, n, C)
        Returns:
            H_pre:  (B, T, n)
            H_post: (B, T, n)
            H_res:  (B, T, n, n)
        """
        B, T, n, C = x.shape
        x_flat = x.view(B, T, n * C)
        x_norm = self.norm(x_flat)

        # Dynamic contributions (small at init due to alpha ≈ 0)
        dyn_pre = self.alpha_pre * torch.tanh(self.phi_pre(x_norm))  # (B,T,n)
        dyn_post = self.alpha_post * torch.tanh(self.phi_post(x_norm))  # (B,T,n)
        dyn_res = self.alpha_res * torch.tanh(self.phi_res(x_norm))  # (B,T,n*n)

        H_pre_logit = dyn_pre + self.b_pre  # (B,T,n)
        H_post_logit = dyn_post + self.b_post  # (B,T,n)
        H_res_logit = (dyn_res + self.b_res).view(B, T, n, n)  # (B,T,n,n)

        H_pre = torch.sigmoid(H_pre_logit)  # ∈ (0,1)
        H_post = 2.0 * torch.sigmoid(H_post_logit)  # ∈ (0,2)
        H_res = sinkhorn_knopp(H_res_logit)  # doubly stochastic

        return H_pre, H_post, H_res

    def forward(
        self,
        x: Tensor,
        layer_fn,
        **layer_kwargs,
    ) -> Tensor:
        """Apply one mHC residual update.

        Args:
            x: (B, T, n, C) n-stream input
            layer_fn: callable (B,T,C) → (B,T,C)  [attention or MLP, pre-normed outside]
            **layer_kwargs: forwarded to layer_fn
        Returns:
            (B, T, n, C) updated n-stream
        """
        H_pre, H_post, H_res = self._compute_mappings(x)

        # Aggregate n streams → single C-dim layer input
        # H_pre: (B,T,n,1) broadcast * x: (B,T,n,C) → sum dim 2
        x_in = (H_pre.unsqueeze(-1) * x).sum(dim=2)  # (B,T,C)

        # Standard layer computation on C-dim input
        out = layer_fn(x_in, **layer_kwargs)  # (B,T,C)

        # Broadcast output to n streams
        # H_post: (B,T,n,1) * out: (B,T,1,C) → (B,T,n,C)
        x_post = H_post.unsqueeze(-1) * out.unsqueeze(2)

        # Mix streams via doubly stochastic H_res
        # H_res: (B,T,n,n), x: (B,T,n,C) → (B,T,n,C)
        x_res = torch.einsum("btnm,btmc->btnc", H_res, x)

        return x_res + x_post


def expand_to_streams(x: Tensor, n_streams: int) -> Tensor:
    """Expand single-stream (B,T,C) to n-stream (B,T,n,C) by copying."""
    return x.unsqueeze(2).expand(-1, -1, n_streams, -1).contiguous()


def collapse_streams(x: Tensor) -> Tensor:
    """Collapse n-stream (B,T,n,C) to single-stream (B,T,C) by averaging."""
    return x.mean(dim=2)
