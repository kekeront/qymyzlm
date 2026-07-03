"""RMSNorm (pre-norm, Llama-style) used by Engram's internal normalisation."""

import torch
import torch.nn as nn
from torch import Tensor


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation.

    y = x / sqrt(mean(x^2) + eps) * weight
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)
