"""SwiGLU feed-forward network."""

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from kazllm.model.config import KazLLMConfig


class SwiGLUMLP(nn.Module):
    def __init__(self, config: KazLLMConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
