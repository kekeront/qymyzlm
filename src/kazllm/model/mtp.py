"""
Multi-Token Prediction (MTP) module — DeepSeek-V3 style.

Predicts D additional future tokens beyond the standard next-token objective.
Each MTP depth k sequentially combines the previous depth's hidden states with
the embedding of the token predicted at depth k-1, then processes through a
lightweight Transformer block and shared output head.

During inference the MTP modules are discarded entirely — the deployed model
has identical cost to the base architecture.

The MTP block always operates in single-stream mode (no mHC) since it receives
collapsed hidden states from the main model. This keeps MTP lightweight.

Training loss:
    L = L_main + (lambda / D) * sum_{k=1}^{D} L_MTP^k

where L_MTP^k is cross-entropy at prediction depth k.

Reference: DeepSeek-V3 Technical Report (arXiv 2412.19437), Section 3.3
"""

import copy

import torch
import torch.nn as nn
from torch import Tensor

from kazllm.model.block import TransformerBlock
from kazllm.model.config import KazLLMConfig
from kazllm.model.norm import RMSNorm


class MTPModule(nn.Module):
    """Single MTP prediction depth.

    Combines the hidden representation from the previous depth with token
    embeddings, processes through one standard Transformer block, and returns
    hidden states. The shared LM head is applied externally by the caller.

    Always operates in single-stream (B, T, C) — mHC is not used here
    regardless of the main model's mHC setting.
    """

    def __init__(self, config: KazLLMConfig):
        super().__init__()
        C = config.hidden_size
        # Project concatenated [h^{k-1}; embed(token)] from 2C -> C
        self.projection = nn.Linear(2 * C, C, bias=False)
        self.proj_norm = RMSNorm(C, config.rms_norm_eps)

        # One Transformer block in standard mode (no mHC overhead)
        mtp_config = copy.copy(config)
        mtp_config.use_mhc = False
        self.trm_block = TransformerBlock(mtp_config)

        # Small init so MTP contribution starts near zero
        nn.init.normal_(self.projection.weight, std=0.01)

    def forward(
        self,
        prev_hidden: Tensor,
        token_embeds: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Compute hidden states for this MTP depth.

        Args:
            prev_hidden: (B, T, C) hidden from previous depth (always single-stream).
            token_embeds: (B, T, C) embeddings of the tokens being fed at this depth.
            attention_mask: optional attention mask.
        Returns:
            (B, T, C) hidden states for this depth.
        """
        # Concatenate previous hidden + token embedding, project down
        combined = self.proj_norm(self.projection(torch.cat([prev_hidden, token_embeds], dim=-1)))

        # Process through one standard Transformer block
        return self.trm_block(combined, attention_mask)
