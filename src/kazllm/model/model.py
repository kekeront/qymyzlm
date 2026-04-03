"""KazLLM causal language model.

Supports three architectural modes controlled by KazLLMConfig:

  Standard (use_mhc=False, use_engram=False, use_mtp=False):
    Classic Llama-style decoder; input flows as (B, T, C) through all layers.

  KazLLM-v2 (use_mhc=True, use_engram=True, use_mtp=False):
    mHC expands residual to (B, T, n, C) n-stream.
    Engram injects N-gram memory at designated layers before the block computation.
    The stream is collapsed back to (B, T, C) after the final layer for the LM head.

  KazLLM-v3 (use_mhc=True, use_engram=True, use_mtp=True):
    Same as v2, plus Multi-Token Prediction (DeepSeek-V3 style).
    During training, predicts D additional future tokens via sequential MTP modules.
    MTP modules are discarded at inference — zero deployment cost.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from kazllm.model.block import TransformerBlock
from kazllm.model.config import KazLLMConfig
from kazllm.model.engram import EngramModule
from kazllm.model.mhc import MHCStreamManager, collapse_streams, expand_to_streams
from kazllm.model.mtp import MTPModule
from kazllm.model.norm import RMSNorm

# Default initializer std — matches Llama convention
_INIT_STD = 0.02
# Small std for gating projections that should start near-zero
_SMALL_STD = 0.01


class KazLLMModel(PreTrainedModel):
    config_class = KazLLMConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: KazLLMConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # Engram modules at designated layer indices
        self.engram_layers: dict[int, EngramModule] = nn.ModuleDict()
        if config.use_engram:
            for layer_idx in config.engram_layer_indices:
                if 0 <= layer_idx < config.num_hidden_layers:
                    self.engram_layers[str(layer_idx)] = EngramModule(
                        hidden_size=config.hidden_size,
                        ngram_orders=config.engram_ngram_orders,
                        num_heads=config.engram_num_heads,
                        table_size=config.engram_table_size,
                        slot_dim=config.engram_slot_dim,
                        vocab_size=config.vocab_size,
                        conv_kernel_size=config.engram_conv_kernel_size,
                        rms_norm_eps=config.rms_norm_eps,
                    )

        self.gradient_checkpointing = False
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        """Custom weight initialization that preserves critical invariants.

        HuggingFace's default _init_weights applies normal_(0, 0.02) to ALL
        Linear/Conv1d/Embedding modules. This overwrites Engram's zero-init
        conv (needed for identity at step 0) and the small-std gating projections.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=_INIT_STD)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            # Only Engram uses Conv1d — zero-init for identity at start
            # (SiLU(Conv=0) + x = 0 + x = x)
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=_INIT_STD)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, EngramModule):
            # Re-apply small init for gating projections (after Linear init ran)
            nn.init.normal_(module.v_proj.weight, std=_SMALL_STD)
            nn.init.normal_(module.k_proj.weight, std=_SMALL_STD)
            nn.init.normal_(module.q_proj.weight, std=_SMALL_STD)
        elif isinstance(module, MHCStreamManager):
            # Re-apply small init for dynamic projections
            nn.init.normal_(module.phi_pre.weight, std=_SMALL_STD)
            nn.init.normal_(module.phi_post.weight, std=_SMALL_STD)
            nn.init.normal_(module.phi_res.weight, std=_SMALL_STD)

    def _set_gradient_checkpointing(
        self, enable: bool = True, gradient_checkpointing_func=None
    ) -> None:
        # Use the new-format signature expected by transformers >= 4.35.
        # The old (module, value) signature triggers a deprecation warning and
        # causes gradient_checkpointing_enable() to bypass custom kwargs.
        self.gradient_checkpointing = enable

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        x = self.embed_tokens(input_ids)  # (B, T, C)

        if self.config.use_mhc:
            x = expand_to_streams(x, self.config.mhc_streams)  # (B, T, n, C)

        for layer_idx, layer in enumerate(self.layers):
            # Engram injection before this layer's computation.
            # Operates on the single-stream (C-dim) representation.
            key = str(layer_idx)
            engram = self.engram_layers[key] if key in self.engram_layers else None
            if engram is not None:
                if self.config.use_mhc:
                    # For n-stream: apply Engram on the mean of streams,
                    # then add contribution to all streams equally.
                    h_mean = x.mean(dim=2)  # (B, T, C)
                    mem = engram(h_mean, input_ids)  # (B, T, C)
                    x = x + mem.unsqueeze(2)  # broadcast to (B,T,n,C)
                else:
                    x = x + engram(x, input_ids)  # (B, T, C)

            if self.gradient_checkpointing and self.training:
                import torch.utils.checkpoint

                x = torch.utils.checkpoint.checkpoint(layer, x, attention_mask, use_reentrant=False)
            else:
                x = layer(x, attention_mask)

        if self.config.use_mhc:
            x = collapse_streams(x)  # (B, T, n, C) → (B, T, C)

        return self.norm(x)


class KazLLMForCausalLM(PreTrainedModel):
    config_class = KazLLMConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    supports_gradient_checkpointing = True

    def __init__(self, config: KazLLMConfig):
        super().__init__(config)
        self.model = KazLLMModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # MTP modules (training only — discarded at inference)
        self.mtp_modules = nn.ModuleList()
        if config.use_mtp:
            for _ in range(config.mtp_depth):
                self.mtp_modules.append(MTPModule(config))

        self.post_init()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        mtp_lambda: float | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden = self.model(input_ids, attention_mask)  # always (B, T, C)
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            # Main next-token prediction loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            main_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss = main_loss

            # MTP: predict additional future tokens (training only)
            if self.config.use_mtp and self.mtp_modules:
                lam = mtp_lambda if mtp_lambda is not None else self.config.mtp_lambda
                D = len(self.mtp_modules)
                mtp_loss_sum = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
                prev_h = hidden  # (B, T, C) — always single-stream after model

                for k, mtp_mod in enumerate(self.mtp_modules):
                    shift = k + 1
                    if shift >= labels.shape[1]:
                        break

                    # Feed: ground-truth token IDs at offset k (teacher forcing).
                    # labels may contain -100 (cross-document mask) at padding positions.
                    # embed_tokens(-100) is an out-of-bounds access; clamp to 0 (pad token)
                    # before lookup.  The corresponding positions are already excluded from
                    # the MTP loss via ignore_index=-100 on mtp_targets_aligned.
                    feed_ids = labels[:, shift:].clamp(min=0)  # (B, T-shift)
                    T_mtp = feed_ids.shape[1]
                    prev_h_trunc = prev_h[:, :T_mtp]

                    feed_embeds = self.model.embed_tokens(feed_ids)  # (B, T_mtp, C)
                    h_k = mtp_mod(prev_h_trunc, feed_embeds, attention_mask)
                    mtp_logits = self.lm_head(h_k)  # (B, T_mtp, V)

                    # Target: predict token at t+k+2
                    target_start = shift + 1
                    if target_start >= labels.shape[1]:
                        break
                    mtp_targets = labels[:, target_start : target_start + T_mtp]
                    # Align lengths
                    min_len = min(mtp_logits.shape[1], mtp_targets.shape[1])
                    mtp_logits_aligned = mtp_logits[:, :min_len].contiguous()
                    mtp_targets_aligned = mtp_targets[:, :min_len].contiguous()

                    mtp_loss_k = F.cross_entropy(
                        mtp_logits_aligned.view(-1, self.config.vocab_size),
                        mtp_targets_aligned.view(-1),
                        ignore_index=-100,
                    )
                    mtp_loss_sum = mtp_loss_sum + mtp_loss_k
                    prev_h = h_k  # sequential: next depth uses this depth's output

                loss = main_loss + (lam / D) * mtp_loss_sum

        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head
