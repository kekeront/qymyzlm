"""
Engram: Conditional N-gram Memory Module.

Retrieves static N-gram embeddings via deterministic hashing and fuses them
with the backbone's hidden states via context-aware gating. Designed to offload
local, stereotyped pattern reconstruction from the Transformer backbone.

For Kazakh specifically: agglutinative suffix chains (plural, case, possessive,
personal suffixes) form highly stereotyped {2,3}-gram patterns at the token level.
Engram retrieves morphological knowledge for these patterns via O(1) lookup,
freeing the backbone's early attention layers for actual semantic reasoning.

Architecture:
    1. Tokenizer compression: raw token IDs → canonical IDs (NFKC + lowercase equiv.)
    2. N-gram hashing: suffix N-grams → embedding table indices (multi-head XOR hash)
    3. Sparse lookup: retrieve static embedding vectors from hash tables
    4. Context-aware gating: hidden state h gates retrieved memory e via dot-product
    5. Depthwise causal conv: expand receptive field, add SiLU non-linearity
    6. Residual injection: H^(l) ← H^(l) + Engram_output

Reference: "Conditional Memory via Scalable Lookup: A New Axis of Sparsity for LLMs"
           (arXiv 2601.07372v1, DeepSeek-AI)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from kazllm.model.norm import RMSNorm


def _ngram_hash_index(
    canonical_ids: Tensor,
    n: int,
    head: int,
    table_size: int,
) -> Tensor:
    """Compute hash index for suffix N-gram ending at each position.

    Uses multiplicative-XOR hash: each token in the N-gram is multiplied by a
    different prime-derived constant and XOR-ed into a running hash value.

    Args:
        canonical_ids: (B, T) long tensor of compressed token IDs
        n: N-gram order (2 or 3)
        head: hash head index (determines hash seed, reducing collisions)
        table_size: number of slots in the embedding table (prime preferred)
    Returns:
        (B, T) long tensor of indices into the embedding table.
        Positions t < n-1 use only available history (padded with 0).
    """
    B, T = canonical_ids.shape
    # Seed per (n, head) pair — deterministic, collision-reducing
    seeds = [(2654435761 * (n * 31 + head * 7 + i + 1)) & 0xFFFF_FFFF for i in range(n)]
    h = torch.zeros(B, T, dtype=torch.long, device=canonical_ids.device)
    for i, seed in enumerate(seeds):
        # Shift: token at position t - i (0 for out-of-bounds positions)
        tok = canonical_ids.roll(shifts=i, dims=1)
        tok[:, :i] = 0  # positions before sequence start → pad token
        h = h ^ (tok * seed)
    return h % table_size


class EngramModule(nn.Module):
    """N-gram conditional memory module.

    Parameters
    ----------
    hidden_size : int
        Backbone hidden dimension (C). Retrieved memory is projected to this size.
    ngram_orders : list[int]
        N-gram orders to use, e.g. [2, 3].
    num_heads : int
        Number of hash heads per N-gram order (K in the paper). More heads reduce
        collision probability at the cost of more parameters.
    table_size : int
        Number of slots per hash table (M in the paper). Should be prime.
    slot_dim : int
        Embedding dimension per slot (d_mem_slot). Each head-order pair has a
        separate table of shape (table_size, slot_dim).
        Total retrieved vector dim = len(ngram_orders) * num_heads * slot_dim.
    vocab_size : int
        Vocabulary size. Used to build the canonical-ID lookup buffer.
    conv_kernel_size : int
        Kernel size for the depthwise causal convolution (4 in the paper).
    rms_norm_eps : float
        Epsilon for RMSNorm layers.
    """

    def __init__(
        self,
        hidden_size: int,
        ngram_orders: list[int] = None,
        num_heads: int = 4,
        table_size: int = 500_003,  # prime near 500K
        slot_dim: int = 64,
        vocab_size: int = 50_000,
        conv_kernel_size: int = 4,
        rms_norm_eps: float = 1e-5,
    ):
        super().__init__()
        if ngram_orders is None:
            ngram_orders = [2, 3]

        self.hidden_size = hidden_size
        self.ngram_orders = ngram_orders
        self.num_heads = num_heads
        self.table_size = table_size
        self.slot_dim = slot_dim
        self.vocab_size = vocab_size
        self.conv_kernel_size = conv_kernel_size

        # Total dimension of the concatenated retrieved vector
        num_tables = len(ngram_orders) * num_heads
        self.d_mem = num_tables * slot_dim  # total retrieved vector size

        # Hash tables: one nn.Embedding per (N-gram order, hash head)
        # Stored as a flat ModuleList; indexed by (n_idx * num_heads + k)
        self.tables = nn.ModuleList(
            [nn.Embedding(table_size, slot_dim, sparse=True) for _ in range(num_tables)]
        )

        # Canonical-ID lookup buffer: raw_token_id → canonical_token_id
        # Identity by default; updated by tokenizer training if compression is applied.
        self.register_buffer(
            "canonical_map",
            torch.arange(vocab_size, dtype=torch.long),
            persistent=True,
        )

        # Context-aware gating: α = σ(RMSNorm(q) · RMSNorm(k) / √d)
        # Q maps hidden → gate_dim; K maps retrieved memory → gate_dim
        gate_dim = max(slot_dim, hidden_size // 8)  # gate query/key dim
        self.q_proj = nn.Linear(hidden_size, gate_dim, bias=False)
        self.k_proj = nn.Linear(self.d_mem, gate_dim, bias=False)
        self.gate_norm_h = RMSNorm(gate_dim, rms_norm_eps)
        self.gate_norm_k = RMSNorm(gate_dim, rms_norm_eps)
        self.gate_scale = math.sqrt(gate_dim)

        # Value projection: retrieved memory → hidden_size
        self.v_proj = nn.Linear(self.d_mem, hidden_size, bias=False)

        # Depthwise causal convolution (expand receptive field, add non-linearity)
        # Dilation = max n-gram order extends receptive field to cover full suffix chains.
        # With kernel=4, dilation=3: effective field = 4 + 3*(4-1) = 13 tokens.
        self.max_ngram = max(ngram_orders)
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=conv_kernel_size,
            padding=(conv_kernel_size - 1) * self.max_ngram,  # causal: trim right later
            dilation=self.max_ngram,
            groups=hidden_size,  # depthwise
            bias=False,
        )
        self.conv_norm = RMSNorm(hidden_size, rms_norm_eps)

        # Output norm before injection
        self.out_norm = RMSNorm(hidden_size, rms_norm_eps)

        # Init: conv weights → 0 so output starts as identity (Ṽ + SiLU(Conv(RMSNorm(Ṽ))) ≈ Ṽ)
        nn.init.zeros_(self.conv.weight)
        # V_proj: small init so early training is not disrupted
        nn.init.normal_(self.v_proj.weight, std=0.01)
        nn.init.normal_(self.k_proj.weight, std=0.01)
        nn.init.normal_(self.q_proj.weight, std=0.01)

    def set_canonical_map(self, mapping: Tensor) -> None:
        """Update the token-to-canonical-ID mapping buffer.

        Args:
            mapping: (vocab_size,) long tensor; mapping[raw_id] = canonical_id.
        """
        assert mapping.shape == (self.vocab_size,)
        self.canonical_map.copy_(mapping)

    def _compress(self, token_ids: Tensor) -> Tensor:
        """Map raw token IDs to canonical IDs."""
        return self.canonical_map[token_ids]

    def _retrieve(self, canonical_ids: Tensor) -> Tensor:
        """Retrieve and concatenate embeddings for all N-gram heads.

        Args:
            canonical_ids: (B, T) canonical token IDs
        Returns:
            (B, T, d_mem) concatenated retrieved embeddings
        """
        B, T = canonical_ids.shape
        parts: list[Tensor] = []
        table_idx = 0
        for n in self.ngram_orders:
            for k in range(self.num_heads):
                idx = _ngram_hash_index(canonical_ids, n, k, self.table_size)  # (B, T)
                # Lookup: (B, T, slot_dim)
                emb = self.tables[table_idx](idx)
                parts.append(emb)
                table_idx += 1
        return torch.cat(parts, dim=-1)  # (B, T, d_mem)

    def _gate(self, h: Tensor, e: Tensor) -> Tensor:
        """Compute gated memory vector.

        Args:
            h: (B, T, hidden_size) current hidden states (query)
            e: (B, T, d_mem) retrieved memory (key + value source)
        Returns:
            (B, T, hidden_size) gated value
        """
        q = self.q_proj(h)  # (B, T, gate_dim)
        k = self.k_proj(e)  # (B, T, gate_dim)
        # Magnitude-sign decomposition (from Engram demo code): compress extreme
        # values via sqrt while preserving sign. Improves collision suppression —
        # negative similarity (context/n-gram mismatch) pushes gate below 0.5.
        raw_gate = (self.gate_norm_h(q) * self.gate_norm_k(k)).sum(dim=-1, keepdim=True) / self.gate_scale
        raw_gate = raw_gate.abs().clamp_min(1e-6).sqrt() * raw_gate.sign()
        alpha = torch.sigmoid(raw_gate)  # (B, T, 1) scalar gate ∈ (0,1)
        v = self.v_proj(e)  # (B, T, hidden_size)
        return alpha * v  # (B, T, hidden_size) gated value

    def _causal_conv(self, x: Tensor) -> Tensor:
        """Depthwise causal convolution with SiLU activation.

        Y = SiLU(Conv1D(RMSNorm(x))) + x

        Args:
            x: (B, T, hidden_size)
        Returns:
            (B, T, hidden_size)
        """
        # Conv1d expects (B, C, T)
        x_norm = self.conv_norm(x).transpose(1, 2)  # (B, hidden, T)
        # Causal: left-pad by (kernel_size-1)*dilation, then trim right
        conv_out = self.conv(x_norm)  # (B, hidden, T + pad)
        conv_out = conv_out[:, :, : x_norm.shape[2]]  # (B, hidden, T)  — trim
        conv_out = F.silu(conv_out).transpose(1, 2)  # (B, T, hidden)
        return conv_out + x

    def forward(self, hidden: Tensor, input_ids: Tensor) -> Tensor:
        """Compute Engram memory contribution.

        Args:
            hidden: (B, T, hidden_size) current backbone hidden states.
                    Used as query for context-aware gating.
            input_ids: (B, T) token IDs (before embedding lookup).
        Returns:
            (B, T, hidden_size) memory contribution to be added to hidden states.
            The caller does: hidden = hidden + engram(hidden, input_ids)
        """
        canonical_ids = self._compress(input_ids)  # (B, T)
        e = self._retrieve(canonical_ids)  # (B, T, d_mem)
        v_gated = self._gate(hidden, e)  # (B, T, hidden_size)
        y = self._causal_conv(v_gated)  # (B, T, hidden_size)

        return self.out_norm(y)
