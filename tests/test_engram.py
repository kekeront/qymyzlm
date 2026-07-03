"""EngramModule invariants per its docstrings and CLAUDE.md.

Documented invariants under test:
- conv.weight initialised to 0 => the depthwise causal conv stage is identity at step 0
- hash tables are nn.Embedding(sparse=True) — critical for performance with large tables
- q/k/v projections are small-init (std 0.01) so early training is not disrupted
- canonical_map defaults to the identity mapping
"""

import torch

from kazllm.model.engram import EngramModule, _ngram_hash_index

HIDDEN = 32
VOCAB = 64
TABLE = 101  # small prime
SLOT = 8
HEADS = 2
ORDERS = [2, 3]


def tiny_engram() -> EngramModule:
    torch.manual_seed(0)
    return EngramModule(
        hidden_size=HIDDEN,
        ngram_orders=ORDERS,
        num_heads=HEADS,
        table_size=TABLE,
        slot_dim=SLOT,
        vocab_size=VOCAB,
        conv_kernel_size=4,
    )


def test_conv_weight_zero_init() -> None:
    engram = tiny_engram()
    assert torch.count_nonzero(engram.conv.weight).item() == 0


def test_causal_conv_is_identity_at_init() -> None:
    """Zero conv weights => Y = SiLU(0) + x = x exactly (docstring: 'starts as identity')."""
    engram = tiny_engram()
    x = torch.randn(2, 16, HIDDEN)
    with torch.no_grad():
        y = engram._causal_conv(x)
    torch.testing.assert_close(y, x)


def test_embedding_tables_are_sparse() -> None:
    tables = tiny_engram().tables
    assert len(tables) == len(ORDERS) * HEADS
    for table in tables:
        assert isinstance(table, torch.nn.Embedding)
        assert table.sparse is True


def test_gating_projections_small_init() -> None:
    """q/k/v projections use std=0.01 init: weights must be tiny, not default-scale."""
    engram = tiny_engram()
    for proj in (engram.q_proj, engram.k_proj, engram.v_proj):
        weight = proj.weight.detach()
        assert weight.abs().max().item() < 0.1
        assert weight.std().item() < 0.05


def test_canonical_map_defaults_to_identity() -> None:
    engram = tiny_engram()
    ids = torch.randint(0, VOCAB, (2, 16))
    assert torch.equal(engram._compress(ids), ids)


def test_set_canonical_map() -> None:
    engram = tiny_engram()
    mapping = torch.zeros(VOCAB, dtype=torch.long)
    engram.set_canonical_map(mapping)
    ids = torch.randint(0, VOCAB, (2, 16))
    assert torch.equal(engram._compress(ids), torch.zeros(2, 16, dtype=torch.long))


def test_d_mem_dimension() -> None:
    engram = tiny_engram()
    assert engram.d_mem == len(ORDERS) * HEADS * SLOT


def test_hash_index_range_and_determinism() -> None:
    ids = torch.randint(0, VOCAB, (2, 32))
    for n in ORDERS:
        for head in range(HEADS):
            idx1 = _ngram_hash_index(ids, n, head, TABLE)
            idx2 = _ngram_hash_index(ids, n, head, TABLE)
            assert torch.equal(idx1, idx2)
            assert idx1.min().item() >= 0
            assert idx1.max().item() < TABLE
            assert idx1.shape == ids.shape


def test_forward_shape_and_finiteness() -> None:
    engram = tiny_engram()
    batch, seq = 2, 16
    hidden = torch.randn(batch, seq, HIDDEN)
    input_ids = torch.randint(0, VOCAB, (batch, seq))
    with torch.no_grad():
        out = engram(hidden, input_ids)
    assert out.shape == (batch, seq, HIDDEN)
    assert torch.isfinite(out).all()


def test_forward_gradients_flow_to_tables() -> None:
    """A backward pass must reach the sparse hash tables (they are the memory being trained)."""
    engram = tiny_engram()
    hidden = torch.randn(2, 8, HIDDEN, requires_grad=True)
    input_ids = torch.randint(0, VOCAB, (2, 8))
    out = engram(hidden, input_ids)
    out.sum().backward()
    assert any(table.weight.grad is not None for table in engram.tables)
