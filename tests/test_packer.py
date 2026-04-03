"""Tests for sequence packing logic."""

import numpy as np

from kazllm.data.packer import pack_sequences


def test_basic_packing():
    tokens = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    packed = list(pack_sequences(iter(tokens), context_length=4, eos_token_id=0))
    assert all(len(seq) == 4 for seq in packed)
    assert all(isinstance(seq, np.ndarray) for seq in packed)


def test_packing_fills_context():
    # Single long document should be split across sequences
    long_doc = list(range(1, 101))
    packed = list(pack_sequences(iter([long_doc]), context_length=10, eos_token_id=0))
    # 100 tokens + 1 EOS = 101 tokens, with padding = ceil(101/10) = 11 sequences
    assert len(packed) >= 10
    for seq in packed:
        assert len(seq) == 10


def test_eos_inserted():
    tokens = [[1, 2, 3]]
    packed = list(pack_sequences(iter(tokens), context_length=8, eos_token_id=99))
    flat = [t for seq in packed for t in seq.tolist()]
    assert 99 in flat  # EOS token must appear
