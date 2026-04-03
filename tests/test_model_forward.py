"""Smoke test: model forward pass and loss computation.

Tests both standard (no mHC/Engram) and full KazLLM-v2 (mHC + Engram) modes.
"""

import torch

from kazllm.model.config import KazLLMConfig
from kazllm.model.model import KazLLMForCausalLM


def test_forward_no_labels(tiny_config, device):
    model = KazLLMForCausalLM(tiny_config).to(device)
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 16)).to(device)
    out = model(input_ids)
    assert out.logits.shape == (2, 16, tiny_config.vocab_size)
    assert out.loss is None


def test_forward_with_labels(tiny_config, device):
    model = KazLLMForCausalLM(tiny_config).to(device)
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 16)).to(device)
    labels = input_ids.clone()
    out = model(input_ids, labels=labels)
    assert out.loss is not None
    assert out.loss.item() > 0
    assert out.logits.shape == (2, 16, tiny_config.vocab_size)


def test_param_count(tiny_config):
    model = KazLLMForCausalLM(tiny_config)
    total = sum(p.numel() for p in model.parameters())
    assert total > 0
    print(f"Tiny model params: {total:,}")


def test_forward_mhc_only(device):
    """mHC without Engram: n-stream expands and collapses correctly."""
    cfg = KazLLMConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        rope_theta=10000.0,
        use_flash_attention=False,
        use_mhc=True,
        mhc_streams=4,
        use_engram=False,
    )
    model = KazLLMForCausalLM(cfg).to(device)
    input_ids = torch.randint(0, 256, (2, 16)).to(device)
    labels = input_ids.clone()
    out = model(input_ids, labels=labels)
    assert out.logits.shape == (2, 16, 256)
    assert out.loss is not None and out.loss.item() > 0


def test_forward_full_kazllmv2(device):
    """Full KazLLM-v2: mHC + Engram end-to-end forward + backward."""
    cfg = KazLLMConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        rope_theta=10000.0,
        use_flash_attention=False,
        use_mhc=True,
        mhc_streams=4,
        use_engram=True,
        engram_layer_indices=[1, 2],
        engram_ngram_orders=[2, 3],
        engram_num_heads=2,
        engram_table_size=97,
        engram_slot_dim=8,
    )
    model = KazLLMForCausalLM(cfg).to(device)
    input_ids = torch.randint(0, 256, (2, 16)).to(device)
    labels = input_ids.clone()
    out = model(input_ids, labels=labels)
    assert out.logits.shape == (2, 16, 256)
    assert out.loss is not None

    out.loss.backward()
    # Engram embedding tables should receive sparse gradients
    engram = model.model.engram_layers["1"]
    assert any(t.weight.grad is not None for t in engram.tables), (
        "Engram tables should receive gradients"
    )
