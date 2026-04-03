"""Tests for Multi-Token Prediction (MTP) module.

Tests MTP in isolation and integrated into KazLLMForCausalLM,
both with and without mHC/Engram.
"""

import torch

from kazllm.model.config import KazLLMConfig
from kazllm.model.model import KazLLMForCausalLM
from kazllm.model.mtp import MTPModule


def _tiny_mtp_config(**overrides) -> KazLLMConfig:
    defaults = dict(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        rope_theta=10000.0,
        use_flash_attention=False,
        use_mhc=False,
        use_engram=False,
        use_mtp=True,
        mtp_depth=1,
        mtp_lambda=0.3,
    )
    defaults.update(overrides)
    return KazLLMConfig(**defaults)


def test_mtp_module_forward(device):
    """MTP module produces correct output shape."""
    cfg = _tiny_mtp_config()
    mod = MTPModule(cfg).to(device)
    B, T, C = 2, 16, cfg.hidden_size
    prev_hidden = torch.randn(B, T, C, device=device)
    token_embeds = torch.randn(B, T, C, device=device)
    out = mod(prev_hidden, token_embeds)
    assert out.shape == (B, T, C)


def test_mtp_module_always_single_stream(device):
    """MTP module always operates in single-stream even when main model uses mHC."""
    cfg = _tiny_mtp_config(use_mhc=True, mhc_streams=4)
    mod = MTPModule(cfg).to(device)
    # MTP receives collapsed (B, T, C) hidden states
    B, T, C = 2, 16, cfg.hidden_size
    prev_hidden = torch.randn(B, T, C, device=device)
    token_embeds = torch.randn(B, T, C, device=device)
    out = mod(prev_hidden, token_embeds)
    assert out.shape == (B, T, C)
    # Verify MTP's TRM block doesn't use mHC
    assert not mod.trm_block.use_mhc


def test_mtp_loss_computed(device):
    """MTP loss is added to main loss when use_mtp=True."""
    cfg = _tiny_mtp_config()
    model = KazLLMForCausalLM(cfg).to(device)
    input_ids = torch.randint(0, 256, (2, 16), device=device)
    labels = input_ids.clone()

    out_mtp = model(input_ids, labels=labels)
    assert out_mtp.loss is not None
    assert out_mtp.loss.item() > 0

    # Compare to no-MTP: loss should differ due to MTP contribution
    cfg_no_mtp = _tiny_mtp_config(use_mtp=False)
    model_no_mtp = KazLLMForCausalLM(cfg_no_mtp).to(device)
    # Copy weights (excluding MTP modules)
    state = {k: v for k, v in model.state_dict().items() if "mtp_modules" not in k}
    model_no_mtp.load_state_dict(state, strict=True)
    out_no_mtp = model_no_mtp(input_ids, labels=labels)
    # Losses should be different (MTP adds an extra term)
    assert out_mtp.loss.item() != out_no_mtp.loss.item()


def test_mtp_gradient_flows(device):
    """Gradients flow through MTP modules back to the main model."""
    cfg = _tiny_mtp_config()
    model = KazLLMForCausalLM(cfg).to(device)
    input_ids = torch.randint(0, 256, (2, 16), device=device)
    labels = input_ids.clone()

    out = model(input_ids, labels=labels)
    out.loss.backward()

    # MTP projection should have gradients
    mtp_mod = model.mtp_modules[0]
    assert mtp_mod.projection.weight.grad is not None
    assert mtp_mod.projection.weight.grad.abs().sum() > 0

    # Main model embed_tokens should also have gradients (shared with MTP)
    assert model.model.embed_tokens.weight.grad is not None


def test_mtp_lambda_override(device):
    """mtp_lambda can be overridden at forward time for scheduling."""
    cfg = _tiny_mtp_config(mtp_lambda=0.3)
    model = KazLLMForCausalLM(cfg).to(device)
    input_ids = torch.randint(0, 256, (2, 16), device=device)
    labels = input_ids.clone()

    loss_03 = model(input_ids, labels=labels, mtp_lambda=0.3).loss.item()
    loss_01 = model(input_ids, labels=labels, mtp_lambda=0.1).loss.item()
    loss_00 = model(input_ids, labels=labels, mtp_lambda=0.0).loss.item()

    # Lower lambda → MTP contributes less → loss closer to main-only
    assert loss_03 != loss_01
    # lambda=0 should give same loss as no MTP (within float tolerance)
    cfg_no_mtp = _tiny_mtp_config(use_mtp=False)
    model_no_mtp = KazLLMForCausalLM(cfg_no_mtp).to(device)
    state = {k: v for k, v in model.state_dict().items() if "mtp_modules" not in k}
    model_no_mtp.load_state_dict(state, strict=True)
    loss_no_mtp = model_no_mtp(input_ids, labels=labels).loss.item()
    assert abs(loss_00 - loss_no_mtp) < 1e-5


def test_mtp_no_labels_no_loss(device):
    """Without labels, MTP should not contribute any loss."""
    cfg = _tiny_mtp_config()
    model = KazLLMForCausalLM(cfg).to(device)
    input_ids = torch.randint(0, 256, (2, 16), device=device)
    out = model(input_ids)
    assert out.loss is None
    assert out.logits.shape == (2, 16, 256)


def test_mtp_with_mhc_and_engram(device):
    """Full KazLLM-v3: mHC + Engram + MTP end-to-end."""
    cfg = _tiny_mtp_config(
        num_hidden_layers=4,
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
    input_ids = torch.randint(0, 256, (2, 16), device=device)
    labels = input_ids.clone()

    out = model(input_ids, labels=labels)
    assert out.logits.shape == (2, 16, 256)
    assert out.loss is not None

    out.loss.backward()
    # MTP, Engram, and main model should all have gradients
    assert model.mtp_modules[0].projection.weight.grad is not None
    engram = model.model.engram_layers["1"]
    assert any(t.weight.grad is not None for t in engram.tables)


def test_mtp_param_count():
    """MTP adds expected number of parameters."""
    cfg_no_mtp = _tiny_mtp_config(use_mtp=False)
    cfg_mtp = _tiny_mtp_config(use_mtp=True, mtp_depth=1)

    model_no = KazLLMForCausalLM(cfg_no_mtp)
    model_mtp = KazLLMForCausalLM(cfg_mtp)

    params_no = sum(p.numel() for p in model_no.parameters())
    params_mtp = sum(p.numel() for p in model_mtp.parameters())

    # MTP adds: projection (2C*C) + one TRM block (attn + MLP + norms)
    mtp_params = params_mtp - params_no
    assert mtp_params > 0
    # Projection alone: 2 * 64 * 64 = 8192
    assert mtp_params >= 2 * 64 * 64
    print(f"MTP added params: {mtp_params:,} ({mtp_params / params_no * 100:.1f}% of base)")


def test_mtp_disabled_matches_v2(device):
    """With use_mtp=False, model behaves identically to v2."""
    cfg = _tiny_mtp_config(use_mtp=False)
    model = KazLLMForCausalLM(cfg).to(device)
    assert len(model.mtp_modules) == 0

    input_ids = torch.randint(0, 256, (2, 16), device=device)
    labels = input_ids.clone()
    out = model(input_ids, labels=labels)
    assert out.loss is not None
