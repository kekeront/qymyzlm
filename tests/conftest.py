"""Shared pytest fixtures."""

import pytest
import torch

from kazllm.model.config import KazLLMConfig


@pytest.fixture
def tiny_config() -> KazLLMConfig:
    """Minimal model config for fast unit tests."""
    return KazLLMConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rope_theta=10000.0,
        use_flash_attention=False,
    )


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
