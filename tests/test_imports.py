"""Import smoke tests: every kazllm module must import without weights or downloads."""

import importlib

import pytest

MODULES = [
    "kazllm",
    "kazllm.eval",
    "kazllm.eval.benchmarks",
    "kazllm.eval.harness",
    "kazllm.eval.metrics",
    "kazllm.eval.results",
    "kazllm.model",
    "kazllm.model.engram",
    "kazllm.model.norm",
    "kazllm.model.qwen_engram_wrapper",
    "kazllm.tokenizer",
    "kazllm.tokenizer.fertility",
    "kazllm.training",
    "kazllm.training.callbacks",
    "kazllm.utils",
    "kazllm.utils.io",
    "kazllm.utils.logging",
    "kazllm.utils.seed",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name: str) -> None:
    importlib.import_module(module_name)


def test_wrapper_symbols_available_without_weights() -> None:
    """qwen_engram_wrapper must expose its API without loading any pretrained model."""
    module = importlib.import_module("kazllm.model.qwen_engram_wrapper")
    assert hasattr(module, "QymyzForCausalLM")
    assert hasattr(module, "EngramConfig")
    assert hasattr(module, "EngramWrappedLayer")
