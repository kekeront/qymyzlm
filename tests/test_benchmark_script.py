"""Regression tests for scripts/benchmark_baselines.py prompt handling and reporting.

The script is loaded as a module from its file path (no downloads, no model runs).
Covers the confirmed defects: right-truncation losing the question/answer cue, and
reported reference rows rendered as if measured.
"""

import importlib.util
from pathlib import Path

import torch

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bb = _load_script("benchmark_baselines")


class FakeTokenizer:
    """One token per whitespace-delimited chunk; ids are sequential positions."""

    def __call__(self, text: str, return_tensors: str = "pt") -> dict:
        n = len(text.split())
        ids = torch.arange(n).unsqueeze(0)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def _row(question: str, answer: str = "A") -> dict:
    return {
        "Question": question,
        "Option A": "бір",
        "Option B": "екі",
        "Option C": "үш",
        "Option D": "төрт",
        "Option E": "",
        "Answer Key": answer,
    }


DEV = [_row(f"Дев сұрақ {i} қосымша сөздер осында") for i in range(3)]
TEST_ROW = _row("Тест сұрағы қандай жауап дұрыс")


def test_get_valid_options_skips_empty_option_e() -> None:
    assert bb.get_valid_options(TEST_ROW) == ["A", "B", "C", "D"]


def test_prompt_ends_with_answer_cue() -> None:
    prompt = bb.build_fewshot_prompt(DEV, TEST_ROW, n_shot=3)
    assert prompt.endswith("Жауап:")
    assert prompt.count("Жауап:") == 4  # 3 shots + test question


def test_no_truncation_when_prompt_fits() -> None:
    inputs, shots_used, tail_truncated = bb.encode_prompt_left_truncated(
        FakeTokenizer(), DEV, TEST_ROW, n_shot=3, max_length=10_000
    )
    assert shots_used == 3
    assert tail_truncated is False


def test_left_truncation_drops_earliest_shots_not_the_question() -> None:
    tokenizer = FakeTokenizer()
    full_len = tokenizer(bb.build_fewshot_prompt(DEV, TEST_ROW, n_shot=3))["input_ids"].shape[1]
    zero_len = tokenizer(bb.build_fewshot_prompt(DEV, TEST_ROW, n_shot=0))["input_ids"].shape[1]
    budget = (full_len + zero_len) // 2  # fits some shots, not all

    inputs, shots_used, tail_truncated = bb.encode_prompt_left_truncated(
        tokenizer, DEV, TEST_ROW, n_shot=3, max_length=budget
    )
    assert 0 <= shots_used < 3
    assert tail_truncated is False
    assert inputs["input_ids"].shape[1] <= budget
    # The kept prompt is exactly the k-shot prompt => question + cue intact.
    kept = bb.build_fewshot_prompt(DEV, TEST_ROW, n_shot=shots_used)
    assert kept.endswith("Жауап:")
    assert inputs["input_ids"].shape[1] == len(kept.split())


def test_tail_kept_when_even_zero_shot_is_too_long() -> None:
    tokenizer = FakeTokenizer()
    zero_len = tokenizer(bb.build_fewshot_prompt(DEV, TEST_ROW, n_shot=0))["input_ids"].shape[1]
    budget = zero_len - 2

    inputs, shots_used, tail_truncated = bb.encode_prompt_left_truncated(
        tokenizer, DEV, TEST_ROW, n_shot=3, max_length=budget
    )
    assert shots_used == 0
    assert tail_truncated is True
    assert inputs["input_ids"].shape[1] == budget
    # FakeTokenizer ids are positions: the LAST token (answer cue end) must survive.
    assert inputs["input_ids"][0, -1].item() == zero_len - 1


def test_markdown_table_separates_measured_from_reported() -> None:
    results = [
        {
            "name": "Qwen3-0.6B",
            "params_b": 0.6,
            "kazmmlu_acc": 32.8,
            "fertility": 4.88,
            "speed_tok_s": 31.3,
        },
        {"name": "Broken", "model_id": "x", "error": "boom"},
    ]
    table = bb.markdown_table(results, n_shot=3)
    measured_part, reported_part = table.split("Reported in their papers")
    assert "KazMMLU 3-shot" in measured_part
    assert "Qwen3-0.6B" in measured_part
    assert "FAILED" in measured_part
    assert "Sherkala" not in measured_part  # reported rows never mixed into measured
    assert "Sherkala-Chat-8B" in reported_part
    assert "Llama-3.1-70B" in reported_part


def test_qlora_script_imports_and_detects_precision_capability() -> None:
    """qlora_continual must import without downloads; bf16 use is capability-gated."""
    qlora = _load_script("qlora_continual")
    source = (SCRIPTS_DIR / "qlora_continual.py").read_text()
    # Regression for the RTX 2070 (Turing, no bf16) crash: no hardcoded bf16.
    assert "bnb_4bit_compute_dtype=torch.bfloat16" not in source
    assert "bf16=True" not in source
    assert "is_bf16_supported" in source
    assert hasattr(qlora, "main")
