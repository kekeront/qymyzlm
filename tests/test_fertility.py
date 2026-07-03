"""Fertility benchmark tests on a tiny sentencepiece model trained on inline Kazakh text.

No downloads: the tokenizer is trained in-process on the sample below (fractions of a second).
"""

import json
from pathlib import Path

import pytest
import sentencepiece as spm

from kazllm.tokenizer.fertility import benchmark_tokenizer, compute_fertility

KAZAKH_SAMPLE = [
    "Қазақстан Республикасы — Орталық Азиядағы мемлекет.",
    "Мемлекеттік тілі — қазақ тілі, ол түркі тілдер тобына жатады.",
    "Қазақ тілі агглютинативті тіл болып табылады.",
    "Сөздерге жұрнақтар мен жалғаулар қосылып, жаңа мағыналар жасалады.",
    "Астанасы — Астана қаласы, халық саны жиырма миллионнан асады.",
    "Балаларымыздың кітаптарындағы суреттер өте әдемі екен.",
]


@pytest.fixture(scope="module")
def tiny_sp_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train a tiny BPE sentencepiece model on the inline Kazakh sample."""
    prefix = tmp_path_factory.mktemp("spm") / "tiny_kaz"
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(KAZAKH_SAMPLE),
        model_prefix=str(prefix),
        vocab_size=120,
        model_type="bpe",
        character_coverage=1.0,
    )
    return prefix.with_suffix(".model")


def test_compute_fertility_positive_and_plausible(tiny_sp_model: Path) -> None:
    fertility = compute_fertility(tiny_sp_model, KAZAKH_SAMPLE)
    # A 120-piece BPE model must split Kazakh words into >1 token on average,
    # and cannot produce more tokens than characters.
    assert 1.0 < fertility < 20.0


def test_compute_fertility_empty_texts(tiny_sp_model: Path) -> None:
    assert compute_fertility(tiny_sp_model, []) == 0.0
    assert compute_fertility(tiny_sp_model, ["", "   "]) == 0.0


def test_benchmark_tokenizer_results_and_output_file(tiny_sp_model: Path, tmp_path: Path) -> None:
    out_path = tmp_path / "sub" / "fertility.json"
    results = benchmark_tokenizer(tiny_sp_model, KAZAKH_SAMPLE, output_path=out_path)

    assert results["num_test_texts"] == len(KAZAKH_SAMPLE)
    assert results["kazakh_fertility"] > 0
    expected_ratio = results["kazakh_fertility"] / 4.73  # Llama-3.1 kaz baseline
    assert results["vs_llama31_baseline"] == pytest.approx(expected_ratio)
    assert isinstance(results["target_met"], bool)
    assert isinstance(results["excellent"], bool)

    assert out_path.exists()
    on_disk = json.loads(out_path.read_text())
    assert on_disk["kazakh_fertility"] == pytest.approx(results["kazakh_fertility"])
