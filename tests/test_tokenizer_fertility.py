"""Tests for tokenizer fertility benchmarking (no SPM model required)."""

from unittest.mock import MagicMock, patch

from kazllm.tokenizer.fertility import LLAMA31_KAZ_FERTILITY, compute_fertility


def test_compute_fertility_mock():
    """Verify fertility computation with a mocked SP model."""
    with patch("kazllm.tokenizer.fertility.spm.SentencePieceProcessor") as mock_sp_cls:
        mock_sp = MagicMock()
        mock_sp_cls.return_value = mock_sp
        # Mock encode to return 3 tokens per word → fertility ≈ 3.0
        mock_sp.encode = lambda text: [0] * (3 * len(text.split()))

        texts = ["слово одно два три", "мен сені жақсы көремін"]
        fertility = compute_fertility("fake_model.model", texts)
        assert abs(fertility - 3.0) < 0.01, f"Expected ~3.0, got {fertility}"


def test_llama31_baseline_constant():
    assert LLAMA31_KAZ_FERTILITY == 4.73
