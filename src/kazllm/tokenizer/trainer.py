"""Train a SentencePiece tokenizer on Kazakh text corpus."""

import logging
from pathlib import Path

import sentencepiece as spm

log = logging.getLogger(__name__)


def train_tokenizer(
    input_files: list[str],
    output_dir: str | Path,
    vocab_size: int = 50_000,
    model_type: str = "unigram",
    character_coverage: float = 0.9999,
    byte_fallback: bool = True,
    add_dummy_prefix: bool = False,
    max_sentence_length: int = 8192,
    num_threads: int = 16,
    input_sentence_size: int = 10_000_000,
) -> Path:
    """Train a SentencePiece tokenizer with Kazakh-optimized settings.

    Args:
        input_files: List of text file paths (one sentence per line).
        output_dir: Directory to save tokenizer artifacts.
        vocab_size: Vocabulary size (default: 50K).
        model_type: "unigram" (recommended) or "bpe".
        character_coverage: Coverage of Unicode chars (0.9999 covers full Kazakh Cyrillic+Latin).
        byte_fallback: If True, use byte-level fallback for OOV chars (no UNK tokens).
        add_dummy_prefix: Whether to add space prefix to first token (False = Llama convention).
        max_sentence_length: Max chars per sentence.
        num_threads: Parallel training threads.
        input_sentence_size: Max sentences to use for training (10M is sufficient).

    Returns:
        Path to the trained .model file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = str(output_dir / f"tokenizer_{model_type}_{vocab_size // 1000}k")

    input_str = ",".join(input_files)

    log.info(f"Training {model_type} tokenizer (vocab={vocab_size}) on {len(input_files)} files")

    spm.SentencePieceTrainer.train(
        input=input_str,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        byte_fallback=byte_fallback,
        add_dummy_prefix=add_dummy_prefix,
        max_sentence_length=max_sentence_length,
        num_threads=num_threads,
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=True,
        # Special tokens
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        # User-defined tokens for chat (reserved for SFT)
        user_defined_symbols=["<pad>", "<|system|>", "<|user|>", "<|assistant|>"],
        train_extremely_large_corpus=True,
    )

    model_path = Path(model_prefix + ".model")
    log.info(f"Tokenizer saved to {model_path}")
    return model_path
