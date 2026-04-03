"""Train a SentencePiece Unigram tokenizer on Kazakh text corpus."""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from kazllm.tokenizer.fertility import benchmark_tokenizer
from kazllm.tokenizer.hf_wrapper import sp_to_hf_tokenizer
from kazllm.tokenizer.trainer import train_tokenizer
from kazllm.utils.logging import setup_logging

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()

    deduped_dir = Path(cfg.data.deduped_dir)
    output_dir = Path(cfg.tokenizer.output_dir)

    # Collect text files from all deduped sources
    # For large datasets, we sample cfg.tokenizer.sampling_sentences sentences
    text_file = output_dir / "tokenizer_train_sample.txt"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not text_file.exists():
        log.info(
            f"Collecting {cfg.tokenizer.sampling_sentences:,} sentences for tokenizer training"
        )
        import random

        from datasets import load_from_disk

        sentences = []
        for source_dir in deduped_dir.iterdir():
            if not source_dir.is_dir():
                continue
            ds = load_from_disk(str(source_dir))
            for ex in ds:
                sentences.extend(ex["text"].split("\n"))
                if len(sentences) >= cfg.tokenizer.sampling_sentences * 2:
                    break

        random.shuffle(sentences)
        sentences = sentences[: cfg.tokenizer.sampling_sentences]
        with open(text_file, "w") as f:
            f.write("\n".join(sentences))
        log.info(f"Saved {len(sentences):,} sentences to {text_file}")

    # Train tokenizer
    model_path = train_tokenizer(
        input_files=[str(text_file)],
        output_dir=output_dir,
        vocab_size=cfg.tokenizer.vocab_size,
        model_type=cfg.tokenizer.model_type,
        character_coverage=cfg.tokenizer.character_coverage,
        byte_fallback=cfg.tokenizer.byte_fallback,
        add_dummy_prefix=cfg.tokenizer.add_dummy_prefix,
        num_threads=cfg.tokenizer.num_threads,
        input_sentence_size=cfg.tokenizer.sampling_sentences,
    )

    # Benchmark fertility
    test_lines = text_file.read_text().split("\n")[:5000]
    fertility_results = benchmark_tokenizer(
        sp_model_path=model_path,
        test_texts=test_lines,
        output_path=output_dir / "fertility_report.json",
    )
    log.info(f"Fertility benchmark: {fertility_results}")

    # Wrap as HF tokenizer
    sp_to_hf_tokenizer(model_path, output_dir / "hf_tokenizer")
    log.info("Done. HF tokenizer saved.")


if __name__ == "__main__":
    main()
