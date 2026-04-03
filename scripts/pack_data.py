"""Tokenize and pack data into uint16 binary shards for training."""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerFast

from kazllm.data.packer import pack_sequences
from kazllm.data.shard_writer import ShardWriter
from kazllm.utils.logging import setup_logging

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()

    deduped_dir = Path(cfg.data.deduped_dir)
    tokenizer_dir = Path(cfg.tokenizer.output_dir) / "hf_tokenizer"
    output_dir = Path("data/tokenized")
    context_length = cfg.data.get("context_length", cfg.training.context_length)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir))
    log.info(f"Loaded tokenizer (vocab_size={tokenizer.vocab_size}) from {tokenizer_dir}")

    writer = ShardWriter(output_dir, tokens_per_shard=500_000_000)

    def token_iterator():
        from datasets import load_from_disk

        for source_dir in sorted(deduped_dir.iterdir()):
            if not source_dir.is_dir():
                continue
            log.info(f"Tokenizing {source_dir.name}")
            ds = load_from_disk(str(source_dir))
            for ex in ds:
                ids = tokenizer.encode(ex["text"], add_special_tokens=False)
                if ids:
                    yield ids

    for sequence in pack_sequences(
        token_iterator(),
        context_length=context_length,
        eos_token_id=tokenizer.eos_token_id or 3,
    ):
        writer.write(sequence)

    manifest = writer.finalize()
    log.info(f"Packed {manifest['total_tokens']:,} tokens into {manifest['num_shards']} shards")
    log.info(f"Manifest saved to {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
