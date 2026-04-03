"""Convert SentencePiece model to HuggingFace PreTrainedTokenizerFast."""

import json
import logging
import shutil
from pathlib import Path

import sentencepiece as spm
from tokenizers import AddedToken, Tokenizer
from tokenizers.models import Unigram
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

log = logging.getLogger(__name__)

SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<pad>"]
ADDITIONAL_SPECIAL_TOKENS = ["<|system|>", "<|user|>", "<|assistant|>"]


def sp_to_hf_tokenizer(
    sp_model_path: str | Path,
    output_dir: str | Path,
) -> PreTrainedTokenizerFast:
    """Wrap a trained SentencePiece Unigram model as a HuggingFace PreTrainedTokenizerFast.

    Args:
        sp_model_path: Path to the .model file from SentencePiece training.
        output_dir: Directory to save the HF tokenizer files.

    Returns:
        A PreTrainedTokenizerFast ready for use with transformers.
    """
    sp_model_path = Path(sp_model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load SP model to extract vocab and scores
    sp = spm.SentencePieceProcessor()
    sp.Load(str(sp_model_path))
    vocab_size = sp.GetPieceSize()

    # Build vocab: list of (piece, score) for the tokenizers Unigram model
    pieces = [(sp.IdToPiece(i), sp.GetScore(i)) for i in range(vocab_size)]
    vocab_map = {piece: i for i, (piece, _) in enumerate(pieces)}

    # Build tokenizers library Unigram tokenizer; unk_id=0 per SP convention
    unk_id = vocab_map.get("<unk>", 0)
    tokenizer_obj = Tokenizer(Unigram(pieces, unk_id=unk_id))
    # SP uses ▁ (U+2581) as a space prefix — Metaspace pre-tokenizer handles this
    tokenizer_obj.pre_tokenizer = Metaspace(replacement="▁", prepend_scheme="always")

    # Add special tokens
    all_special = SPECIAL_TOKENS + ADDITIONAL_SPECIAL_TOKENS
    tokenizer_obj.add_special_tokens([AddedToken(t, special=True) for t in all_special])

    # No BOS prepending by default (pack_data adds EOS, train uses labels)
    bos_id = vocab_map.get("<s>", 1)
    eos_id = vocab_map.get("</s>", 2)
    tokenizer_obj.post_processor = TemplateProcessing(
        single="$A",
        pair="$A $B",
        special_tokens=[("<s>", bos_id), ("</s>", eos_id)],
    )

    # Save tokenizer.json
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer_obj.save(str(tokenizer_path))

    # Copy SP model file for reference
    shutil.copy2(sp_model_path, output_dir / "tokenizer.model")

    # Write tokenizer_config.json
    config = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS,
        "model_max_length": 4096,
        "tokenizer_class": "PreTrainedTokenizerFast",
    }
    (output_dir / "tokenizer_config.json").write_text(json.dumps(config, indent=2))

    # Load and return
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path),
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        additional_special_tokens=ADDITIONAL_SPECIAL_TOKENS,
        model_max_length=4096,
    )
    log.info(f"HF tokenizer saved to {output_dir} (vocab_size={hf_tok.vocab_size})")
    return hf_tok
