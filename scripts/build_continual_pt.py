"""Build the QymyzLM continual pretraining checkpoint.

Takes Qwen-2.5-1.5B, expands its vocabulary with Kazakh-specific tokens from our
SentencePiece 50K tokenizer, grafts Engram modules at specified layers, and saves
the result as a ready-to-train checkpoint.

Usage:
    python scripts/build_continual_pt.py
    python scripts/build_continual_pt.py --dry-run        # load + 1 forward pass, no save
    python scripts/build_continual_pt.py --max-new-tokens 5000
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_base_model_and_tokenizer(base_model_name: str):
    """Load the base Qwen model and tokenizer from HuggingFace."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    log.info(
        f"Base model loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params, "
        f"vocab={model.config.vocab_size}, layers={model.config.num_hidden_layers}"
    )
    return model, tokenizer


def load_kazakh_tokenizer(tokenizer_path: str):
    """Load the trained Kazakh SentencePiece tokenizer."""
    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    log.info(f"Kazakh tokenizer loaded: vocab_size={tok.vocab_size}")
    return tok


def find_new_tokens(base_tokenizer, kazakh_tokenizer, max_new_tokens: int) -> list[str]:
    """Find Kazakh tokens not in the base model's vocabulary.

    Returns the top-frequency tokens sorted by their ID in the Kazakh tokenizer
    (lower ID = higher frequency for SentencePiece Unigram).
    """
    base_vocab = set(base_tokenizer.get_vocab().keys())
    kaz_vocab = kazakh_tokenizer.get_vocab()

    # Tokens in our Kazakh tokenizer but not in Qwen's vocab
    new_tokens = []
    for token, idx in sorted(kaz_vocab.items(), key=lambda x: x[1]):
        if token not in base_vocab and not token.startswith("<"):  # skip special tokens
            new_tokens.append(token)

    # Limit to max_new_tokens (highest frequency = lowest ID in Unigram)
    new_tokens = new_tokens[:max_new_tokens]
    log.info(
        f"Vocab expansion: {len(new_tokens)} new Kazakh tokens "
        f"(out of {len(kaz_vocab)} total, {len(base_vocab)} base)"
    )
    return new_tokens


def expand_vocab(
    model: nn.Module,
    tokenizer,
    new_tokens: list[str],
) -> int:
    """Add new tokens to the model and tokenizer, initializing embeddings via mean-of-subwords.

    For each new token, we tokenize it using the BASE tokenizer, then average
    the embeddings of those subword pieces. This gives a semantically meaningful
    starting point for the new embedding without requiring an external API.

    Returns the new vocab size.
    """
    if not new_tokens:
        log.info("No new tokens to add.")
        return model.config.vocab_size

    # Add tokens to the tokenizer
    num_added = tokenizer.add_tokens(new_tokens)
    log.info(f"Added {num_added} tokens to tokenizer (new vocab_size={len(tokenizer)})")

    # Resize model embeddings
    old_vocab_size = model.config.vocab_size
    new_vocab_size = len(tokenizer)
    model.resize_token_embeddings(new_vocab_size)

    # Initialize new embeddings via mean-of-subwords
    embed_weight = model.get_input_embeddings().weight
    lm_head_weight = model.get_output_embeddings().weight

    initialized = 0
    with torch.no_grad():
        for i, token in enumerate(new_tokens):
            new_idx = old_vocab_size + i
            if new_idx >= new_vocab_size:
                break

            # Tokenize the new token with the OLD vocab (before the token was added,
            # the tokenizer will split it into existing subwords)
            # We use encode which will split unknown tokens into known subwords
            subword_ids = tokenizer.encode(token, add_special_tokens=False)

            if subword_ids and all(sid < old_vocab_size for sid in subword_ids):
                # Average the embeddings of the subword pieces
                subword_embeds = embed_weight[subword_ids]
                mean_embed = subword_embeds.mean(dim=0)
                embed_weight[new_idx] = mean_embed

                # Same for LM head (if separate from embed)
                if lm_head_weight is not embed_weight:
                    subword_lm = lm_head_weight[subword_ids]
                    lm_head_weight[new_idx] = subword_lm.mean(dim=0)

                initialized += 1

    log.info(
        f"Initialized {initialized}/{num_added} new embeddings via mean-of-subwords "
        f"(remaining {num_added - initialized} use random init)"
    )

    # Update config
    model.config.vocab_size = new_vocab_size
    return new_vocab_size


def graft_engram(model: nn.Module, engram_layer_indices: list[int], **engram_kwargs):
    """Wrap model with Engram modules at specified layers."""
    from kazllm.model.qwen_engram_wrapper import EngramConfig, QymyzForCausalLM

    engram_config = EngramConfig(
        layer_indices=engram_layer_indices,
        **engram_kwargs,
    )

    wrapped = QymyzForCausalLM(model, engram_config)

    # Cast Engram modules to match base model dtype (bf16)
    base_dtype = next(model.parameters()).dtype
    for param in wrapped.engram_parameters():
        param.data = param.data.to(base_dtype)

    return wrapped


def save_checkpoint(model, tokenizer, output_dir: Path):
    """Save the QymyzLM checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model state dict
    state_dict = model.state_dict()
    torch.save(state_dict, output_dir / "pytorch_model.bin")

    # Save tokenizer
    tokenizer.save_pretrained(output_dir / "tokenizer")

    # Save config
    config = {
        "base_model": "Qwen/Qwen2.5-1.5B",
        "model_type": "qymyz",
        "engram_layer_indices": model.engram_config.layer_indices,
        "engram_ngram_orders": model.engram_config.ngram_orders,
        "engram_num_heads": model.engram_config.num_heads,
        "engram_table_size": model.engram_config.table_size,
        "engram_slot_dim": model.engram_config.slot_dim,
        "vocab_size": model.config.vocab_size,
        "hidden_size": model.config.hidden_size,
        "num_hidden_layers": model.config.num_hidden_layers,
        "num_attention_heads": model.config.num_attention_heads,
        "num_key_value_heads": model.config.num_key_value_heads,
    }
    (output_dir / "qymyz_config.json").write_text(json.dumps(config, indent=2))

    # Also save the base model's HF config for reference
    model.config.save_pretrained(output_dir)

    total_params = sum(p.numel() for p in model.parameters())
    engram_params = sum(p.numel() for p in model.engram_parameters())
    log.info(
        f"Checkpoint saved to {output_dir}\n"
        f"  Total params: {total_params / 1e9:.2f}B\n"
        f"  Base params:  {(total_params - engram_params) / 1e9:.2f}B\n"
        f"  Engram params: {engram_params / 1e6:.1f}M\n"
        f"  Vocab size:   {model.config.vocab_size}"
    )


def load_checkpoint(checkpoint_dir: Path, device: str = "cpu"):
    """Load a saved QymyzLM checkpoint for training or inference."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from kazllm.model.qwen_engram_wrapper import EngramConfig, QymyzForCausalLM

    checkpoint_dir = Path(checkpoint_dir)
    qymyz_config = json.loads((checkpoint_dir / "qymyz_config.json").read_text())

    # Load base model with expanded vocab
    base_model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Graft Engram
    engram_config = EngramConfig(
        layer_indices=qymyz_config["engram_layer_indices"],
        ngram_orders=qymyz_config["engram_ngram_orders"],
        num_heads=qymyz_config["engram_num_heads"],
        table_size=qymyz_config["engram_table_size"],
        slot_dim=qymyz_config["engram_slot_dim"],
    )
    model = QymyzForCausalLM(base_model, engram_config)

    # Load full state dict (includes Engram weights)
    state_dict = torch.load(checkpoint_dir / "pytorch_model.bin", map_location=device)
    model.load_state_dict(state_dict, strict=False)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir / "tokenizer")

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Build QymyzLM continual PT checkpoint")
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-1.5B",
        help="HuggingFace model ID for the base model",
    )
    parser.add_argument(
        "--kazakh-tokenizer",
        default=str(PROJECT_ROOT / "data/tokenizer/kaz_sp_unigram_50k/hf_tokenizer"),
        help="Path to trained Kazakh HF tokenizer",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8000,
        help="Maximum number of Kazakh tokens to add to vocabulary",
    )
    parser.add_argument(
        "--engram-layers",
        type=int,
        nargs="+",
        default=[2, 7],
        help="Layer indices to inject Engram",
    )
    parser.add_argument(
        "--engram-table-size",
        type=int,
        default=500_003,
        help="Engram hash table size (prime preferred)",
    )
    parser.add_argument(
        "--engram-num-heads",
        type=int,
        default=4,
        help="Number of hash heads per N-gram order",
    )
    parser.add_argument(
        "--engram-slot-dim",
        type=int,
        default=64,
        help="Embedding dimension per hash table slot",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "checkpoints/qymyz1_5b_init"),
        help="Directory to save the initialized checkpoint",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load model and run 1 forward pass without saving",
    )
    args = parser.parse_args()

    # Step 1: Load base model and tokenizers
    model, base_tokenizer = load_base_model_and_tokenizer(args.base_model)
    kaz_tokenizer = load_kazakh_tokenizer(args.kazakh_tokenizer)

    # Step 2: Find and add new Kazakh tokens
    new_tokens = find_new_tokens(base_tokenizer, kaz_tokenizer, args.max_new_tokens)
    expand_vocab(model, base_tokenizer, new_tokens)

    # Step 3: Graft Engram
    model = graft_engram(
        model,
        engram_layer_indices=args.engram_layers,
        num_heads=args.engram_num_heads,
        table_size=args.engram_table_size,
        slot_dim=args.engram_slot_dim,
    )

    # Step 4: Smoke test
    log.info("Running smoke test (1 forward pass)...")
    model.eval()
    with torch.no_grad():
        dummy_ids = torch.randint(0, model.config.vocab_size, (1, 32), device=model.device)
        outputs = model(input_ids=dummy_ids)
        log.info(f"Smoke test passed. Output logits shape: {outputs.logits.shape}")

    if args.dry_run:
        log.info("Dry run complete. Not saving checkpoint.")
        return

    # Step 5: Save
    save_checkpoint(model, base_tokenizer, Path(args.output_dir))
    log.info("Done! Ready for continual pretraining.")


if __name__ == "__main__":
    main()
