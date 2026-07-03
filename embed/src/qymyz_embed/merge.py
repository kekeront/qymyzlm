"""Model soup: weight-average two checkpoints of the same base (arXiv 2603.22290).

The Less-is-More recipe averages the fine-tuned weights 0.5/0.5 with the ORIGINAL mE5 base
model. For the mE5 module stack (Transformer -> mean Pooling -> Normalize) only the
Transformer carries parameters — Pooling and Normalize contribute zero state-dict keys —
so averaging the full state_dict is exactly "average the Transformer weights".

Verified pitfall: SentenceTransformer.state_dict() returns LIVE tensor references;
load_state_dict() would mutate the source tensors mid-average. soup_state_dicts() clones
every tensor before touching it.

Usage:
    python -m qymyz_embed.merge CKPT_FINETUNED CKPT_BASE --output OUT [--alpha 0.5]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path

import torch


def soup_state_dicts(
    sd_a: Mapping[str, torch.Tensor],
    sd_b: Mapping[str, torch.Tensor],
    alpha: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Return alpha * sd_a + (1 - alpha) * sd_b, computed in fp32, cast back to sd_a dtypes.

    Refuses on any key or shape mismatch (checkpoints must share the architecture).
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if sd_a.keys() != sd_b.keys():
        only_a = sorted(sd_a.keys() - sd_b.keys())
        only_b = sorted(sd_b.keys() - sd_a.keys())
        raise ValueError(
            f"state-dict key mismatch (not the same architecture): "
            f"only in A: {only_a[:5]}{'...' if len(only_a) > 5 else ''}; "
            f"only in B: {only_b[:5]}{'...' if len(only_b) > 5 else ''}"
        )
    merged: dict[str, torch.Tensor] = {}
    for key, tensor_a in sd_a.items():
        tensor_b = sd_b[key]
        if tensor_a.shape != tensor_b.shape:
            raise ValueError(
                f"shape mismatch at {key!r}: {tuple(tensor_a.shape)} vs {tuple(tensor_b.shape)}"
            )
        # clone(): state_dict tensors are live references — never average in place.
        a32 = tensor_a.detach().clone().float()
        b32 = tensor_b.detach().clone().float()
        merged[key] = (alpha * a32 + (1.0 - alpha) * b32).to(tensor_a.dtype)
    return merged


def soup_checkpoints(
    path_or_id_a: str, path_or_id_b: str, output: Path, alpha: float = 0.5
) -> None:
    """Load two SentenceTransformer checkpoints, average, save to output.

    Loads A last so its non-weight config (prompts, pooling) is what gets saved.
    """
    from sentence_transformers import SentenceTransformer

    model_a = SentenceTransformer(path_or_id_a)
    model_b = SentenceTransformer(path_or_id_b)
    merged = soup_state_dicts(model_a.state_dict(), model_b.state_dict(), alpha=alpha)
    model_a.load_state_dict(merged)
    output.parent.mkdir(parents=True, exist_ok=True)
    model_a.save(str(output))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="0.5/0.5 model soup of two same-base checkpoints (arXiv 2603.22290)"
    )
    parser.add_argument("checkpoint_a", help="path or HF id (its config wins in the output)")
    parser.add_argument("checkpoint_b", help="path or HF id (e.g. the mE5 base)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="weight of checkpoint_a (default 0.5)"
    )
    args = parser.parse_args(argv)
    soup_checkpoints(args.checkpoint_a, args.checkpoint_b, args.output, alpha=args.alpha)
    print(
        f"souped {args.alpha:.2f}*{args.checkpoint_a} + "
        f"{1 - args.alpha:.2f}*{args.checkpoint_b} -> {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
