"""Loss computation with cross-document masking."""

import torch.nn.functional as F
from torch import Tensor


def compute_lm_loss(
    logits: Tensor,
    labels: Tensor,
    vocab_size: int,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Cross-entropy loss with optional label smoothing.

    Args:
        logits: (batch, seq_len, vocab_size)
        labels: (batch, seq_len) — use -100 to mask positions
        vocab_size: size of vocabulary
        label_smoothing: label smoothing coefficient (0 = no smoothing)
    """
    shift_logits = logits[..., :-1, :].contiguous().view(-1, vocab_size)
    shift_labels = labels[..., 1:].contiguous().view(-1)
    return F.cross_entropy(
        shift_logits,
        shift_labels,
        ignore_index=-100,
        label_smoothing=label_smoothing,
    )
