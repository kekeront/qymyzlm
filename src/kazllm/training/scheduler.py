"""Learning rate schedulers for pretraining."""

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def cosine_with_warmup(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Cosine decay with linear warmup.

    Args:
        warmup_steps: Number of warmup steps.
        total_steps: Total training steps.
        min_lr_ratio: Minimum LR as fraction of peak LR.
    """

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)
