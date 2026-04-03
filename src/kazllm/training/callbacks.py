"""Training callbacks for throughput logging and monitoring."""

import time

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class ThroughputCallback(TrainerCallback):
    """Logs tokens/sec and model FLOP utilisation (MFU) to W&B and console."""

    # A100 80GB theoretical bf16 peak: 312 TFLOP/s
    A100_TFLOPS = 312e12

    def __init__(self, context_length: int = 2048, model_params: int = 1_000_000_000):
        self.context_length = context_length
        self.model_params = model_params
        self._t0 = None
        self._tokens_at_t0 = 0

    def on_step_begin(self, args, state, control, **kwargs):
        if self._t0 is None:
            self._t0 = time.perf_counter()
            self._tokens_at_t0 = state.global_step * (
                args.per_device_train_batch_size
                * args.gradient_accumulation_steps
                * self.context_length
                * args.world_size
            )

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):
        if logs is None or self._t0 is None:
            return

        elapsed = time.perf_counter() - self._t0
        tokens_seen = (
            state.global_step
            * args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            * self.context_length
            * args.world_size
        ) - self._tokens_at_t0

        tok_per_sec = tokens_seen / max(elapsed, 1e-6)
        # MFU approximation: 6 * N * T / (wall_time * num_gpus * peak_flops)
        flops_per_token = 6 * self.model_params
        mfu = tok_per_sec * flops_per_token / (args.world_size * self.A100_TFLOPS)

        logs["throughput_tok_per_sec"] = tok_per_sec
        logs["mfu"] = mfu
