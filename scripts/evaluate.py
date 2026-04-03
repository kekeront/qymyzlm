"""Run evaluation benchmarks on a trained KazLLM checkpoint."""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from kazllm.eval.harness import run_benchmarks
from kazllm.utils.logging import setup_logging

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()

    model_path = Path(cfg.training.checkpoint_dir) / "hf_export"
    if not model_path.exists():
        log.error(f"No HF export found at {model_path}. Run consolidate_checkpoint first.")
        return

    benchmark_names = [b["name"] if isinstance(b, dict) else b for b in cfg.eval.benchmarks]
    results_dir = Path(cfg.eval.results_dir) / model_path.parent.name

    log.info(f"Evaluating {model_path} on {benchmark_names}")
    run = run_benchmarks(
        model_path=model_path,
        benchmark_names=benchmark_names,
        output_dir=results_dir,
        model_dtype=cfg.eval.model_dtype,
    )

    for result in run.results:
        log.info(f"  {result.benchmark}: {result.metric}={result.value:.4f}")


if __name__ == "__main__":
    main()
