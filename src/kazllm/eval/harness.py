"""Thin wrapper over lm-evaluation-harness for benchmark evaluation."""

import logging
from pathlib import Path

from kazllm.eval.benchmarks import BENCHMARK_TASKS
from kazllm.eval.results import BenchmarkResult, EvalRun

log = logging.getLogger(__name__)


def run_benchmarks(
    model_path: str | Path,
    benchmark_names: list[str],
    output_dir: str | Path,
    model_dtype: str = "bfloat16",
) -> EvalRun:
    """Run lm-eval benchmarks and save results.

    Args:
        model_path: Path to HF-format model checkpoint.
        benchmark_names: List of benchmark names (keys in BENCHMARK_TASKS).
        output_dir: Directory to save results.json.
        model_dtype: Inference dtype ("bfloat16" or "float16").

    Returns:
        EvalRun with all benchmark results.
    """
    try:
        import lm_eval
    except ImportError:
        raise ImportError("Install lm-eval: pip install lm-eval>=0.4.3")

    run = EvalRun(model_path=str(model_path))
    tasks_to_run = []

    for name in benchmark_names:
        if name not in BENCHMARK_TASKS:
            log.warning(f"Unknown benchmark: {name}, skipping")
            continue
        tasks_to_run.append(BENCHMARK_TASKS[name]["task"])

    if not tasks_to_run:
        log.warning("No valid benchmarks to run")
        return run

    log.info(f"Running benchmarks: {tasks_to_run}")
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},dtype={model_dtype}",
        tasks=tasks_to_run,
        batch_size=16,
        log_samples=False,
    )

    for name, task_name in zip(benchmark_names, tasks_to_run):
        cfg = BENCHMARK_TASKS.get(name, {})
        metric = cfg.get("metric", "acc")
        task_results = results.get("results", {}).get(task_name, {})
        value = task_results.get(f"{metric},none", task_results.get(metric, 0.0))
        run.add(
            BenchmarkResult(
                benchmark=name,
                metric=metric,
                value=value,
                num_fewshot=cfg.get("num_fewshot", 0),
                num_examples=task_results.get("num_examples", 0),
            )
        )
        log.info(f"{name}: {metric}={value:.4f}")

    run.save(output_dir)
    return run
