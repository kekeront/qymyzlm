"""Thin wrapper over lm-evaluation-harness for benchmark evaluation.

Installed lm-eval ships no Kazakh tasks, so the bundled custom task directory
(``kazllm/eval/tasks``) is registered via ``TaskManager(include_path=...)``.
Unknown benchmark names and unregistered lm-eval tasks fail fast — published
numbers come from ``evallab/`` runners, never from silently-skipped benchmarks.
"""

import logging
from pathlib import Path

from kazllm.eval.benchmarks import BENCHMARK_TASKS, TASKS_DIR
from kazllm.eval.results import BenchmarkResult, EvalRun

log = logging.getLogger(__name__)


def run_benchmarks(
    model_path: str | Path,
    benchmark_names: list[str],
    output_dir: str | Path,
    model_dtype: str = "float16",
) -> EvalRun:
    """Run lm-eval benchmarks and save results.

    Args:
        model_path: Path to HF-format model checkpoint.
        benchmark_names: List of benchmark names (keys in BENCHMARK_TASKS).
        output_dir: Directory to save results.json.
        model_dtype: Inference dtype ("float16" default — Kaggle T4/P100 and other
            pre-Ampere GPUs have no bf16).

    Returns:
        EvalRun with all benchmark results.

    Raises:
        ValueError: On unknown benchmark names, or if a mapped lm-eval task is
            missing from the registry even with the bundled task dir included.
    """
    try:
        import lm_eval
        from lm_eval.tasks import TaskManager
    except ImportError as err:
        raise ImportError("Install lm-eval: pip install 'lm-eval>=0.4.3'") from err

    unknown = [name for name in benchmark_names if name not in BENCHMARK_TASKS]
    if unknown:
        raise ValueError(
            f"Unknown benchmark(s) {unknown}; this package defines {sorted(BENCHMARK_TASKS)}. "
            "TUMLU-mini, KazQAD, and FLORES-200 have no runner here — the canonical home "
            "for all published benchmark numbers is evallab/ (package kazeval)."
        )

    run = EvalRun(model_path=str(model_path))
    if not benchmark_names:
        log.warning("No benchmarks requested")
        return run

    tasks_to_run = [BENCHMARK_TASKS[name]["task"] for name in benchmark_names]

    task_manager = TaskManager(include_path=str(TASKS_DIR))
    missing = [task for task in tasks_to_run if task not in task_manager.all_tasks]
    if missing:
        raise ValueError(
            f"lm-eval task(s) {missing} not in the registry even with the bundled task dir "
            f"({TASKS_DIR}). Check that kazllm/eval/tasks/** is intact; published numbers "
            "come from evallab/ runners."
        )

    log.info(f"Running benchmarks: {tasks_to_run}")
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},dtype={model_dtype}",
        tasks=tasks_to_run,
        task_manager=task_manager,
        batch_size=16,
        log_samples=False,
    )

    for name, task_name in zip(benchmark_names, tasks_to_run):
        cfg = BENCHMARK_TASKS[name]
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
