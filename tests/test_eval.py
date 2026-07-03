"""Eval package tests: metrics, result serialisation, and truthful harness plumbing.

lm-eval 0.4.11 ships no Kazakh tasks, so the harness must (a) register the
bundled custom task dir via TaskManager(include_path=...) and (b) fail fast on
anything it cannot actually run — never skip silently.
"""

import json
from pathlib import Path

import pytest

from kazllm.eval.benchmarks import BENCHMARK_TASKS, TASKS_DIR
from kazllm.eval.metrics import accuracy
from kazllm.eval.results import BenchmarkResult, EvalRun

KAZMMLU_SUBJECT_SLUGS = [
    "biology",
    "chemistry",
    "geography",
    "informatics",
    "kazakh_history",
    "kazakh_language",
    "kazakh_literature",
    "law",
    "math",
    "physics",
    "reading_literacy",
    "world_history",
]


def test_accuracy_basic() -> None:
    assert accuracy(["A", "B", "C"], ["A", "B", "D"]) == pytest.approx(2 / 3)
    assert accuracy([], []) == 0.0


def test_eval_run_save_round_trip(tmp_path: Path) -> None:
    run = EvalRun(model_path="dummy/model")
    run.add(
        BenchmarkResult(
            benchmark="kazmmlu", metric="acc", value=0.328, num_fewshot=3, num_examples=9870
        )
    )
    path = run.save(tmp_path)
    data = json.loads(path.read_text())
    assert data["model_path"] == "dummy/model"
    assert data["results"][0]["benchmark"] == "kazmmlu"
    assert data["results"][0]["value"] == pytest.approx(0.328)


def test_benchmark_tasks_are_dev_limited_three_shot() -> None:
    """KazMMLU dev has 3 exemplars/subject: any '5-shot' config would be a lie."""
    assert BENCHMARK_TASKS["kazmmlu"]["num_fewshot"] == 3
    assert BENCHMARK_TASKS["kazmmlu"]["task"] == "kazmmlu_kaz"


def test_bundled_kazmmlu_task_dir_registers() -> None:
    """The bundled task dir must register the group + all 12 subject tasks (offline)."""
    pytest.importorskip("lm_eval")
    from lm_eval.tasks import TaskManager

    tm = TaskManager(include_path=str(TASKS_DIR), include_defaults=False)
    assert "kazmmlu_kaz" in tm.all_tasks
    for slug in KAZMMLU_SUBJECT_SLUGS:
        assert f"kazmmlu_kaz_{slug}" in tm.all_tasks, slug


def test_run_benchmarks_unknown_name_fails_fast(tmp_path: Path) -> None:
    """Unknown benchmark names must raise, not skip silently (truthful plumbing)."""
    pytest.importorskip("lm_eval")
    from kazllm.eval.harness import run_benchmarks

    with pytest.raises(ValueError, match="evallab"):
        run_benchmarks(
            model_path="dummy/model",
            benchmark_names=["tumlu_mini"],
            output_dir=tmp_path,
        )


def test_run_benchmarks_uses_bundled_task_and_task_manager(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_benchmarks must resolve 'kazmmlu' to the bundled kazmmlu_kaz lm-eval task."""
    lm_eval = pytest.importorskip("lm_eval")
    from kazllm.eval.harness import run_benchmarks

    def fake_simple_evaluate(**kwargs):
        assert kwargs["tasks"] == ["kazmmlu_kaz"]
        assert kwargs["task_manager"] is not None
        return {"results": {"kazmmlu_kaz": {"acc,none": 0.42, "num_examples": 10}}}

    monkeypatch.setattr(lm_eval, "simple_evaluate", fake_simple_evaluate)

    run = run_benchmarks(
        model_path="dummy/model",
        benchmark_names=["kazmmlu"],
        output_dir=tmp_path,
    )

    assert len(run.results) == 1
    result = run.results[0]
    assert result.benchmark == "kazmmlu"
    assert result.value == pytest.approx(0.42)
    assert result.num_fewshot == 3
    assert (tmp_path / "results.json").exists()
