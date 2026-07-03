"""Benchmark task definitions for the engine's lm-eval plumbing.

Installed lm-eval (0.4.11) ships NO Kazakh knowledge tasks — there is no
``kazmmlu``, ``tumlu_mini``, or ``kazqad`` in its default TaskManager registry.
KazMMLU (Kazakh subset) is therefore bundled here as a custom task directory
(``kazllm/eval/tasks/kazmmlu_kaz/``) and registered at runtime via
``TaskManager(include_path=str(TASKS_DIR))`` — see :mod:`kazllm.eval.harness`.

TUMLU-mini and KazQAD have no runner in this package; the canonical home for
all published benchmark numbers is ``evallab/`` (package ``kazeval``).
"""

from pathlib import Path

# Bundled custom lm-eval task directory. Passed to TaskManager(include_path=...)
# by harness.run_benchmarks; also usable from the CLI:
#   lm_eval --tasks kazmmlu_kaz --include_path src/kazllm/eval/tasks ...
TASKS_DIR = Path(__file__).resolve().parent / "tasks"

BENCHMARK_TASKS = {
    "kazmmlu": {
        # Bundled group task: 12 Kazakh-language subjects, 9,870 test questions,
        # micro-averaged acc (weight_by_size).
        "task": "kazmmlu_kaz",
        # Dev-limited: the KazMMLU dev split has only 3 exemplars per subject, so
        # 3-shot is the maximum (lm-eval's first_n sampler hard-fails at 5).
        # The shot count is set in the task yaml; this entry is result metadata.
        "num_fewshot": 3,
        "metric": "acc",
        "description": (
            "KazMMLU Kazakh subset: 9,870 multiple-choice questions, 3-shot (dev-limited)"
        ),
    },
}

# FLORES-200 translation pairs: no runner in this package yet — the chrF++ helper
# lives in kazllm.eval.metrics; the canonical runner home is evallab/.
FLORES_TASKS = [
    {"src_lang": "kaz_Cyrl", "tgt_lang": "eng_Latn", "name": "flores200_kaz_eng"},
    {"src_lang": "kaz_Cyrl", "tgt_lang": "rus_Cyrl", "name": "flores200_kaz_rus"},
]
