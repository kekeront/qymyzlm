"""Run KazMMLU (Kazakh-language subset) via lm-eval and commit a result record.

lm-eval 0.4.11 ships NO KazMMLU task (verified against the installed registry),
so kazeval bundles its own task YAMLs at ``kazeval/lm_eval_tasks/kazmmlu_kaz/``
and registers them through ``TaskManager(include_path=...)``. The group task
``kazmmlu_kaz`` micro-averages accuracy over the 12 Kazakh-language subjects
(9,870 test questions) of ``MBZUAI/KazMMLU``.

SHOT COUNT: KazMMLU's dev split holds only 3 exemplars per subject, so 3-shot is
the maximum with dev-sourced shots — lm-eval's ``first_n`` sampler raises an
AssertionError at ``num_fewshot=5`` (verified live). The canonical lab number is
therefore **3-shot**; the repo README's historical "5-shot" 32.8% baseline was
effectively 3-shot for the same reason.

Usage (downloads the model weights + the KazMMLU dataset; never run in tests)::

    python -m kazeval.run_kazmmlu --model Qwen/Qwen3-0.6B-Base

The full lm-eval output (per-subject accuracies included) is dumped under
``{output-dir}/raw/``; the aggregate accuracy becomes the committed record.
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

from kazeval.results import ResultRecord, save_record

GROUP_TASK = "kazmmlu_kaz"
N_SUBJECTS = 12
N_TEST_QUESTIONS = 9870
MAX_FEWSHOT = 3  # KazMMLU dev has exactly 3 exemplars per subject
TASKS_DIR = Path(__file__).resolve().parent / "lm_eval_tasks"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def build_task_manager() -> Any:
    """TaskManager with the bundled KazMMLU task YAMLs registered.

    lm-eval 0.4.11 has no built-in KazMMLU; if a future lm-eval ships one, the
    bundled YAMLs still pin OUR protocol (prompt format, 3-shot, micro-average).
    """
    from lm_eval.tasks import TaskManager

    task_manager = TaskManager(include_path=str(TASKS_DIR))
    if GROUP_TASK not in task_manager.all_tasks:
        raise RuntimeError(
            f"bundled task dir {TASKS_DIR} failed to register {GROUP_TASK!r}; check the YAML files"
        )
    return task_manager


def fewshot(value: str) -> int:
    number = int(value)
    if not 0 <= number <= MAX_FEWSHOT:
        raise argparse.ArgumentTypeError(
            f"--num-fewshot must be 0..{MAX_FEWSHOT}: KazMMLU dev has only "
            f"{MAX_FEWSHOT} exemplars per subject (lm-eval asserts beyond that)"
        )
    return number


def extract_group_accuracy(results: dict[str, Any]) -> float:
    """Aggregate accuracy of the group task from lm-eval's results payload."""
    for section in ("results", "groups"):
        entry = results.get(section, {}).get(GROUP_TASK)
        if entry and "acc,none" in entry:
            return float(entry["acc,none"])
    raise ValueError(
        f"no aggregate 'acc,none' for {GROUP_TASK!r} in lm-eval output; "
        f"results keys: {sorted(results.get('results', {}))}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kazeval.run_kazmmlu",
        description="Run KazMMLU (Kazakh subset, 3-shot) via lm-eval and record the result.",
    )
    parser.add_argument("--model", required=True, help="HF model id or local checkpoint path")
    parser.add_argument(
        "--model-revision", default=None, help="model revision to pin in the record"
    )
    parser.add_argument(
        "--model-args",
        default="dtype=float16",
        help="extra lm-eval hf model_args, comma-separated (default: dtype=float16)",
    )
    parser.add_argument(
        "--num-fewshot",
        type=fewshot,
        default=MAX_FEWSHOT,
        help=f"shots from dev, 0..{MAX_FEWSHOT} (default {MAX_FEWSHOT} — the dev split maximum)",
    )
    parser.add_argument("--batch-size", default="8", help="int or 'auto' (lm-eval semantics)")
    parser.add_argument("--device", default=None, help="e.g. cuda:0 (default: lm-eval auto)")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="debug: cap docs per task; NO record is written when set",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="where the kazeval record is written (default: evallab/results)",
    )
    parser.add_argument(
        "--date",
        default=dt.date.today().isoformat(),
        help="ISO date stamped on the record (default: today)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    import lm_eval

    model_args = f"pretrained={args.model}"
    if args.model_revision:
        model_args += f",revision={args.model_revision}"
    if args.model_args:
        model_args += f",{args.model_args}"

    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=[GROUP_TASK],
        # the bundled YAML already pins 3-shot; only override when asked for less
        num_fewshot=args.num_fewshot if args.num_fewshot != MAX_FEWSHOT else None,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
        task_manager=build_task_manager(),
    )
    if results is None:  # lm-eval returns None on non-main ranks
        return 0

    accuracy = extract_group_accuracy(results)
    model_slug = args.model.replace("/", "-")
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"kazmmlu_kaz__{model_slug}__{args.date}.json"
    raw_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"kazmmlu_kaz acc={accuracy:.4f} ({args.num_fewshot}-shot)")
    print(f"  full lm-eval output: {raw_path}")

    if args.limit is not None:
        print(f"--limit={args.limit} set: debug run, record NOT written", file=sys.stderr)
        return 0

    record = ResultRecord(
        model=args.model,
        revision=args.model_revision,
        task="KazMMLU-kk",
        protocol=(
            f"KazMMLU Kazakh-language subset ({N_SUBJECTS} subjects, "
            f"{N_TEST_QUESTIONS} test questions), {args.num_fewshot}-shot from dev, "
            f"multiple_choice loglikelihood over answer letters, bundled kazeval "
            f"task YAMLs (group {GROUP_TASK})"
        ),
        split="test",
        metrics={"acc": accuracy},
        provenance="measured",
        source=f"kazeval.run_kazmmlu (lm-eval {lm_eval.__version__})",
        date=args.date,
    )
    path = save_record(record, args.output_dir)
    print(f"  record: {path}")
    print("now regenerate the leaderboard: python -m kazeval.leaderboard")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
