"""Render the evallab leaderboard into README.md from committed result records.

The README section between ``<!-- LEADERBOARD:START -->`` and
``<!-- LEADERBOARD:END -->`` is owned by this module — NEVER edit it by hand.
Regenerate after every new record::

    python -m kazeval.leaderboard

One markdown table per track (embedding/retrieval, generative, safety), with a
provenance column: ``measured`` rows were produced by a kazeval runner in this
repo; ``reported`` rows are copied from external sources and await in-lab
reproduction.
"""

import argparse
import sys
from pathlib import Path

from kazeval.results import ResultRecord, load_records

START_MARKER = "<!-- LEADERBOARD:START -->"
END_MARKER = "<!-- LEADERBOARD:END -->"

EVALLAB_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = EVALLAB_DIR / "results"
DEFAULT_README = EVALLAB_DIR / "README.md"

#: Track title -> task ids shown in that track's table (row order follows this).
TRACKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Embedding / retrieval (KazQAD)",
        ("KazQADRetrieval", "KazQADReranking", "KazQAD-HardNeg"),
    ),
    ("Generative (KazMMLU)", ("KazMMLU-kk",)),
    ("Safety (Qorgau)", ("Qorgau-kk",)),
)

#: Ranking metric per task (rows sort descending by it within a task).
MAIN_METRIC = {
    "KazQADRetrieval": "ndcg_at_10",
    "KazQADReranking": "map_at_1000",
    "KazQAD-HardNeg": "mrr",
    "KazMMLU-kk": "acc",
    "Qorgau-kk": "safe_rate",
}

_MISSING = "—"


def _format_metric(value: float) -> str:
    return f"{value:.4f}"


def _short_revision(revision: str | None) -> str:
    return revision[:8] if revision else _MISSING


def _metric_columns(task_ids: tuple[str, ...], records: list[ResultRecord]) -> list[str]:
    """Union of metric names: main metrics first (track order), then the rest sorted."""
    present: set[str] = set()
    for record in records:
        present.update(record.metrics)
    columns = [
        MAIN_METRIC[task_id]
        for task_id in task_ids
        if task_id in MAIN_METRIC and MAIN_METRIC[task_id] in present
    ]
    ordered = list(dict.fromkeys(columns))
    ordered += sorted(present - set(ordered))
    return ordered


def _sort_key(record: ResultRecord, task_ids: tuple[str, ...]) -> tuple[int, float, str, str]:
    task_pos = task_ids.index(record.task) if record.task in task_ids else len(task_ids)
    main = record.metrics.get(MAIN_METRIC.get(record.task, ""), float("-inf"))
    return (task_pos, -main, record.model, record.date)


def _render_table(task_ids: tuple[str, ...], records: list[ResultRecord]) -> str:
    metric_columns = _metric_columns(task_ids, records)
    header = ["Model", "Revision", "Task", "Split", *metric_columns, "Provenance", "Source", "Date"]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    for record in sorted(records, key=lambda r: _sort_key(r, task_ids)):
        cells = [
            record.model,
            _short_revision(record.revision),
            record.task,
            record.split,
            *(
                _format_metric(record.metrics[name]) if name in record.metrics else _MISSING
                for name in metric_columns
            ),
            record.provenance,
            record.source,
            record.date,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_leaderboard(records: list[ResultRecord]) -> str:
    """Full leaderboard markdown (all tracks) from validated records."""
    remaining = list(records)
    sections: list[str] = [
        "_Auto-generated from `evallab/results/*.json` by `python -m kazeval.leaderboard` "
        "— do not edit by hand._",
    ]
    tracks = [*TRACKS]
    leftover_tasks = tuple(
        sorted({r.task for r in records} - {t for _, ids in TRACKS for t in ids})
    )
    if leftover_tasks:
        tracks.append(("Other", leftover_tasks))
    for title, task_ids in tracks:
        track_records = [r for r in remaining if r.task in task_ids]
        sections.append(f"### {title}")
        if track_records:
            sections.append(_render_table(task_ids, track_records))
        else:
            sections.append("_No results yet._")
    return "\n\n".join(sections) + "\n"


def inject(readme_path: Path, content: str) -> bool:
    """Replace the marker-delimited README section; return True if it changed."""
    text = readme_path.read_text(encoding="utf-8")
    if START_MARKER not in text or END_MARKER not in text:
        raise ValueError(f"{readme_path}: missing {START_MARKER!r} / {END_MARKER!r} markers")
    before, rest = text.split(START_MARKER, 1)
    _, after = rest.split(END_MARKER, 1)
    updated = f"{before}{START_MARKER}\n{content.rstrip()}\n{END_MARKER}{after}"
    if updated == text:
        return False
    readme_path.write_text(updated, encoding="utf-8")
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kazeval.leaderboard",
        description="Render the leaderboard section of evallab/README.md from result records.",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--readme", type=Path, default=DEFAULT_README)
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 2 if the README section is stale",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = load_records(args.results_dir)
    content = render_leaderboard(records)
    if args.check:
        text = args.readme.read_text(encoding="utf-8")
        expected = f"{START_MARKER}\n{content.rstrip()}\n{END_MARKER}"
        if expected in text:
            print(f"{args.readme}: leaderboard up to date ({len(records)} records)")
            return 0
        print(f"{args.readme}: leaderboard STALE — rerun python -m kazeval.leaderboard")
        return 2
    changed = inject(args.readme, content)
    state = "updated" if changed else "already up to date"
    print(f"{args.readme}: {state} ({len(records)} records)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
