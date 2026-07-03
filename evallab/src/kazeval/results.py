"""Result records — the committed JSON behind every leaderboard row.

One JSON file per record lives in ``evallab/results/``. The README leaderboard is
rendered from these files by :mod:`kazeval.leaderboard` and is never edited by hand.

Record schema (all keys required, no extras):

- ``model``: model id as published (e.g. ``intfloat/multilingual-e5-large``).
- ``revision``: model commit hash / revision, or ``null`` when unknown.
- ``task``: task id (``KazQADRetrieval``, ``KazQADReranking``, ``KazQAD-HardNeg``,
  ``KazMMLU-kk``, ``Qorgau-kk``).
- ``protocol``: human-readable description of the exact measurement protocol.
- ``split``: evaluated split (usually ``test``).
- ``metrics``: non-empty mapping of metric name to finite float.
- ``provenance``: ``"measured"`` (produced in-lab — a kazeval runner or this
  repo's baseline scripts) or ``"reported"`` (copied from an external source,
  pending in-lab reproduction).
- ``source``: where the numbers come from (runner name / URL / arXiv id).
- ``date``: ISO ``YYYY-MM-DD`` date of the measurement or of the source report.
"""

import json
import math
import re
from dataclasses import asdict, dataclass, fields
from datetime import date as _date
from pathlib import Path

PROVENANCE_VALUES = ("measured", "reported")

_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class ResultRecord:
    """A single benchmark measurement (or externally reported number)."""

    model: str
    revision: str | None
    task: str
    protocol: str
    split: str
    metrics: dict[str, float]
    provenance: str
    source: str
    date: str


def validate_record(record: ResultRecord) -> None:
    """Raise ``ValueError`` (with context) if the record violates the schema."""
    for name in ("model", "task", "protocol", "split", "source", "date"):
        value = getattr(record, name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"record field {name!r} must be a non-empty string, got {value!r}")
    if record.revision is not None and (
        not isinstance(record.revision, str) or not record.revision.strip()
    ):
        raise ValueError(
            f"record field 'revision' must be None or a non-empty string, got {record.revision!r}"
        )
    if record.provenance not in PROVENANCE_VALUES:
        raise ValueError(
            f"record field 'provenance' must be one of {PROVENANCE_VALUES}, "
            f"got {record.provenance!r}"
        )
    if not isinstance(record.metrics, dict) or not record.metrics:
        raise ValueError(f"record field 'metrics' must be a non-empty dict, got {record.metrics!r}")
    for key, value in record.metrics.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"metric names must be non-empty strings, got {key!r}")
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise ValueError(f"metric {key!r} must be a finite number, got {value!r}")
    try:
        _date.fromisoformat(record.date)
    except ValueError as err:
        raise ValueError(
            f"record field 'date' must be ISO YYYY-MM-DD, got {record.date!r}"
        ) from err


def record_to_dict(record: ResultRecord) -> dict[str, object]:
    """Validated plain-dict form of a record (JSON-ready)."""
    validate_record(record)
    return asdict(record)


def record_from_dict(payload: dict[str, object], *, context: str = "record") -> ResultRecord:
    """Build and validate a record from a plain dict; reject unknown/missing keys."""
    expected = {f.name for f in fields(ResultRecord)}
    got = set(payload)
    if got != expected:
        missing = sorted(expected - got)
        unknown = sorted(got - expected)
        raise ValueError(f"{context}: bad keys (missing={missing}, unknown={unknown})")
    record = ResultRecord(**payload)  # type: ignore[arg-type]
    record.metrics = (
        {k: float(v) for k, v in record.metrics.items()}
        if isinstance(record.metrics, dict)
        else record.metrics
    )
    validate_record(record)
    return record


def _slug(value: str) -> str:
    return _SLUG_RE.sub("-", value).strip("-")


def record_path(record: ResultRecord, results_dir: Path) -> Path:
    """Deterministic file path for a record: ``{date}__{task}__{model}.json``.

    Re-running the same (date, task, model) overwrites — rows stay regenerable.
    """
    return results_dir / f"{record.date}__{_slug(record.task)}__{_slug(record.model)}.json"


def save_record(record: ResultRecord, results_dir: Path) -> Path:
    """Validate and write a record as pretty JSON; return the written path."""
    payload = record_to_dict(record)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = record_path(record, results_dir)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def load_record(path: Path) -> ResultRecord:
    """Load and validate a single record file."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise ValueError(f"{path}: not valid JSON: {err}") from err
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(payload).__name__}")
    return record_from_dict(payload, context=str(path))


def load_records(results_dir: Path) -> list[ResultRecord]:
    """Load every ``*.json`` record in a directory, sorted by (task, date, model)."""
    if not results_dir.is_dir():
        raise ValueError(f"results dir does not exist: {results_dir}")
    records = [load_record(path) for path in sorted(results_dir.glob("*.json"))]
    records.sort(key=lambda r: (r.task, r.date, r.model))
    return records
