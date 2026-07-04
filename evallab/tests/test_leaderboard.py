"""Leaderboard rendering (golden output) and README marker injection."""

from pathlib import Path

import pytest
from kazeval.leaderboard import (
    END_MARKER,
    START_MARKER,
    inject,
    main,
    render_leaderboard,
)
from kazeval.results import load_records

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
FIXTURE_RESULTS = FIXTURES_DIR / "results"
GOLDEN = FIXTURES_DIR / "leaderboard_golden.md"


def test_render_matches_golden():
    records = load_records(FIXTURE_RESULTS)
    assert render_leaderboard(records) == GOLDEN.read_text(encoding="utf-8")


def test_render_covers_all_tracks_and_other():
    content = render_leaderboard(load_records(FIXTURE_RESULTS))
    assert "### Embedding / retrieval (KazQAD)" in content
    assert "### Generative (KazMMLU)" in content
    assert "### Safety (Qorgau)" in content
    assert "_No results yet._" in content  # Qorgau track is empty in fixtures
    assert "### Other" in content  # unknown-task record is not silently dropped
    assert "do not edit by hand" in content


def test_render_marks_provenance_and_missing_metrics():
    content = render_leaderboard(load_records(FIXTURE_RESULTS))
    # the detailed row (carries the source + full metrics), not the headline row
    hardneg_row = next(
        line for line in content.splitlines() if "KazQAD-HardNeg" in line and "0.9090" in line
    )
    assert "reported" in hardneg_row
    assert "0.9090" in hardneg_row
    assert "—" in hardneg_row  # ndcg_at_10 not reported for the hardneg row


def test_headline_precedes_full_records_and_groups_by_track():
    content = render_leaderboard(load_records(FIXTURE_RESULTS))
    assert content.index("### At a glance") < content.index("### Full records")
    glance = content.split("### Full records")[0]
    assert "nDCG@10" in glance  # headline uses one comparable metric per track
    assert "0.4120" in glance  # measured retrieval score surfaces in the headline


def _readme(tmp_path: Path, section: str = "stale") -> Path:
    path = tmp_path / "README.md"
    path.write_text(
        f"# Title\n\nintro\n\n{START_MARKER}\n{section}\n{END_MARKER}\n\ntail\n",
        encoding="utf-8",
    )
    return path


def test_inject_replaces_only_marked_section(tmp_path):
    readme = _readme(tmp_path)
    assert inject(readme, "NEW CONTENT\n") is True
    text = readme.read_text(encoding="utf-8")
    assert text.startswith("# Title\n\nintro\n\n")
    assert text.endswith("\n\ntail\n")
    assert f"{START_MARKER}\nNEW CONTENT\n{END_MARKER}" in text
    assert "stale" not in text
    assert inject(readme, "NEW CONTENT\n") is False  # idempotent


def test_inject_requires_markers(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("no markers here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="markers"):
        inject(readme, "content")


def test_cli_write_then_check(tmp_path, capsys):
    readme = _readme(tmp_path)
    argv = ["--results-dir", str(FIXTURE_RESULTS), "--readme", str(readme)]
    assert main(argv) == 0
    assert main([*argv, "--check"]) == 0
    # hand-edit the generated section -> --check must flag it
    readme.write_text(
        readme.read_text(encoding="utf-8").replace("0.9090", "0.9999"), encoding="utf-8"
    )
    assert main([*argv, "--check"]) == 2
    assert "STALE" in capsys.readouterr().out
