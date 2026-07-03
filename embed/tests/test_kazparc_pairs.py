"""Offline tests for KazParC pair extraction — local fixture JSONL, no downloads."""

import json
from pathlib import Path

import pytest
from qymyz_embed.data import kazparc_pairs

FIXTURE = Path(__file__).parent / "fixtures" / "kazparc_sample.jsonl"
# Fixture layout: rows 1-3 fully parallel; row 4 empty kk; row 5 null en; row 6 == row 1.


def fixture_rows() -> list[dict[str, str | None]]:
    return list(kazparc_pairs.read_jsonl_rows(FIXTURE))


def test_kk_ru_pairs_prefixed_and_skips_empty() -> None:
    pairs = list(kazparc_pairs.rows_to_pairs(fixture_rows(), (("kk", "ru"),)))
    # rows 1, 2, 3, 5 valid; row 4 empty kk skipped; row 6 deduped
    assert len(pairs) == 4
    assert all(set(p) == {"anchor", "positive"} for p in pairs)
    assert all(p["anchor"].startswith("query: ") for p in pairs)
    assert all(p["positive"].startswith("passage: ") for p in pairs)
    assert pairs[0]["anchor"] == "query: Қазақстан — Орталық Азиядағы мемлекет."
    assert pairs[0]["positive"] == "passage: Казахстан — государство в Центральной Азии."


def test_kk_en_skips_null_side() -> None:
    pairs = list(kazparc_pairs.rows_to_pairs(fixture_rows(), (("kk", "en"),)))
    assert len(pairs) == 3  # rows 1, 2, 3 (row 5 has null en)


def test_bidirectional_directions() -> None:
    directions = kazparc_pairs.parse_directions("kk-ru,ru-kk")
    pairs = list(kazparc_pairs.rows_to_pairs(fixture_rows(), directions))
    assert len(pairs) == 8  # 4 per direction
    ru_anchors = [p for p in pairs if p["anchor"] == "query: Астана — столица Казахстана."]
    assert len(ru_anchors) == 1
    assert ru_anchors[0]["positive"] == "passage: Астана — Қазақстанның астанасы."


def test_no_prefix_emits_raw_text() -> None:
    pairs = list(kazparc_pairs.rows_to_pairs(fixture_rows(), (("kk", "en"),), prefixed=False))
    assert pairs[1] == {
        "anchor": "Астана — Қазақстанның астанасы.",
        "positive": "Astana is the capital of Kazakhstan.",
    }


def test_dedupe_off_keeps_duplicates() -> None:
    pairs = list(kazparc_pairs.rows_to_pairs(fixture_rows(), (("kk", "ru"),), dedupe=False))
    assert len(pairs) == 5  # duplicate row 6 retained


def test_parse_directions() -> None:
    assert kazparc_pairs.parse_directions("kk-ru, en-kk") == (("kk", "ru"), ("en", "kk"))


@pytest.mark.parametrize("spec", ["kk-kk", "kk-de", "kkru", "", "kk-"])
def test_parse_directions_rejects_bad_specs(spec: str) -> None:
    with pytest.raises(ValueError):
        kazparc_pairs.parse_directions(spec)


def test_cli_jsonl_with_limit(tmp_path: Path) -> None:
    out = tmp_path / "pairs.jsonl"
    code = kazparc_pairs.main(
        [
            "--input",
            str(FIXTURE),
            "--output",
            str(out),
            "--directions",
            "kk-ru",
            "--limit",
            "2",
        ]
    )
    assert code == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert all(r["anchor"].startswith("query: ") for r in rows)


def test_cli_hf_dataset(tmp_path: Path) -> None:
    out = tmp_path / "pairs_ds"
    code = kazparc_pairs.main(
        [
            "--input",
            str(FIXTURE),
            "--output",
            str(out),
            "--format",
            "hf",
            "--directions",
            "kk-en",
            "--no-prefix",
        ]
    )
    assert code == 0
    from datasets import Dataset, load_from_disk

    dataset = load_from_disk(str(out))
    assert isinstance(dataset, Dataset)
    assert dataset.column_names == ["anchor", "positive"]  # order defines MNRL roles
    assert len(dataset) == 3
