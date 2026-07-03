"""Offline tests for the Less-is-More filter — stub sims, exact boundary semantics."""

import pytest
from qymyz_embed.data.synth_filter import (
    DRIFT_MAX,
    TRANSLATION_MIN,
    PairSims,
    filter_rows,
    keep_mask,
    keep_pair,
    sims_from_row,
)

GOOD = {"sim_en": 0.9, "sim_lrl": 0.88, "trans_q": 0.95, "trans_p": 0.93}


def ps(**overrides: float) -> PairSims:
    return PairSims(**{**GOOD, **overrides})


def test_defaults_match_paper() -> None:
    assert DRIFT_MAX == 0.05
    assert TRANSLATION_MIN == 0.85


def test_keeps_good_pair() -> None:
    assert keep_pair(ps())


def test_drift_boundary_exactly_005_keeps() -> None:
    # 0.05 - 0.0 is exactly the double 0.05 -> |drift| == drift_max -> KEEP (<= rule)
    assert keep_pair(ps(sim_en=0.05, sim_lrl=0.0))


def test_drift_above_005_drops() -> None:
    assert not keep_pair(ps(sim_en=0.051, sim_lrl=0.0))
    assert not keep_pair(ps(sim_en=0.0, sim_lrl=0.051))  # drift is symmetric


def test_translation_boundary_exactly_085_drops() -> None:
    # strict > rule: similarity equal to the threshold is dropped
    assert not keep_pair(ps(trans_q=0.85))
    assert not keep_pair(ps(trans_p=0.85))


def test_translation_above_085_keeps() -> None:
    assert keep_pair(ps(trans_q=0.8501, trans_p=0.8501))


def test_both_translation_sides_must_pass() -> None:
    assert not keep_pair(ps(trans_q=0.99, trans_p=0.2))
    assert not keep_pair(ps(trans_q=0.2, trans_p=0.99))


def test_custom_thresholds() -> None:
    sims = ps(sim_en=1.0, sim_lrl=0.5)  # drift exactly 0.5
    assert not keep_pair(sims)
    assert keep_pair(sims, drift_max=0.5)
    assert keep_pair(ps(trans_q=0.85), translation_min=0.8)


def test_keep_mask() -> None:
    sims = [ps(), ps(trans_q=0.5), ps(sim_en=0.3, sim_lrl=0.9)]
    assert keep_mask(sims) == [True, False, False]


def test_sims_from_row() -> None:
    row = {"anchor": "x", **GOOD}
    assert sims_from_row(row) == ps()


def test_sims_from_row_missing_field_raises() -> None:
    with pytest.raises(ValueError, match="sim field"):
        sims_from_row({"sim_en": 0.9})


def test_filter_rows_precomputed() -> None:
    rows = [
        {"id": "keep", **GOOD},
        {"id": "drift", **GOOD, "sim_lrl": 0.5},
        {"id": "trans", **GOOD, "trans_p": 0.1},
    ]
    kept = filter_rows(rows)
    assert [r["id"] for r in kept] == ["keep"]


def test_filter_rows_with_external_sims() -> None:
    rows = [{"id": "a"}, {"id": "b"}]
    kept = filter_rows(rows, sims=[ps(), ps(trans_q=0.0)])
    assert [r["id"] for r in kept] == ["a"]


def test_filter_rows_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="sims"):
        filter_rows([{"id": "a"}], sims=[ps(), ps()])
