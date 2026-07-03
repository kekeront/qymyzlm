"""Offline tests for the evaluate shim — fake kazeval modules, no evallab dependency."""

import sys
import types

import pytest
from qymyz_embed import evaluate


def test_forwards_to_kazeval_main(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    fake = types.ModuleType("kazeval")

    def fake_main(args: list[str]) -> int:
        calls.append(args)
        return 3

    fake.main = fake_main  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kazeval", fake)
    assert evaluate.main(["--task", "hardneg"]) == 3
    assert calls == [["--task", "hardneg"]]


def test_main_returning_none_maps_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.ModuleType("kazeval")
    fake.main = lambda args: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kazeval", fake)
    assert evaluate.main([]) == 0


def test_missing_kazeval_exits_with_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "kazeval", None)  # forces ImportError on import
    with pytest.raises(SystemExit, match="evallab"):
        evaluate.main([])


def test_kazeval_without_cli_exits_with_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.ModuleType("kazeval")  # no main, and run_module finds no kazeval.__main__
    monkeypatch.setitem(sys.modules, "kazeval", fake)
    with pytest.raises(SystemExit, match="kazeval.main"):
        evaluate.main([])
