"""CLI plumbing of kazeval.run_retrieval that must not regress offline."""

import pytest
from kazeval.run_retrieval import build_parser, parse_device


def test_parse_device_single():
    assert parse_device("cuda:0") == "cuda:0"


def test_parse_device_multi_kaggle_t4x2():
    assert parse_device("cuda:0,cuda:1") == ["cuda:0", "cuda:1"]
    assert parse_device(" cuda:0 , cuda:1 ") == ["cuda:0", "cuda:1"]


def test_parse_device_empty_rejected():
    with pytest.raises(ValueError):
        parse_device(" , ")


def test_parser_device_default_none():
    args = build_parser().parse_args(["--model", "m"])
    assert args.device is None
    args = build_parser().parse_args(["--model", "m", "--device", "cuda:0,cuda:1"])
    assert args.device == "cuda:0,cuda:1"
