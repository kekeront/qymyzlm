"""Offline tests for the Less-is-More training setup (no model downloads, no training)."""

import json
from pathlib import Path

import pytest
from datasets import Dataset
from qymyz_embed import train_lim
from qymyz_embed.prefixes import TRAIN_PROMPTS


def test_build_args_pins_the_protocol(tmp_path: Path) -> None:
    args = train_lim.build_args(tmp_path, prompts=TRAIN_PROMPTS, fp16=False)
    assert args.per_device_train_batch_size == 512  # effective batch == negative pool
    assert args.gradient_accumulation_steps == 1  # accumulation must NOT fake batch 512
    assert args.learning_rate == pytest.approx(7e-5)
    assert args.num_train_epochs == 5
    assert args.warmup_steps == pytest.approx(0.2)  # float == ratio (transformers v5)
    assert args.lr_scheduler_type == "linear"
    assert not args.bf16  # Turing
    assert args.prompts == TRAIN_PROMPTS
    assert args.seed == 42


def test_load_pairs_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    rows = [{"anchor": f"q{i}", "positive": f"p{i}"} for i in range(3)]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    dataset = train_lim.load_pairs(path)
    assert dataset.column_names == ["anchor", "positive"]
    assert len(dataset) == 3


def test_load_pairs_empty_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no rows"):
        train_lim.load_pairs(path)


def test_prepare_pairs_caps_with_seeded_shuffle() -> None:
    dataset = Dataset.from_list([{"anchor": f"q{i}", "positive": f"p{i}"} for i in range(20)])
    capped = train_lim.prepare_pairs(dataset, max_pairs=5, seed=42)
    assert len(capped) == 5
    again = train_lim.prepare_pairs(dataset, max_pairs=5, seed=42)
    assert capped["anchor"] == again["anchor"]  # deterministic
    assert len(train_lim.prepare_pairs(dataset, max_pairs=0)) == 20  # 0 disables the cap


def test_prepare_pairs_enforces_role_column_order() -> None:
    # column ORDER defines roles for (Cached)MNRL — anchor must be first
    dataset = Dataset.from_list(
        [
            {
                "positive": "p",
                "negative_10": "n10",
                "anchor": "a",
                "negative_2": "n2",
                "negative": "n",
                "id": "x",
            }
        ]
    )
    prepared = train_lim.prepare_pairs(dataset)
    assert prepared.column_names == ["anchor", "positive", "negative", "negative_2", "negative_10"]


def test_prepare_pairs_missing_columns_raises() -> None:
    dataset = Dataset.from_list([{"question": "q", "answer": "a"}])
    with pytest.raises(ValueError, match="anchor"):
        train_lim.prepare_pairs(dataset)


def test_default_checkpoint_root_is_gitignored_embed_dir() -> None:
    assert train_lim.DEFAULT_CHECKPOINT_ROOT.name == "checkpoints"
    assert train_lim.DEFAULT_CHECKPOINT_ROOT.parent.name == "embed"
    assert train_lim.DEFAULT_MAX_PAIRS == 10_000
