"""Extract contrastive (anchor, positive) pairs from KazParC parallel sentences.

Source: issai/kazparc, config "kazparc_raw" (371,902 rows; columns id/kk/en/ru/tr/domain —
the only config with unambiguous text columns), pinned to a revision. The repo is
gated:auto — accept the conditions on https://huggingface.co/datasets/issai/kazparc with
the logged-in HF account before the first download. License note: the HF repo carries no
license tag; the GitHub badge claims CC BY 4.0 [UNVERIFIED on HF].

Each parallel sentence becomes a contrastive pair per direction: anchor = source-language
sentence (e5 "query: " prefix), positive = target-language sentence ("passage: " prefix).

Offline mode: --input FILE.jsonl reads local rows with the same schema (tests, smoke runs).

Usage:
    python -m qymyz_embed.data.kazparc_pairs --output pairs.jsonl --limit 1000
    python -m qymyz_embed.data.kazparc_pairs --output pairs_ds --format hf \
        --directions kk-ru,ru-kk
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

from qymyz_embed.prefixes import add_passage_prefix, add_query_prefix

KAZPARC_REPO = "issai/kazparc"
KAZPARC_REVISION = "41df65bd299ae0b5f2222d86f3db2f4fdd44e8e6"  # lastModified 2024-04-12
KAZPARC_CONFIG = "kazparc_raw"
LANG_COLUMNS = ("kk", "en", "ru", "tr")
DEFAULT_DIRECTIONS = "kk-ru,ru-kk,kk-en,en-kk"


def parse_directions(spec: str) -> tuple[tuple[str, str], ...]:
    """Parse "kk-ru,ru-kk" into (("kk", "ru"), ("ru", "kk"))."""
    directions: list[tuple[str, str]] = []
    for item in spec.split(","):
        item = item.strip()
        src, sep, tgt = item.partition("-")
        if not sep or src not in LANG_COLUMNS or tgt not in LANG_COLUMNS or src == tgt:
            raise ValueError(
                f"bad direction {item!r}: expected 'src-tgt' with distinct languages "
                f"from {LANG_COLUMNS}"
            )
        directions.append((src, tgt))
    if not directions:
        raise ValueError("no directions given")
    return tuple(directions)


def rows_to_pairs(
    rows: Iterable[dict[str, str | None]],
    directions: tuple[tuple[str, str], ...],
    *,
    prefixed: bool = True,
    dedupe: bool = True,
) -> Iterator[dict[str, str]]:
    """Turn kazparc_raw-schema rows into {"anchor": ..., "positive": ...} pairs.

    Rows with an empty/missing side are skipped for that direction, as are pairs whose two
    sides are identical (untranslated leakage). Deduplication is exact on the raw
    (anchor, positive) text pair.
    """
    seen: set[tuple[str, str]] = set()
    for row in rows:
        for src, tgt in directions:
            anchor = (row.get(src) or "").strip()
            positive = (row.get(tgt) or "").strip()
            if not anchor or not positive or anchor == positive:
                continue
            if dedupe:
                key = (anchor, positive)
                if key in seen:
                    continue
                seen.add(key)
            if prefixed:
                anchor = add_query_prefix(anchor)
                positive = add_passage_prefix(positive)
            yield {"anchor": anchor, "positive": positive}


def read_jsonl_rows(path: Path) -> Iterator[dict[str, str | None]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def stream_kazparc_rows(revision: str = KAZPARC_REVISION) -> Iterator[dict[str, str | None]]:
    """Stream kazparc_raw rows from the hub (no full download). Requires gate acceptance."""
    from datasets import load_dataset
    from huggingface_hub.errors import GatedRepoError

    try:
        dataset = load_dataset(
            KAZPARC_REPO, KAZPARC_CONFIG, revision=revision, split="train", streaming=True
        )
    except GatedRepoError as exc:
        raise RuntimeError(
            f"{KAZPARC_REPO} is gated (auto-approval): open "
            f"https://huggingface.co/datasets/{KAZPARC_REPO} with the logged-in HF account, "
            "click 'Agree and access', then retry."
        ) from exc
    yield from dataset


def write_jsonl(pairs: Iterable[dict[str, str]], path: Path) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_hf(pairs: Iterable[dict[str, str]], path: Path) -> int:
    from datasets import Dataset

    rows = list(pairs)
    if not rows:
        raise ValueError("no pairs to write — check --directions / --input")
    Dataset.from_list(rows).save_to_disk(str(path))
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0] if __doc__ else None)
    parser.add_argument("--output", type=Path, required=True, help="JSONL file or HF dataset dir")
    parser.add_argument("--format", choices=("jsonl", "hf"), default="jsonl")
    parser.add_argument(
        "--directions",
        default=DEFAULT_DIRECTIONS,
        help=f"comma-separated src-tgt pairs (default: {DEFAULT_DIRECTIONS})",
    )
    parser.add_argument("--limit", type=int, default=None, help="cap emitted pairs (smoke runs)")
    parser.add_argument(
        "--input", type=Path, default=None, help="local JSONL with kazparc_raw-schema rows"
    )
    parser.add_argument("--revision", default=KAZPARC_REVISION, help="pinned HF revision")
    parser.add_argument(
        "--no-prefix",
        action="store_true",
        help="emit raw text (default: e5 'query: '/'passage: ' prefixes applied)",
    )
    parser.add_argument("--no-dedupe", action="store_true", help="keep exact-duplicate pairs")
    args = parser.parse_args(argv)

    directions = parse_directions(args.directions)
    rows = read_jsonl_rows(args.input) if args.input else stream_kazparc_rows(args.revision)
    pairs = rows_to_pairs(rows, directions, prefixed=not args.no_prefix, dedupe=not args.no_dedupe)
    if args.limit is not None:
        pairs = itertools.islice(pairs, args.limit)

    writer = write_jsonl if args.format == "jsonl" else write_hf
    count = writer(pairs, args.output)
    print(f"wrote {count} pairs -> {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
