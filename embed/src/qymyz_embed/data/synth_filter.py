"""Less-is-More synthetic-pair filtering (arXiv 2603.22290).

Keep a (query, passage) pair iff BOTH:
  1. semantic-drift rule:  |sim(q_en, p_en) - sim(q_kk, p_kk)| <= 0.05  (boundary KEEPS)
  2. translation quality:  sim(q_en, q_kk) > 0.85 AND sim(p_en, p_kk) > 0.85 (boundary DROPS)

All similarities are cosine, measured with multilingual-e5-base and the e5 prefixes
(queries: "query: ", passages: "passage: ") — the paper's own gate.

Pure filter logic (keep_pair / keep_mask / filter_rows) is separated from encoding
(compute_sims) so tests run offline on stub sims.

CLI: rows are JSONL. Precomputed path expects float fields sim_en/sim_lrl/trans_q/trans_p;
with --model, fields q_en/p_en/q_kk/p_kk are encoded to compute the sims first.

    python -m qymyz_embed.data.synth_filter --input scored.jsonl --output kept.jsonl
    python -m qymyz_embed.data.synth_filter --input raw.jsonl --output kept.jsonl \
        --model intfloat/multilingual-e5-base
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from qymyz_embed.prefixes import PASSAGE_PREFIX, QUERY_PREFIX

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

DRIFT_MAX = 0.05  # |sim(q_en,p_en) - sim(q_lrl,p_lrl)| above this => drop
TRANSLATION_MIN = 0.85  # sim(x_en, x_lrl) at or below this => drop
SIM_FIELDS = ("sim_en", "sim_lrl", "trans_q", "trans_p")
TEXT_FIELDS = ("q_en", "p_en", "q_kk", "p_kk")


@dataclass(frozen=True, slots=True)
class PairSims:
    """The four cosine similarities the Less-is-More gate needs for one pair."""

    sim_en: float  # sim(q_en, p_en)
    sim_lrl: float  # sim(q_kk, p_kk)
    trans_q: float  # sim(q_en, q_kk)
    trans_p: float  # sim(p_en, p_kk)


def keep_pair(
    sims: PairSims,
    *,
    drift_max: float = DRIFT_MAX,
    translation_min: float = TRANSLATION_MIN,
) -> bool:
    """True iff the pair passes both Less-is-More gates (see module docstring)."""
    return (
        abs(sims.sim_en - sims.sim_lrl) <= drift_max
        and sims.trans_q > translation_min
        and sims.trans_p > translation_min
    )


def keep_mask(
    sims: Iterable[PairSims],
    *,
    drift_max: float = DRIFT_MAX,
    translation_min: float = TRANSLATION_MIN,
) -> list[bool]:
    return [keep_pair(s, drift_max=drift_max, translation_min=translation_min) for s in sims]


def sims_from_row(row: Mapping[str, object]) -> PairSims:
    """Read precomputed similarities from a row's sim_en/sim_lrl/trans_q/trans_p fields."""
    try:
        values = [float(row[field]) for field in SIM_FIELDS]  # type: ignore[arg-type]
    except KeyError as exc:
        raise ValueError(
            f"row is missing precomputed sim field {exc}; expected {SIM_FIELDS}"
        ) from exc
    return PairSims(*values)


def filter_rows(
    rows: Sequence[Mapping[str, object]],
    sims: Sequence[PairSims] | None = None,
    *,
    drift_max: float = DRIFT_MAX,
    translation_min: float = TRANSLATION_MIN,
) -> list[Mapping[str, object]]:
    """Return the rows that pass the gate. sims=None reads them from the rows' sim fields."""
    if sims is None:
        sims = [sims_from_row(row) for row in rows]
    if len(sims) != len(rows):
        raise ValueError(f"got {len(rows)} rows but {len(sims)} sims")
    return [
        row
        for row, s in zip(rows, sims, strict=True)
        if keep_pair(s, drift_max=drift_max, translation_min=translation_min)
    ]


def compute_sims(
    q_en: Sequence[str],
    p_en: Sequence[str],
    q_kk: Sequence[str],
    p_kk: Sequence[str],
    model: SentenceTransformer,
    *,
    batch_size: int = 128,
) -> list[PairSims]:
    """Encode all four text columns with e5 prefixes and compute the gate similarities.

    Texts must be RAW (unprefixed) — prefixes are applied here via prompt=.
    """
    if not len(q_en) == len(p_en) == len(q_kk) == len(p_kk):
        raise ValueError("all four text columns must have equal length")

    def enc(texts: Sequence[str], prompt: str):  # noqa: ANN202 — torch.Tensor, kept lazy
        return model.encode(
            list(texts), prompt=prompt, convert_to_tensor=True, batch_size=batch_size
        )

    eq_en, eq_kk = enc(q_en, QUERY_PREFIX), enc(q_kk, QUERY_PREFIX)
    ep_en, ep_kk = enc(p_en, PASSAGE_PREFIX), enc(p_kk, PASSAGE_PREFIX)
    sim_en = model.similarity(eq_en, ep_en).diagonal()
    sim_lrl = model.similarity(eq_kk, ep_kk).diagonal()
    trans_q = model.similarity(eq_en, eq_kk).diagonal()
    trans_p = model.similarity(ep_en, ep_kk).diagonal()
    return [
        PairSims(float(a), float(b), float(c), float(d))
        for a, b, c, d in zip(sim_en, sim_lrl, trans_q, trans_p, strict=True)
    ]


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Less-is-More pair filter (arXiv 2603.22290)")
    parser.add_argument("--input", type=Path, required=True, help="JSONL rows to filter")
    parser.add_argument("--output", type=Path, required=True, help="JSONL of kept rows")
    parser.add_argument("--drift-max", type=float, default=DRIFT_MAX)
    parser.add_argument("--translation-min", type=float, default=TRANSLATION_MIN)
    parser.add_argument(
        "--model",
        default=None,
        help=f"encoder to compute sims from {TEXT_FIELDS} fields "
        "(default: read precomputed sim fields)",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args(argv)

    rows = _read_jsonl(args.input)
    sims: list[PairSims] | None = None
    if args.model is not None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(args.model)
        columns = {field: [str(row[field]) for row in rows] for field in TEXT_FIELDS}
        sims = compute_sims(
            columns["q_en"],
            columns["p_en"],
            columns["q_kk"],
            columns["p_kk"],
            model,
            batch_size=args.batch_size,
        )

    kept = filter_rows(rows, sims, drift_max=args.drift_max, translation_min=args.translation_min)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for row in kept:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"kept {len(kept)}/{len(rows)} pairs -> {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
