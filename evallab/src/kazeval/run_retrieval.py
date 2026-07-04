"""Run the kk-MTEB retrieval/reranking tasks and commit kazeval result records.

Wraps ``mteb.evaluate`` (mteb 2.16 API). In addition to mteb's native result
JSON (written by its ``ResultCache``), every run writes one kazeval
:class:`~kazeval.results.ResultRecord` per (task, split) into ``--output-dir``
(default ``evallab/results/``) — the committed files behind the leaderboard.

Usage (downloads model weights + the gated issai/kazqad-retrieval dataset —
accept the conditions on the HF dataset page first; never run in tests)::

    python -m kazeval.run_retrieval \
        --model intfloat/multilingual-e5-large \
        --tasks KazQADRetrieval KazQADReranking

Model loading: ``--model`` is first resolved against mteb's model registry
(``mteb.get_model``, which knows the correct prompts/prefixes for e5 etc.);
unknown names — e.g. our own fine-tuned checkpoints — fall back to a plain
``SentenceTransformer``, which ``mteb.evaluate`` wraps automatically.
"""

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path
from typing import Any

from kazeval.hardneg import PROTOCOL as HARDNEG_PROTOCOL
from kazeval.results import ResultRecord, save_record
from kazeval.tasks import KazQADReranking, KazQADRetrieval

logger = logging.getLogger(__name__)

TASK_CLASSES = {
    "KazQADRetrieval": KazQADRetrieval,
    "KazQADReranking": KazQADReranking,
}

TASK_PROTOCOLS = {
    "KazQADRetrieval": (
        "Full-corpus retrieval: KazQAD test questions vs the 825,309-passage Kazakh "
        "Wikipedia corpus (issai/kazqad-retrieval, pinned revision); comparable to "
        "arXiv:2404.04487 baselines (best pipeline NDCG@10 0.389 / MRR 0.382)."
    ),
    "KazQADReranking": (
        f"Reranking per {HARDNEG_PROTOCOL}: 100 deterministic candidates per test "
        "question (gold positives + BM25 hard negatives from the judged-passage pool)."
    ),
}

#: Metric keys copied from mteb's per-split scores into the kazeval record.
RECORD_METRICS = (
    "ndcg_at_10",
    "ndcg_at_100",
    "mrr_at_10",
    "map_at_1000",
    "recall_at_10",
    "recall_at_100",
)

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def load_model(name: str, revision: str | None, dtype: str = "float16") -> Any:
    """Resolve a model name via mteb's registry, else load a SentenceTransformer.

    Casts to fp16 by default: encoding the 825k-passage KazQAD corpus in fp32 on a
    T4 is compute-bound at ~4.3s/64-batch (measured), which overruns Kaggle's 12h
    session cap; fp16 is ~4x faster and shifts mE5-large retrieval scores by <1e-3
    (standard eval precision). Under Linux fork, the fp16 weights are inherited by
    the multi-GPU encode pool workers, so cuda:0,cuda:1 stays fp16.
    """
    import mteb

    try:
        model = mteb.get_model(name, revision=revision)
    except (KeyError, ValueError) as err:
        logger.warning(
            "%r not in mteb's model registry (%s); loading as SentenceTransformer", name, err
        )
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(name, revision=revision)
    if dtype == "float16":
        _cast_fp16(model)
    return model


def _cast_fp16(model: Any) -> None:
    """Cast the underlying transformer to fp16 in-place (T4 has no bf16).

    mteb wraps SentenceTransformer in a wrapper exposing ``.model``; try that first,
    then a raw SentenceTransformer.
    """
    for target in (getattr(model, "model", None), model):
        if target is not None and hasattr(target, "half"):
            try:
                target.half()
                logger.info("cast %s -> float16", type(target).__name__)
                return
            except Exception as exc:  # noqa: BLE001 - fp16 is best-effort, log and continue
                logger.warning("fp16 cast failed on %s: %s", type(target).__name__, exc)
    logger.warning("could not cast model to fp16; encoding stays fp32")


def parse_device(spec: str) -> str | list[str]:
    """'cuda:0' -> 'cuda:0'; 'cuda:0,cuda:1' -> ['cuda:0', 'cuda:1'] (multi-GPU pool)."""
    devices = [d.strip() for d in spec.split(",") if d.strip()]
    if not devices:
        raise ValueError(f"empty --device spec: {spec!r}")
    return devices if len(devices) > 1 else devices[0]


def extract_metrics(split_scores: list[dict[str, Any]]) -> dict[str, float]:
    """Curated metrics dict from one split's mteb subset-score entries."""
    entry = split_scores[0]  # monolingual tasks have the single subset "default"
    metrics = {key: float(entry[key]) for key in RECORD_METRICS if key in entry}
    if not metrics:
        raise ValueError(f"no known metrics in mteb scores entry: {sorted(entry)}")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kazeval.run_retrieval",
        description="Run kk-MTEB KazQAD tasks and write kazeval result records.",
    )
    parser.add_argument("--model", required=True, help="HF model id or local checkpoint path")
    parser.add_argument(
        "--model-revision", default=None, help="model revision to pin in the record"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=sorted(TASK_CLASSES),
        default=sorted(TASK_CLASSES),
        help="task names to run (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="where kazeval result records are written (default: evallab/results)",
    )
    parser.add_argument(
        "--mteb-cache",
        type=Path,
        default=None,
        help="mteb ResultCache path for native output (default: ~/.cache/mteb or $MTEB_CACHE)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "float32"],
        help="encode precision; float16 (default) is ~4x faster on T4 with a <1e-3 "
        "score delta for mE5-large. Use float32 only to reproduce fp32 exactly.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "encode device(s); comma-separated list spreads encoding over multiple "
            "GPUs via sentence-transformers' multi-process pool, e.g. 'cuda:0,cuda:1' "
            "on Kaggle's T4 x2 (default: single-device auto)"
        ),
    )
    parser.add_argument(
        "--overwrite-strategy",
        default="always",
        choices=("always", "only-missing"),
        help="mteb cache strategy; 'only-missing' silently reuses cached scores",
    )
    parser.add_argument(
        "--date",
        default=dt.date.today().isoformat(),
        help="ISO date stamped on the records (default: today)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)

    import mteb
    from mteb import ResultCache

    tasks = [TASK_CLASSES[name]() for name in args.tasks]
    model = load_model(args.model, args.model_revision, args.dtype)
    cache = ResultCache(cache_path=args.mteb_cache) if args.mteb_cache else ResultCache()

    encode_kwargs: dict[str, Any] = {"batch_size": args.batch_size}
    if args.device:
        encode_kwargs["device"] = parse_device(args.device)

    results = mteb.evaluate(
        model,
        tasks,
        cache=cache,
        encode_kwargs=encode_kwargs,
        overwrite_strategy=args.overwrite_strategy,
    )

    revision = args.model_revision or results.model_revision
    if revision in ("no_revision_available", ""):
        revision = None
    written: list[Path] = []
    for task_result in results.task_results:
        for split, split_scores in task_result.scores.items():
            record = ResultRecord(
                model=results.model_name,
                revision=revision,
                task=task_result.task_name,
                protocol=TASK_PROTOCOLS[task_result.task_name],
                split=split,
                metrics=extract_metrics(split_scores),
                provenance="measured",
                source=f"kazeval.run_retrieval (mteb {mteb.__version__})",
                date=args.date,
            )
            path = save_record(record, args.output_dir)
            written.append(path)
            print(f"{task_result.task_name} [{split}] main_score={task_result.get_score():.4f}")
            print(f"  record: {path}")
    if not written:
        print("no task results produced — nothing written", file=sys.stderr)
        return 1
    print(f"native mteb output under: {cache.cache_path}")
    print("now regenerate the leaderboard: python -m kazeval.leaderboard")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
