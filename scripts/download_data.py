"""Download Kazakh text data from HuggingFace Hub.

Reads the source list from cfg.data.sources (configs/data/default.yaml).
Each source is saved as a HuggingFace dataset to data/raw/<name>/.
Already-downloaded sources are skipped (resumable).

Uses non-streaming (Parquet cache) downloads — resumable at the file level.
Large sources take time but interruption only loses the current Parquet shard.

Gated sources (e.g. CulturaX) require HF auth:
    huggingface-cli login       # interactive
    # or: export HF_TOKEN=hf_...
"""

import logging
import os
import shutil
from pathlib import Path

import hydra
from datasets import load_dataset
from omegaconf import DictConfig

from kazllm.utils.logging import setup_logging

log = logging.getLogger(__name__)

_GATED_HELP = (
    "This is a gated dataset. To download it:\n"
    "  1. Accept the license at https://huggingface.co/datasets/{repo}\n"
    "  2. Run: huggingface-cli login   (or set HF_TOKEN env var)\n"
    "  3. Re-run this script — already-downloaded sources will be skipped."
)


def _download_source(src_cfg, raw_dir: Path) -> None:
    name = src_cfg.name
    out_dir = raw_dir / name

    if out_dir.exists() and (out_dir / "dataset_info.json").exists():
        log.info(f"[skip] {name} already downloaded")
        return

    is_gated = src_cfg.get("gated", False)
    token_file = Path("~/.cache/huggingface/token").expanduser()
    has_token = bool(os.environ.get("HF_TOKEN") or token_file.exists())
    if is_gated and not has_token:
        log.warning(f"[skip-gated] {name}: no HF token found.")
        log.warning(_GATED_HELP.format(repo=src_cfg.hf_repo))
        return

    # Clean up any partial download
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"[download] {name} from {src_cfg.hf_repo} ...")

    load_kwargs: dict = {
        "split": src_cfg.split,
        "streaming": False,  # cache-based: resumable at Parquet shard level
    }
    if src_cfg.get("lang"):
        load_kwargs["name"] = src_cfg.lang

    try:
        ds = load_dataset(src_cfg.hf_repo, **load_kwargs)

        # Normalise to single "text" column
        if src_cfg.text_col != "text":
            if src_cfg.text_col in ds.column_names:
                ds = ds.rename_column(src_cfg.text_col, "text")
        keep_cols = [c for c in ds.column_names if c == "text"]
        ds = ds.select_columns(keep_cols)

        log.info(f"[save] {name}: {len(ds):,} examples → {out_dir}")
        ds.save_to_disk(str(out_dir), num_proc=4)
        log.info(f"[done] {name}")
    except Exception as e:
        log.error(f"[error] {name}: {e}")
        shutil.rmtree(out_dir, ignore_errors=True)
        raise


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    raw_dir = Path(cfg.data.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    sources = cfg.data.sources
    log.info(f"Downloading {len(sources)} sources → {raw_dir}")

    failed = []
    for src in sources:
        try:
            _download_source(src, raw_dir)
        except Exception:
            failed.append(src.name)

    if failed:
        log.error(f"Failed sources (re-run to retry): {failed}")
        raise SystemExit(1)

    log.info("All sources downloaded.")


if __name__ == "__main__":
    main()
