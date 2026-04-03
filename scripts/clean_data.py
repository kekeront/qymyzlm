"""Apply Kazakh-aware quality filters and deduplication to raw data.

Pipeline stages:
  1. Text normalization (homoglyphs, unicode, URLs, whitespace)
  2. Paragraph-level filtering (remove non-Kazakh / too-short paragraphs)
  3. Kazakh language identification + quality scoring + domain scoring
  4. Paragraph-level boilerplate removal (two-pass)
  5. Exact deduplication (SHA-256, cross-source)
  6. Near deduplication (MinHash LSH, per-source)
"""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from kazllm.data.dedup import ExactDeduplicator, NearDeduplicator, ParagraphDeduplicator
from kazllm.data.filters import apply_all_filters, normalize_text
from kazllm.utils.logging import setup_logging

log = logging.getLogger(__name__)


def _get_filter_cfg(cfg: DictConfig) -> dict:
    """Extract filter parameters from Hydra config with defaults."""
    filters = cfg.data.get("filters", {})
    return {
        "min_chars": filters.get("min_chars", 50),
        "max_chars": filters.get("max_chars", 100_000),
        "min_kazakh_score": filters.get("min_kazakh_score", 0.35),
        "min_quality_score": filters.get("min_quality_score", 0.25),
        "min_domain_score": filters.get("min_domain_score", 0.15),
        "paragraph_filter": filters.get("paragraph_filter", True),
    }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    raw_dir = Path(cfg.data.raw_dir)
    deduped_dir = Path(cfg.data.deduped_dir)

    from datasets import load_from_disk

    filter_cfg = _get_filter_cfg(cfg)
    dedup_cfg = cfg.data.get("dedup", {})
    paragraph_dedup_enabled = dedup_cfg.get("paragraph_dedup", True)
    shingle_size = dedup_cfg.get("shingle_size", 3)

    exact_dedup = ExactDeduplicator()

    source_dirs = sorted(d for d in raw_dir.iterdir() if d.is_dir())
    log.info(f"Found {len(source_dirs)} sources in {raw_dir}")

    for source_dir in source_dirs:
        log.info(f"{'='*60}")
        log.info(f"Processing {source_dir.name}")
        ds = load_from_disk(str(source_dir))
        n_original = len(ds)
        log.info(f"  Loaded: {n_original:,} documents")

        # ----- Stage 1: Normalize text -----
        ds = ds.map(
            lambda ex: {"text": normalize_text(ex["text"])},
            num_proc=8,
            desc=f"Normalizing {source_dir.name}",
        )

        # ----- Stage 2+3: Kazakh-aware quality filter -----
        # (includes paragraph filtering, kazakh scoring, quality, domain)
        ds = ds.filter(
            lambda ex: apply_all_filters(ex["text"], **filter_cfg).passed,
            num_proc=8,
            desc=f"Filtering {source_dir.name}",
        )
        n_after_filter = len(ds)
        log.info(
            f"  After Kazakh quality filter: {n_after_filter:,} "
            f"({n_after_filter / max(n_original, 1) * 100:.1f}%)"
        )

        # ----- Stage 4: Paragraph-level boilerplate removal (two-pass) -----
        if paragraph_dedup_enabled and n_after_filter > 100:
            para_dedup = ParagraphDeduplicator(
                max_count=dedup_cfg.get("paragraph_max_count", 10),
            )
            # Pass 1: Count paragraph frequencies
            log.info("  Paragraph dedup pass 1: counting...")
            for ex in ds:
                para_dedup.count_paragraphs(ex["text"])
            n_boilerplate = para_dedup.num_boilerplate_paragraphs
            log.info(f"  Found {n_boilerplate:,} boilerplate paragraph types")

            if n_boilerplate > 0:
                # Pass 2: Remove boilerplate paragraphs
                ds = ds.map(
                    lambda ex: {"text": para_dedup.filter_boilerplate(ex["text"])},
                    desc=f"Removing boilerplate {source_dir.name}",
                )
                # Drop documents that became empty after boilerplate removal
                ds = ds.filter(lambda ex: len(ex["text"].strip()) >= filter_cfg["min_chars"])
                log.info(f"  After paragraph dedup: {len(ds):,} documents")

        # ----- Stage 5: Exact deduplication (cross-source) -----
        ds = ds.filter(lambda ex: not exact_dedup.is_duplicate(ex["text"]))
        n_after_exact = len(ds)
        log.info(f"  After exact dedup: {n_after_exact:,}")

        # ----- Stage 6: Near deduplication (per-source) -----
        source_near_dedup = NearDeduplicator(
            threshold=dedup_cfg.get("jaccard_threshold", 0.85),
            num_perm=dedup_cfg.get("minhash_num_perm", 128),
            shingle_size=shingle_size,
        )
        ds = ds.filter(lambda ex: not source_near_dedup.is_duplicate(ex["text"]))
        n_final = len(ds)
        log.info(f"  After near dedup: {n_final:,}")

        # ----- Summary -----
        kept_pct = n_final / max(n_original, 1) * 100
        log.info(f"  TOTAL: {n_original:,} -> {n_final:,} ({kept_pct:.1f}% kept)")

        # ----- Save -----
        out_path = deduped_dir / source_dir.name
        ds.save_to_disk(str(out_path))
        log.info(f"  Saved to {out_path}")

    log.info(f"{'='*60}")
    log.info("All sources cleaned.")


if __name__ == "__main__":
    main()
