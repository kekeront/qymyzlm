"""Hard-negative mining — thin wrapper over sentence_transformers.util.mine_hard_negatives.

Our defaults (verified against sentence-transformers 5.6.0):
- range_min=1: skip rank 0, which is usually the positive itself;
- range_max=100: candidates come from the top-100 ranked passages;
- absolute_margin=0.0: require sim(anchor, negative) < sim(anchor, positive);
- sampling_strategy="top": hardest candidates within the rank window;
- e5 prefixes applied during mining via query_prompt/corpus_prompt (mining with the wrong
  prefixes produces negatives from a different embedding geometry than training sees).

Note: mine_hard_negatives deduplicates repeated anchors; output_format="triplet" yields one
row per mined negative with columns [anchor, positive, negative].
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from qymyz_embed.prefixes import PASSAGE_PREFIX, QUERY_PREFIX

if TYPE_CHECKING:
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer


def mine(
    pairs: Dataset,
    model: SentenceTransformer,
    *,
    corpus: list[str] | None = None,
    num_negatives: int = 2,
    range_min: int = 1,
    range_max: int = 100,
    absolute_margin: float = 0.0,
    sampling_strategy: Literal["top", "random"] = "top",
    output_format: Literal["triplet", "n-tuple", "labeled-pair", "labeled-list"] = "triplet",
    batch_size: int = 256,
    use_faiss: bool = False,
    texts_are_prefixed: bool = False,
) -> Dataset:
    """Mine hard negatives for an (anchor, positive) pairs dataset.

    pairs: columns [anchor, positive] — RAW text unless texts_are_prefixed=True (then the
    e5 prefixes are NOT re-applied; kazparc_pairs.py emits prefixed text by default).
    corpus: optional extra negative candidates beyond the positives pool.
    use_faiss=True is recommended for large corpora (e.g. the ~815k KazQAD passages).
    """
    from sentence_transformers.util import mine_hard_negatives

    return mine_hard_negatives(
        dataset=pairs,
        model=model,
        corpus=corpus,
        num_negatives=num_negatives,
        range_min=range_min,
        range_max=range_max,
        absolute_margin=absolute_margin,
        sampling_strategy=sampling_strategy,
        query_prompt=None if texts_are_prefixed else QUERY_PREFIX,
        corpus_prompt=None if texts_are_prefixed else PASSAGE_PREFIX,
        output_format=output_format,
        batch_size=batch_size,
        use_faiss=use_faiss,
    )
