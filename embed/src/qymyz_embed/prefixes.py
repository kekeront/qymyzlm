"""Single source of truth for the e5 "query: " / "passage: " prefixes.

The intfloat/multilingual-e5-* model cards require: "Each input text should start with
'query: ' or 'passage: ', even for non-English texts." Yet the mE5 hub repos ship NO
prompts in their config (config_sentence_transformers.json is absent), so
sentence-transformers 5.6 synthesizes EMPTY defaults — encode_query() and
encode(prompt_name="query") silently apply NOTHING for stock mE5.

Every prefix in this package (training, mining, filtering, inference) flows through this
module so train and eval can never drift apart (embed/CLAUDE.md hard constraint).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

# Inference prompts: register on the model (see register_e5_prompts) so encode_query() /
# encode(prompt_name="query") apply the real prefixes. Persists through model.save().
E5_PROMPTS: dict[str, str] = {"query": QUERY_PREFIX, "passage": PASSAGE_PREFIX}

# Training column -> prompt mapping for SentenceTransformerTrainingArguments(prompts=...),
# for the standard (anchor, positive[, negative]) contrastive layout.
TRAIN_PROMPTS: dict[str, str] = {
    "anchor": QUERY_PREFIX,
    "positive": PASSAGE_PREFIX,
    "negative": PASSAGE_PREFIX,
}


def has_query_prefix(text: str) -> bool:
    return text.startswith(QUERY_PREFIX)


def has_passage_prefix(text: str) -> bool:
    return text.startswith(PASSAGE_PREFIX)


def add_query_prefix(text: str) -> str:
    """Prefix with "query: ". Idempotent: already-prefixed text is returned unchanged."""
    return text if has_query_prefix(text) else QUERY_PREFIX + text


def add_passage_prefix(text: str) -> str:
    """Prefix with "passage: ". Idempotent: already-prefixed text is returned unchanged."""
    return text if has_passage_prefix(text) else PASSAGE_PREFIX + text


def strip_prefix(text: str) -> str:
    """Remove a leading e5 prefix (either kind) if present."""
    for prefix in (QUERY_PREFIX, PASSAGE_PREFIX):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def training_prompts_for_columns(columns: Sequence[str]) -> dict[str, str]:
    """Column->prompt mapping for a contrastive dataset with RAW (unprefixed) text.

    anchor -> "query: "; positive and negative/negative_1..n -> "passage: ". Extra columns
    (id, domain, ...) get no prompt. Raises if the required contrastive columns are absent.
    """
    prompts: dict[str, str] = {}
    for col in columns:
        if col == "anchor":
            prompts[col] = QUERY_PREFIX
        elif col == "positive" or col.startswith("negative"):
            prompts[col] = PASSAGE_PREFIX
    if "anchor" not in prompts or "positive" not in prompts:
        raise ValueError(f"expected 'anchor' and 'positive' columns, got {list(columns)}")
    return prompts


def register_e5_prompts(model: SentenceTransformer) -> None:
    """Merge the e5 prompts into model.prompts (assignment merges with ST 5.6 defaults).

    Without this, stock mE5 silently encodes with EMPTY prompts via encode_query() /
    prompt_name="query". The registered prompts persist through model.save().
    """
    model.prompts = {**(getattr(model, "prompts", None) or {}), **E5_PROMPTS}
