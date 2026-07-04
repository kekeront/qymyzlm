"""Offline tests for the e5 prefix single-source-of-truth module."""

from types import SimpleNamespace

import pytest
from qymyz_embed import prefixes


def test_constants() -> None:
    assert prefixes.QUERY_PREFIX == "query: "
    assert prefixes.PASSAGE_PREFIX == "passage: "
    # "document" is mteb's only valid doc-side prompt key (and ST 5.6 encode_document's
    # first candidate); "passage" stays for direct prompt_name="passage" callers.
    assert prefixes.E5_PROMPTS == {
        "query": "query: ",
        "document": "passage: ",
        "passage": "passage: ",
    }
    assert prefixes.TRAIN_PROMPTS == {
        "anchor": "query: ",
        "positive": "passage: ",
        "negative": "passage: ",
    }


def test_add_prefixes() -> None:
    assert prefixes.add_query_prefix("Астана қайда?") == "query: Астана қайда?"
    assert prefixes.add_passage_prefix("Астана — елорда.") == "passage: Астана — елорда."


def test_add_prefixes_idempotent() -> None:
    once = prefixes.add_query_prefix("мәтін")
    assert prefixes.add_query_prefix(once) == once
    once_p = prefixes.add_passage_prefix("мәтін")
    assert prefixes.add_passage_prefix(once_p) == once_p


def test_has_prefix() -> None:
    assert prefixes.has_query_prefix("query: x")
    assert not prefixes.has_query_prefix("passage: x")
    assert prefixes.has_passage_prefix("passage: x")
    assert not prefixes.has_passage_prefix("x")


def test_strip_prefix() -> None:
    assert prefixes.strip_prefix("query: мәтін") == "мәтін"
    assert prefixes.strip_prefix("passage: мәтін") == "мәтін"
    assert prefixes.strip_prefix("мәтін") == "мәтін"


def test_training_prompts_for_columns() -> None:
    prompts = prefixes.training_prompts_for_columns(
        ["anchor", "positive", "negative", "negative_2", "negative_10", "id"]
    )
    assert prompts == {
        "anchor": "query: ",
        "positive": "passage: ",
        "negative": "passage: ",
        "negative_2": "passage: ",
        "negative_10": "passage: ",
    }


def test_training_prompts_requires_contrastive_columns() -> None:
    with pytest.raises(ValueError, match="anchor"):
        prefixes.training_prompts_for_columns(["question", "answer"])


def test_register_e5_prompts_merges() -> None:
    # ST 5.6 synthesizes EMPTY query/document defaults for stock mE5 — registration must
    # OVERWRITE those (they are the bug), while unrelated keys are preserved.
    model = SimpleNamespace(prompts={"query": "", "document": "", "custom": "keep me"})
    prefixes.register_e5_prompts(model)  # type: ignore[arg-type]
    assert model.prompts["query"] == "query: "
    assert model.prompts["document"] == "passage: "
    assert model.prompts["passage"] == "passage: "
    assert model.prompts["custom"] == "keep me"  # unrelated entries preserved


def test_register_e5_prompts_handles_missing_prompts_attr() -> None:
    model = SimpleNamespace(prompts=None)
    prefixes.register_e5_prompts(model)  # type: ignore[arg-type]
    assert model.prompts == prefixes.E5_PROMPTS
