"""Hand-computed metric checks and determinism of the hard-negatives protocol."""

import math

import pytest
from kazeval.hardneg import (
    Bm25Index,
    bm25_scores,
    build_bm25_index,
    build_candidates,
    hits_at_k,
    mrr_at_k,
    rank_by_score,
    reciprocal_rank,
    tokenize,
)

# ---------------------------------------------------------------- MRR / hits


def test_mrr_hand_computed_exact_fraction():
    run = {
        "q1": ["d2", "d1", "d3"],  # first relevant at rank 2 -> 1/2
        "q2": ["x", "y", "d3"],  # first relevant at rank 3 -> 1/3
        "q3": ["a", "b"],  # no relevant in run -> 0
    }
    qrels = {"q1": {"d1"}, "q2": {"d3"}, "q3": {"z"}}
    assert mrr_at_k(run, qrels, k=10) == (1 / 2 + 1 / 3 + 0) / 3


def test_mrr_counts_missing_query_as_zero():
    run = {"q1": ["d1"]}
    qrels = {"q1": {"d1"}, "q2": {"d2"}}
    assert mrr_at_k(run, qrels, k=10) == (1.0 + 0.0) / 2


def test_mrr_respects_k_cutoff():
    run = {"q1": ["a", "b", "d1"]}
    qrels = {"q1": {"d1"}}
    assert mrr_at_k(run, qrels, k=2) == 0.0
    assert mrr_at_k(run, qrels, k=3) == 1 / 3


def test_mrr_rejects_bad_qrels():
    with pytest.raises(ValueError, match="empty"):
        mrr_at_k({}, {}, k=10)
    with pytest.raises(ValueError, match="q1"):
        mrr_at_k({"q1": ["d1"]}, {"q1": set()}, k=10)


def test_reciprocal_rank_first_relevant_only():
    assert reciprocal_rank(["a", "d1", "d2"], {"d1", "d2"}, k=10) == 1 / 2


def test_hits_at_k_hand_computed():
    run = {"q1": ["d1", "x"], "q2": ["x", "y"], "q3": ["a", "d3"]}
    qrels = {"q1": {"d1"}, "q2": {"d2"}, "q3": {"d3"}}
    assert hits_at_k(run, qrels, k=1) == 1 / 3
    assert hits_at_k(run, qrels, k=2) == 2 / 3


# ---------------------------------------------------------------------- BM25


def test_tokenize_kazakh_cyrillic_casefolds():
    assert tokenize("Қымыз — бие сүтінен!") == ["қымыз", "бие", "сүтінен"]


def test_bm25_single_term_score_matches_formula():
    docs = {"d1": "alma alma", "d2": "nan", "d3": "su"}
    index = build_bm25_index(docs)
    scores = bm25_scores(index, "alma")
    n_docs, df, tf, doc_len = 3, 1, 2, 2
    avg_len = 4 / 3
    idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
    norm = 1.5 * (1.0 - 0.75 + 0.75 * doc_len / avg_len)
    assert scores == {"d1": pytest.approx(idf * tf / (tf + norm))}


def test_bm25_index_rejects_empty_collection():
    with pytest.raises(ValueError, match="empty"):
        build_bm25_index({})


def test_bm25_index_is_frozen():
    index = build_bm25_index({"d1": "alma"})
    assert isinstance(index, Bm25Index)
    with pytest.raises(AttributeError):
        index.avg_len = 1.0  # type: ignore[misc]


# -------------------------------------------------- candidate construction


def _pool() -> dict[str, str]:
    # 30 docs; only a few mention the query terms -> BM25 shortfall forces
    # the seeded-sampling fill path.
    docs = {f"d{i:02d}": f"мәтін нөмірі {i} туралы жазба" for i in range(30)}
    docs["d00"] = "астана қазақстанның елордасы"
    docs["d01"] = "алматы қазақстанның ең ірі қаласы"
    docs["d02"] = "қымыз бие сүтінен жасалады"
    return docs


def test_build_candidates_deterministic_same_seed():
    index = build_bm25_index(_pool())
    kwargs = dict(query_id="q1", n_candidates=10, seed=13)
    first = build_candidates("қазақстанның елордасы қай қала", {"d00"}, index, **kwargs)
    second = build_candidates("қазақстанның елордасы қай қала", {"d00"}, index, **kwargs)
    assert first == second
    assert len(first) == 10
    assert "d00" in first
    assert len(set(first)) == 10


def test_build_candidates_seed_changes_random_fill():
    index = build_bm25_index(_pool())
    query = "қазақстанның елордасы"  # matches ~2 docs -> most negatives are sampled
    a = build_candidates(query, {"d00"}, index, query_id="q1", n_candidates=10, seed=13)
    b = build_candidates(query, {"d00"}, index, query_id="q1", n_candidates=10, seed=14)
    assert set(a) != set(b)


def test_build_candidates_validates_positives():
    index = build_bm25_index(_pool())
    with pytest.raises(ValueError, match="no positives"):
        build_candidates("сұрақ", set(), index, query_id="q1")
    with pytest.raises(ValueError, match="not in the candidate pool"):
        build_candidates("сұрақ", {"missing"}, index, query_id="q1")


def test_rank_by_score_ties_break_by_docid():
    ranking = rank_by_score(["b", "a", "c"], {"a": 1.0, "b": 1.0, "c": 2.0})
    assert ranking == ["c", "a", "b"]
