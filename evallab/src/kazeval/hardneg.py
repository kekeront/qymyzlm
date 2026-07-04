"""Reproducible KazQAD hard-negatives MRR protocol (the planka re-measurement path).

The reference number ("planka") — off-the-shelf ``intfloat/multilingual-e5-large``
at MRR 0.909 — comes from the ``Nurlykhan/kazembed-v5`` model card, whose protocol
is described in one line ("KazQAD test set with TF-IDF hard negatives, 100
candidates per query") and whose candidate pools were never published. This module
fixes OUR canonical, fully deterministic replacement protocol:

``kazqad-hardneg-bm25-v1``
    * Candidate pool: the union of all judged passages (``positive_passages`` +
      ``negative_passages``) across the train/validation/test splits of
      ``issai/kazqad-retrieval`` @ ``a3999685b5a1eed05b2453233875a14671cc6a4f``,
      deduplicated by ``docid``. No 825,309-passage corpus download is needed.
    * Document text for scoring: ``f"{title} {text}"``.
    * Tokenization: ``str.casefold()`` then Unicode ``\\w+`` (works for kk Cyrillic).
    * Scoring: BM25 with the Lucene idf variant
      ``idf = ln(1 + (N - df + 0.5) / (df + 0.5))``, ``k1 = 1.5``, ``b = 0.75``.
    * Per test query: candidates = all gold positives + the top BM25-ranked
      non-positive pool passages up to ``n_candidates = 100`` total. BM25 ties
      break by ``(-score, docid)``. If BM25 matches fewer negatives than needed,
      the shortfall is filled by ``random.Random(f"{seed}:{query_id}")`` sampling
      (seed 13) over the docid-sorted remaining pool — stable across runs and
      platforms (str seeding hashes via SHA-512).
    * Metric: MRR@10 over ALL test queries (a query missing from the run scores 0).

Everything here is a pure function: no downloads, no globals mutated, no clock.
"""

import math
import random
import re
from collections import Counter
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass

PROTOCOL = "kazqad-hardneg-bm25-v1"
HARDNEG_TASK = "KazQAD-HardNeg"
N_CANDIDATES = 100
DEFAULT_SEED = 13
BM25_K1 = 1.5
BM25_B = 0.75

_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Casefold + Unicode word tokenization (protocol-pinned)."""
    return _TOKEN_RE.findall(text.casefold())


@dataclass(frozen=True)
class Bm25Index:
    """Immutable BM25 index over a docid -> text mapping."""

    doc_ids: tuple[str, ...]
    doc_lens: tuple[int, ...]
    avg_len: float
    postings: dict[str, tuple[tuple[int, int], ...]]  # term -> ((doc_pos, tf), ...)
    k1: float
    b: float


def build_bm25_index(
    docs: Mapping[str, str], *, k1: float = BM25_K1, b: float = BM25_B
) -> Bm25Index:
    """Build a deterministic BM25 index (docs iterated in sorted-docid order)."""
    if not docs:
        raise ValueError("cannot build a BM25 index over an empty document collection")
    doc_ids = tuple(sorted(docs))
    doc_lens: list[int] = []
    postings: dict[str, list[tuple[int, int]]] = {}
    for pos, doc_id in enumerate(doc_ids):
        tokens = tokenize(docs[doc_id])
        doc_lens.append(len(tokens))
        for term, tf in sorted(Counter(tokens).items()):
            postings.setdefault(term, []).append((pos, tf))
    avg_len = sum(doc_lens) / len(doc_lens)
    frozen = {term: tuple(entries) for term, entries in postings.items()}
    return Bm25Index(
        doc_ids=doc_ids,
        doc_lens=tuple(doc_lens),
        avg_len=avg_len,
        postings=frozen,
        k1=k1,
        b=b,
    )


def bm25_scores(index: Bm25Index, query: str) -> dict[str, float]:
    """BM25 (Lucene idf) scores for every document matching >= 1 query term."""
    n_docs = len(index.doc_ids)
    scores: dict[int, float] = {}
    for term in dict.fromkeys(tokenize(query)):  # unique terms, stable order
        entries = index.postings.get(term)
        if not entries:
            continue
        df = len(entries)
        idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        for pos, tf in entries:
            norm = index.k1 * (1.0 - index.b + index.b * index.doc_lens[pos] / index.avg_len)
            scores[pos] = scores.get(pos, 0.0) + idf * tf / (tf + norm)
    return {index.doc_ids[pos]: score for pos, score in scores.items()}


def build_candidates(
    question: str,
    positive_ids: Set[str],
    index: Bm25Index,
    *,
    query_id: str,
    n_candidates: int = N_CANDIDATES,
    seed: int = DEFAULT_SEED,
) -> list[str]:
    """Deterministic candidate list for one query, per ``kazqad-hardneg-bm25-v1``.

    Returns all gold positives plus BM25 hard negatives, ``n_candidates`` docids
    total (fewer only if the pool itself is smaller), ordered by ``(-bm25, docid)``.
    """
    if not positive_ids:
        raise ValueError(f"query {query_id!r} has no positives; cannot build candidates")
    pool = set(index.doc_ids)
    missing = set(positive_ids) - pool
    if missing:
        raise ValueError(
            f"query {query_id!r}: positives not in the candidate pool: {sorted(missing)}"
        )
    scores = bm25_scores(index, question)
    n_negatives = max(n_candidates - len(positive_ids), 0)
    ranked_negatives = sorted(
        (doc_id for doc_id in scores if doc_id not in positive_ids),
        key=lambda doc_id: (-scores[doc_id], doc_id),
    )
    negatives = ranked_negatives[:n_negatives]
    if len(negatives) < n_negatives:
        remaining = sorted(pool - set(positive_ids) - set(negatives))
        rng = random.Random(f"{seed}:{query_id}")
        negatives += rng.sample(remaining, min(n_negatives - len(negatives), len(remaining)))
    candidates = list(positive_ids) + negatives
    return sorted(candidates, key=lambda doc_id: (-scores.get(doc_id, 0.0), doc_id))


def rank_by_score(candidates: Sequence[str], scores: Mapping[str, float]) -> list[str]:
    """Order candidates by a model's scores, ties broken by docid (deterministic)."""
    return sorted(candidates, key=lambda doc_id: (-scores[doc_id], doc_id))


def reciprocal_rank(ranking: Sequence[str], relevant: Set[str], k: int) -> float:
    """1/rank of the first relevant doc within the top-k, else 0.0."""
    for rank, doc_id in enumerate(ranking[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def mrr_at_k(
    run: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Set[str]],
    k: int = 10,
) -> float:
    """Mean reciprocal rank at ``k`` over ALL qrels queries (missing query -> 0)."""
    if not qrels:
        raise ValueError("qrels is empty; MRR is undefined")
    for query_id, relevant in qrels.items():
        if not relevant:
            raise ValueError(f"query {query_id!r} has an empty relevant set")
    total = sum(reciprocal_rank(run.get(qid, ()), relevant, k) for qid, relevant in qrels.items())
    return total / len(qrels)


def hits_at_k(
    run: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Set[str]],
    k: int,
) -> float:
    """Fraction of qrels queries with >= 1 relevant doc in the top-k."""
    if not qrels:
        raise ValueError("qrels is empty; hits@k is undefined")
    hits = sum(
        1 for qid, relevant in qrels.items() if any(d in relevant for d in run.get(qid, ())[:k])
    )
    return hits / len(qrels)
