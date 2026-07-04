"""KazQADReranking — the canonical Kazakh reranking protocol (mteb 2.x task).

mteb 2.x mechanics: a reranking task subclasses ``AbsTaskRetrieval`` with
``type="Reranking"`` and provides ``top_ranked`` (query id -> ordered candidate
docids); the model then reranks only those candidates (the old ``AbsTaskReranking``
is deprecated).

Candidate construction — canonical kk reranking protocol
(``kazqad-hardneg-bm25-v1``, shared with :mod:`kazeval.hardneg`):

1. Candidate pool = union of ALL judged passages (positive + negative) across the
   train/validation/test splits of ``issai/kazqad-retrieval`` @ pinned revision,
   deduplicated by docid. No full-corpus download.
2. For each eval query: candidates = all gold positives + top BM25 hard negatives
   from the pool (Lucene idf, k1=1.5, b=0.75, doc text ``f"{title} {text}"``,
   casefold + ``\\w+`` tokens) up to 100 candidates total; ties break by
   ``(-score, docid)``; any shortfall is filled by seeded sampling with
   ``random.Random(f"13:{query_id}")`` over the docid-sorted remaining pool.
3. ``top_ranked`` lists candidates in ``(-bm25_score, docid)`` order; qrels keep
   only the gold positives (score 1).

Fully deterministic: same dataset revision + same code => identical candidates.
The dataset is gated (auto-approval) — accept conditions on the HF page first.
"""

from typing import Any

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

from kazeval.hardneg import DEFAULT_SEED, N_CANDIDATES, build_bm25_index, build_candidates
from kazeval.tasks.kazqad_retrieval import KAZQAD_BIBTEX, KAZQAD_REPO, KAZQAD_REVISION


class KazQADReranking(AbsTaskRetrieval):
    """Rerank BM25 hard candidates for KazQAD test questions (kaz-Cyrl)."""

    metadata = TaskMetadata(
        name="KazQADReranking",
        description=(
            "Kazakh QA reranking over KazQAD: for each test question, rerank a "
            "deterministic 100-candidate pool (gold positives + BM25 hard negatives "
            "mined from the judged-passage pool of issai/kazqad-retrieval)."
        ),
        reference="https://arxiv.org/abs/2404.04487",
        dataset={"path": KAZQAD_REPO, "revision": KAZQAD_REVISION},
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["kaz-Cyrl"],
        main_score="map_at_1000",
        date=("2022-01-01", "2024-03-01"),
        domains=["Encyclopaedic", "Academic", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=KAZQAD_BIBTEX,
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        """Build the deterministic candidate pools from the gated HF dataset."""
        if self.data_loaded:
            return
        from datasets import Dataset, load_dataset

        queries_and_passages = load_dataset(
            KAZQAD_REPO, "queries-and-passages", revision=KAZQAD_REVISION
        )
        # Step 1: judged-passage pool over ALL splits, deduplicated by docid.
        pool: dict[str, dict[str, str]] = {}
        for split_rows in queries_and_passages.values():
            for row in split_rows:
                for passage in row["positive_passages"] + row["negative_passages"]:
                    pool.setdefault(
                        passage["docid"],
                        {
                            "id": passage["docid"],
                            "title": passage["title"],
                            "text": passage["text"],
                        },
                    )
        index = build_bm25_index(
            {doc_id: f"{doc['title']} {doc['text']}" for doc_id, doc in pool.items()}
        )
        self.dataset = {}
        for split in self.metadata.eval_splits:
            split_rows = queries_and_passages[split]
            relevant_docs: dict[str, dict[str, int]] = {}
            top_ranked: dict[str, list[str]] = {}
            used_doc_ids: set[str] = set()
            query_rows: list[dict[str, str]] = []
            for row in split_rows:
                positives = {p["docid"] for p in row["positive_passages"]}
                if not positives:
                    continue  # mteb drops positive-less queries anyway; skip early
                candidates = build_candidates(
                    row["query"],
                    positives,
                    index,
                    query_id=row["query_id"],
                    n_candidates=N_CANDIDATES,
                    seed=DEFAULT_SEED,
                )
                relevant_docs[row["query_id"]] = {doc_id: 1 for doc_id in positives}
                top_ranked[row["query_id"]] = candidates
                used_doc_ids.update(candidates)
                query_rows.append({"id": row["query_id"], "text": row["query"]})
            corpus = Dataset.from_list([pool[doc_id] for doc_id in sorted(used_doc_ids)])
            self.dataset.setdefault("default", {})[split] = RetrievalSplitData(
                corpus=corpus,
                queries=Dataset.from_list(query_rows),
                relevant_docs=relevant_docs,
                top_ranked=top_ranked,
            )
        self.data_loaded = True
