"""KazQADRetrieval — full-corpus Kazakh open-domain QA retrieval (mteb 2.x task).

Protocol: 1,929 test questions against the full ~815k-passage Kazakh Wikipedia
corpus of ``issai/kazqad-retrieval`` (pinned revision). This is the setting of the
KazQAD paper's retrieval experiments (arXiv:2404.04487), so NDCG@10 / MRR here are
comparable to the paper's reported baselines (NDCG@10 0.389 / MRR 0.382).

NOTE: the HF dataset is gated (auto-approval) — accept the conditions on the
dataset page with the logged-in account before ``load_data`` can download it.
The dataset is NOT in the BEIR-style corpus/queries/qrels config layout, hence
the custom ``load_data``. For the upstream mteb PR the data must additionally be
pushed to the hub in that layout via ``task.push_dataset_to_hub(...)``.
"""

from typing import Any

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

KAZQAD_REPO = "issai/kazqad-retrieval"
KAZQAD_REVISION = "a3999685b5a1eed05b2453233875a14671cc6a4f"

KAZQAD_BIBTEX = r"""
@inproceedings{yeshpanov-etal-2024-kazqad,
  author = {Yeshpanov, Rustem and Efimov, Pavel and Boytsov, Leonid and
            Shalkarbayuli, Ardak and Braslavski, Pavel},
  booktitle = {Proceedings of the 2024 Joint International Conference on Computational
               Linguistics, Language Resources and Evaluation (LREC-COLING 2024)},
  pages = {9645--9656},
  title = {{KazQAD}: Kazakh Open-Domain Question Answering Dataset},
  year = {2024},
}
"""


class KazQADRetrieval(AbsTaskRetrieval):
    """Kazakh open-domain QA retrieval over the full KazQAD Wikipedia corpus."""

    metadata = TaskMetadata(
        name="KazQADRetrieval",
        description=(
            "Kazakh open-domain question answering retrieval: Natural-Questions-translated "
            "and Unified National Testing questions with human relevance judgements over a "
            "~815k-passage Kazakh Wikipedia corpus (KazQAD)."
        ),
        reference="https://arxiv.org/abs/2404.04487",
        dataset={"path": KAZQAD_REPO, "revision": KAZQAD_REVISION},
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["kaz-Cyrl"],
        main_score="ndcg_at_10",
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
        """Download the gated HF dataset and fill mteb's RetrievalSplitData structures."""
        if self.data_loaded:
            return
        from datasets import load_dataset

        queries_and_passages = load_dataset(
            KAZQAD_REPO, "queries-and-passages", revision=KAZQAD_REVISION
        )
        corpus = load_dataset(KAZQAD_REPO, "corpus", revision=KAZQAD_REVISION, split="train")
        corpus = corpus.rename_column("docid", "id")  # mteb corpus columns: id/title/text
        self.dataset = {}
        for split in self.metadata.eval_splits:
            split_rows = queries_and_passages[split]
            relevant_docs: dict[str, dict[str, int]] = {
                row["id"]: {p["docid"]: 1 for p in row["positive_passages"]} for row in split_rows
            }
            queries = split_rows.map(
                lambda row: {"id": row["id"], "text": row["question"]},
                remove_columns=split_rows.column_names,
            )
            self.dataset.setdefault("default", {})[split] = RetrievalSplitData(
                corpus=corpus,
                queries=queries,
                relevant_docs=relevant_docs,
                top_ranked=None,
            )
        self.data_loaded = True
