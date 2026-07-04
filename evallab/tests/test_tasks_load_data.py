"""load_data of both kk-MTEB tasks against the REAL issai/kazqad-retrieval schema.

Regression: the queries-and-passages rows carry ``query_id``/``query`` (verified
against the pinned revision via streaming + HF datasets-server, 2026-07-04), NOT
``id``/``question`` — the offline metadata tests never caught that, so this file
fakes ``datasets.load_dataset`` with the exact production field names and runs
``load_data`` end to end.
"""

import pytest
from datasets import Dataset
from kazeval.tasks import KazQADReranking, KazQADRetrieval

PASSAGES = {
    "d1": {"docid": "d1", "title": "Астана", "text": "Астана — Қазақстанның астанасы."},
    "d2": {"docid": "d2", "title": "Алматы", "text": "Алматы — ең ірі қала."},
    "d3": {"docid": "d3", "title": "Каспий", "text": "Каспий теңізі — әлемдегі ең үлкен көл."},
    "d4": {"docid": "d4", "title": "Абай", "text": "Абай Құнанбайұлы — қазақ ақыны."},
}

# Real schema: query_id / query / positive_passages / negative_passages.
QP_ROWS = {
    "train": [
        {
            "query_id": "q-train-1",
            "query": "Абай кім болған?",
            "positive_passages": [PASSAGES["d4"]],
            "negative_passages": [PASSAGES["d3"]],
        }
    ],
    "validation": [
        {
            "query_id": "q-val-1",
            "query": "Ең ірі қала қайсы?",
            "positive_passages": [PASSAGES["d2"]],
            "negative_passages": [],
        }
    ],
    "test": [
        {
            "query_id": "q-test-1",
            "query": "Қазақстанның астанасы қай қала?",
            "positive_passages": [PASSAGES["d1"]],
            "negative_passages": [PASSAGES["d2"], PASSAGES["d3"]],
        },
        {
            "query_id": "q-test-2",
            "query": "Каспий теңізі деген не?",
            "positive_passages": [PASSAGES["d3"]],
            "negative_passages": [PASSAGES["d4"]],
        },
    ],
}


@pytest.fixture()
def fake_load_dataset(monkeypatch):
    def load(path, name, revision=None, split=None):
        assert path == "issai/kazqad-retrieval"
        assert revision is not None
        if name == "queries-and-passages":
            assert split is None
            return {s: Dataset.from_list(rows) for s, rows in QP_ROWS.items()}
        if name == "corpus":
            assert split == "train"
            return Dataset.from_list(list(PASSAGES.values()))
        raise AssertionError(f"unexpected config {name!r}")

    monkeypatch.setattr("datasets.load_dataset", load)


def test_retrieval_load_data_real_schema(fake_load_dataset):
    task = KazQADRetrieval()
    task.load_data()
    split = task.dataset["default"]["test"]
    assert set(split["queries"].column_names) == {"id", "text"}
    assert sorted(split["queries"]["id"]) == ["q-test-1", "q-test-2"]
    assert split["relevant_docs"] == {"q-test-1": {"d1": 1}, "q-test-2": {"d3": 1}}
    assert set(split["corpus"].column_names) == {"id", "title", "text"}
    assert sorted(split["corpus"]["id"]) == ["d1", "d2", "d3", "d4"]


def test_reranking_load_data_real_schema(fake_load_dataset):
    task = KazQADReranking()
    task.load_data()
    split = task.dataset["default"]["test"]
    assert sorted(split["queries"]["id"]) == ["q-test-1", "q-test-2"]
    assert split["relevant_docs"] == {"q-test-1": {"d1": 1}, "q-test-2": {"d3": 1}}
    for query_id, candidates in split["top_ranked"].items():
        assert set(split["relevant_docs"][query_id]) <= set(candidates)
        assert len(candidates) == len(set(candidates))
    # candidate pool is judged passages only (train/val/test union), never the corpus
    assert set(split["corpus"]["id"]) <= set(PASSAGES)
