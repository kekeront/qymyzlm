"""kk-MTEB task classes instantiate offline: metadata valid, no load_data calls."""

import pytest
from kazeval.tasks import KazQADReranking, KazQADRetrieval
from kazeval.tasks.kazqad_retrieval import KAZQAD_REPO, KAZQAD_REVISION


@pytest.fixture(params=[KazQADRetrieval, KazQADReranking], ids=lambda c: c.__name__)
def task(request):
    return request.param()


def test_metadata_validates_and_is_filled(task):
    task.metadata._validate_metadata()  # BCP-47 check for "kaz-Cyrl" runs here
    assert task.metadata.is_filled()


def test_metadata_pins_gated_kazqad_revision(task):
    assert task.metadata.dataset["path"] == KAZQAD_REPO
    assert task.metadata.dataset["revision"] == KAZQAD_REVISION
    assert len(KAZQAD_REVISION) == 40  # exact HF commit hash


def test_metadata_language_and_license(task):
    assert task.metadata.eval_langs == ["kaz-Cyrl"]
    assert task.metadata.license == "cc-by-sa-4.0"
    assert task.metadata.eval_splits == ["test"]


def test_task_types_and_main_scores():
    retrieval = KazQADRetrieval()
    reranking = KazQADReranking()
    assert retrieval.metadata.type == "Retrieval"
    assert retrieval.metadata.main_score == "ndcg_at_10"
    # mteb 2.x reranking = AbsTaskRetrieval subclass with type="Reranking"
    assert reranking.metadata.type == "Reranking"
    assert reranking.metadata.main_score == "map_at_1000"


def test_instantiation_does_not_load_data(task):
    assert task.data_loaded is False
