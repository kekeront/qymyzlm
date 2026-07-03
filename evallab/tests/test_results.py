"""Result-record roundtrip and validation-error coverage."""

import json

import pytest
from kazeval.results import (
    ResultRecord,
    load_record,
    load_records,
    record_from_dict,
    record_path,
    save_record,
    validate_record,
)


def make_record(**overrides) -> ResultRecord:
    payload = dict(
        model="intfloat/multilingual-e5-large",
        revision="3d7cfbda",
        task="KazQAD-HardNeg",
        protocol="kazqad-hardneg-bm25-v1",
        split="test",
        metrics={"mrr": 0.5, "hits_at_1": 0.25},
        provenance="measured",
        source="kazeval.run_retrieval (mteb 2.16.3)",
        date="2026-07-03",
    )
    payload.update(overrides)
    return ResultRecord(**payload)


def test_roundtrip_save_load(tmp_path):
    record = make_record()
    path = save_record(record, tmp_path)
    assert path == record_path(record, tmp_path)
    assert path.name == "2026-07-03__KazQAD-HardNeg__intfloat-multilingual-e5-large.json"
    assert load_record(path) == record
    assert load_records(tmp_path) == [record]


def test_save_is_idempotent_per_key(tmp_path):
    save_record(make_record(metrics={"mrr": 0.1}), tmp_path)
    save_record(make_record(metrics={"mrr": 0.2}), tmp_path)  # same (date, task, model)
    records = load_records(tmp_path)
    assert len(records) == 1
    assert records[0].metrics == {"mrr": 0.2}


def test_load_records_sorted(tmp_path):
    save_record(make_record(task="ZTask", date="2026-01-01"), tmp_path)
    save_record(make_record(task="ATask", date="2026-02-01"), tmp_path)
    assert [r.task for r in load_records(tmp_path)] == ["ATask", "ZTask"]


def test_validate_rejects_bad_provenance():
    with pytest.raises(ValueError, match="provenance"):
        validate_record(make_record(provenance="estimated"))


def test_validate_rejects_bad_date():
    with pytest.raises(ValueError, match="date"):
        validate_record(make_record(date="2026-13-01"))
    with pytest.raises(ValueError, match="date"):
        validate_record(make_record(date="July 3, 2026"))


def test_validate_rejects_empty_or_nonfinite_metrics():
    with pytest.raises(ValueError, match="metrics"):
        validate_record(make_record(metrics={}))
    with pytest.raises(ValueError, match="finite"):
        validate_record(make_record(metrics={"mrr": float("nan")}))
    with pytest.raises(ValueError, match="finite"):
        validate_record(make_record(metrics={"mrr": float("inf")}))
    with pytest.raises(ValueError, match="finite"):
        validate_record(make_record(metrics={"mrr": True}))


def test_validate_rejects_empty_strings():
    with pytest.raises(ValueError, match="model"):
        validate_record(make_record(model="  "))
    with pytest.raises(ValueError, match="revision"):
        validate_record(make_record(revision=""))


def test_validate_allows_null_revision():
    validate_record(make_record(revision=None))


def test_from_dict_rejects_unknown_and_missing_keys():
    payload = json.loads(json.dumps(make_record().__dict__))
    payload["extra"] = 1
    with pytest.raises(ValueError, match="unknown=\\['extra'\\]"):
        record_from_dict(payload)
    del payload["extra"]
    del payload["split"]
    with pytest.raises(ValueError, match="missing=\\['split'\\]"):
        record_from_dict(payload)


def test_load_record_rejects_invalid_json(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_record(path)
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_record(path)


def test_load_records_requires_directory(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        load_records(tmp_path / "nope")
