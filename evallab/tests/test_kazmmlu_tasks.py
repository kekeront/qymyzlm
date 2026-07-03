"""Bundled KazMMLU lm-eval task dir: utils behavior + YAML consistency (offline)."""

import importlib.util
from pathlib import Path

import kazeval
import yaml
from kazeval.run_kazmmlu import GROUP_TASK, MAX_FEWSHOT, TASKS_DIR

KAZMMLU_DIR = Path(kazeval.__file__).parent / "lm_eval_tasks" / "kazmmlu_kaz"

EXPECTED_SUBJECTS = {
    "biology": "Biology (High School in kaz)",
    "chemistry": "Chemistry (High School in kaz)",
    "geography": "Geography (High School in kaz)",
    "informatics": "Informatics (High School in kaz)",
    "kazakh_history": "Kazakh History (High School in kaz)",
    "kazakh_language": "Kazakh Language (High School in kaz)",
    "kazakh_literature": "Kazakh Literature (High School in kaz)",
    "law": "Law (High School in kaz)",
    "math": "Math (High School in kaz)",
    "physics": "Physics (High School in kaz)",
    "reading_literacy": "Reading Literacy (High School in kaz)",
    "world_history": "World History (High School in kaz)",
}


def load_utils():
    spec = importlib.util.spec_from_file_location("kazmmlu_kaz_utils", KAZMMLU_DIR / "utils.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIVE_OPTION_DOC = {
    "Question": "Қазақстанның астанасы қай қала?",
    "Option A": "Алматы",
    "Option B": "Астана",
    "Option C": "Шымкент",
    "Option D": "Тараз",
    "Option E": "Ақтау",
    "Answer Key": "B",
}

FOUR_OPTION_DOC = {  # "Option E" empty — the variable-option KazMMLU quirk
    "Question": "Қымыз неден жасалады?",
    "Option A": "Бие сүтінен",
    "Option B": "Судан",
    "Option C": "Шайдан",
    "Option D": "Уннан",
    "Option E": "",
    "Answer Key": "A",
}


def test_doc_to_text_format():
    utils = load_utils()
    text = utils.doc_to_text(FIVE_OPTION_DOC)
    assert text.startswith("Қазақстанның астанасы қай қала?\nA. Алматы\n")
    assert text.endswith("\nЖауап:")
    assert "E. Ақтау" in text


def test_variable_option_count_handled():
    utils = load_utils()
    assert utils.doc_to_choice(FIVE_OPTION_DOC) == ["A", "B", "C", "D", "E"]
    assert utils.doc_to_choice(FOUR_OPTION_DOC) == ["A", "B", "C", "D"]
    assert "E." not in utils.doc_to_text(FOUR_OPTION_DOC)


def test_doc_to_target_is_choice_index():
    utils = load_utils()
    assert utils.doc_to_target(FIVE_OPTION_DOC) == 1  # "B"
    assert utils.doc_to_target(FOUR_OPTION_DOC) == 0  # "A"


def test_tasks_dir_is_the_bundled_one():
    assert TASKS_DIR.is_dir()
    assert (TASKS_DIR / "kazmmlu_kaz").resolve() == KAZMMLU_DIR.resolve()


def test_group_yaml_lists_all_12_subjects():
    group = yaml.safe_load((KAZMMLU_DIR / "_kazmmlu_kaz.yaml").read_text(encoding="utf-8"))
    assert group["group"] == GROUP_TASK
    assert group["task"] == [f"kazmmlu_kaz_{slug}" for slug in sorted(EXPECTED_SUBJECTS)]
    assert group["aggregate_metric_list"] == [{"metric": "acc", "weight_by_size": True}]


def test_subject_yamls_pin_exact_hf_config_names():
    for slug, config_name in EXPECTED_SUBJECTS.items():
        payload = yaml.safe_load(
            (KAZMMLU_DIR / f"kazmmlu_kaz_{slug}.yaml").read_text(encoding="utf-8")
        )
        assert payload["task"] == f"kazmmlu_kaz_{slug}"
        assert payload["dataset_name"] == config_name
        assert payload["include"] == "_kazmmlu_kaz_template_yaml"


def test_template_pins_3_shot_from_dev():
    template = (KAZMMLU_DIR / "_kazmmlu_kaz_template_yaml").read_text(encoding="utf-8")
    assert "dataset_path: MBZUAI/KazMMLU" in template
    assert f"num_fewshot: {MAX_FEWSHOT}" in template
    assert "fewshot_split: dev" in template
    assert "output_type: multiple_choice" in template
