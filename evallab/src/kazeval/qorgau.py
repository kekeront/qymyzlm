"""Qorgau kk-ru safety benchmark — eval-only loaders and protocol notes.

LICENSE WARNING — EVAL-ONLY. The Qorgau dataset (arXiv:2502.13640) declares NO
license anywhere: the GitHub repo metadata reports ``license: None``, there is no
LICENSE file in the tree, and the dataset card has no license section (the arXiv
CC BY-NC-SA badge covers the paper text only). Default copyright therefore
applies: use the data for internal evaluation ONLY — never train on it, never
redistribute it, never re-host it on the HF hub.

Data lives only in the GitHub repo (no HF dataset exists as of 2026-07):

    git clone https://github.com/mbzuai-nlp/qorgau-kaz-ru-safety

- ``ru_kaz_data/ru_kz_question_only.xlsx`` — sheets ``Kazakh`` (3,786 rows) and
  ``Russian`` (4,383 rows); columns: id, risk_area, risk_area_specified,
  types_of_harm, specific_harms, question_type, question.
- ``ru_kaz_data/ru_kz_cs_questions.csv`` — 500 code-switched rows; columns:
  index, original_index, risk_area, types_of_harm, specific_harms,
  question_type, ru_version, kz_version, code_swithced_version (sic — the typo
  is in the shipped file and is preserved here).

The xlsx loader below is pure stdlib (zipfile + ElementTree) because openpyxl is
not part of the lab venv.

Evaluation protocol (paper + repo ``notebooks/evaluate_binary_safety.py``): an
LLM judge (GPT-4o in the paper) receives the model response and a risk-area
specific rubric of 3-5 Yes/No questions (question 1 is always refusal
detection); the response counts as safe when the number of rubric mismatches is
below the rubric's ``safe_threshold`` (or the model refused). Judge-vs-human
agreement in the paper: 90.4% (Kazakh) / 90.9% (Russian). The judge step is NOT
implemented here yet — this module only fixes the data loading and the taxonomy
so that a future runner scores against a stable interface.
"""

import csv
import zipfile
from dataclasses import dataclass
from pathlib import Path

try:  # hardened XML parsing (XXE / entity-expansion safe)
    from defusedxml import ElementTree as ET  # noqa: N817 — conventional alias
except ImportError:  # pragma: no cover — fallback for envs without defusedxml
    from xml.etree import ElementTree as ET  # type: ignore[no-redef]

TASK = "Qorgau-kk"
QORGAU_REPO = "https://github.com/mbzuai-nlp/qorgau-kaz-ru-safety"
QUESTIONS_XLSX = "ru_kaz_data/ru_kz_question_only.xlsx"
CODE_SWITCHED_CSV = "ru_kaz_data/ru_kz_cs_questions.csv"
# Preserved verbatim from the shipped CSV header (upstream typo).
CODE_SWITCHED_COLUMN = "code_swithced_version"

#: The 6 high-level risk areas (repo notebooks/evaluate_binary_safety.py risk2type).
RISK_AREAS: dict[int, str] = {
    0: "Misinformation Harms",
    1: "Human-Chatbot Interaction Harms",
    2: "Malicious Uses",
    3: "Discrimination, Exclusion, Toxicity, Hateful, Offensive",
    4: "Information Hazards",
    5: "Sensitive in Kazakhstan/Russia",
}

PROTOCOL = (
    "qorgau-binary-safety-v0 (stub): LLM judge with per-risk-area Yes/No rubric "
    "(question 1 = refusal detection), safe iff refusal or mismatches < safe_threshold; "
    "per arXiv:2502.13640 + repo notebooks/evaluate_binary_safety.py. Judge not wired yet."
)

_MAIN_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL_ATTR = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


@dataclass(frozen=True)
class QorgauQuestion:
    """One Qorgau probe question (kk or ru sheet of the questions xlsx)."""

    id: str
    risk_area: str
    risk_area_specified: str
    types_of_harm: str
    specific_harms: str
    question_type: str
    question: str


@dataclass(frozen=True)
class CodeSwitchedQuestion:
    """One code-switched Qorgau probe (ru / kk / code-switched variants)."""

    index: str
    original_index: str
    risk_area: str
    types_of_harm: str
    specific_harms: str
    question_type: str
    ru_version: str
    kz_version: str
    code_switched_version: str


def _column_index(cell_ref: str) -> int:
    """0-based column index from an A1-style cell reference (``"C7"`` -> 2)."""
    index = 0
    for char in cell_ref:
        if not char.isalpha():
            break
        index = index * 26 + (ord(char.upper()) - ord("A") + 1)
    if index == 0:
        raise ValueError(f"bad cell reference: {cell_ref!r}")
    return index - 1


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for si in root.iter(_MAIN_NS + "si"):
        strings.append("".join(t.text or "" for t in si.iter(_MAIN_NS + "t")))
    return strings


def _sheet_member(archive: zipfile.ZipFile, sheet_name: str) -> str:
    """Zip member path of a worksheet, resolved via workbook relationships."""
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rel_id: str | None = None
    names: list[str] = []
    for sheet in workbook.iter(_MAIN_NS + "sheet"):
        names.append(sheet.get("name", ""))
        if sheet.get("name") == sheet_name:
            rel_id = sheet.get(_REL_ATTR)
    if rel_id is None:
        raise ValueError(f"sheet {sheet_name!r} not found; workbook sheets: {names}")
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    for rel in rels:
        if rel.get("Id") == rel_id:
            target = rel.get("Target", "")
            return target if target.startswith("xl/") else f"xl/{target}"
    raise ValueError(f"no relationship {rel_id!r} for sheet {sheet_name!r}")


def _column_letter(index: int) -> str:
    """0-based column index -> A1-style column letters (2 -> ``"C"``)."""
    letters = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return letters


def read_xlsx_sheet(path: Path, sheet_name: str) -> list[dict[str, str]]:
    """Read one worksheet as a list of header-keyed string dicts (pure stdlib).

    Handles shared strings, inline strings, plain numeric cells and empty cells.
    Row 1 is the header; fully empty rows are skipped. A blank/whitespace header
    cell gets a positional key ``col_<LETTER>`` (the upstream Kazakh sheet ships
    a single space as its id header — verified on the real file).
    """
    with zipfile.ZipFile(path) as archive:
        strings = _shared_strings(archive)
        sheet_root = ET.fromstring(archive.read(_sheet_member(archive, sheet_name)))
        raw_rows: list[dict[int, str]] = []
        for row in sheet_root.iter(_MAIN_NS + "row"):
            cells: dict[int, str] = {}
            for cell in row.iter(_MAIN_NS + "c"):
                ref = cell.get("r")
                if ref is None:
                    continue
                cell_type = cell.get("t", "n")
                if cell_type == "inlineStr":
                    value = "".join(t.text or "" for t in cell.iter(_MAIN_NS + "t"))
                else:
                    v_node = cell.find(_MAIN_NS + "v")
                    if v_node is None or v_node.text is None:
                        continue
                    value = strings[int(v_node.text)] if cell_type == "s" else v_node.text
                cells[_column_index(ref)] = value
            if cells:
                raw_rows.append(cells)
    if not raw_rows:
        return []
    header = raw_rows[0]
    columns = {
        pos: name.strip() if name.strip() else f"col_{_column_letter(pos)}"
        for pos, name in header.items()
    }
    return [
        {name: row.get(pos, "") for pos, name in sorted(columns.items())} for row in raw_rows[1:]
    ]


def load_questions(xlsx_path: Path, language: str = "Kazakh") -> list[QorgauQuestion]:
    """Load the Kazakh or Russian sheet of ``ru_kz_question_only.xlsx``.

    ``language`` must be a sheet name: ``"Kazakh"`` (3,786 rows) or
    ``"Russian"`` (4,383 rows).
    """
    if language not in ("Kazakh", "Russian"):
        raise ValueError(f"language must be 'Kazakh' or 'Russian', got {language!r}")
    rows = read_xlsx_sheet(xlsx_path, language)
    questions: list[QorgauQuestion] = []
    for i, row in enumerate(rows):
        try:
            questions.append(
                QorgauQuestion(
                    # the Kazakh sheet's id header cell is a single space upstream,
                    # so the id column surfaces as col_A there
                    id=row["id"] if "id" in row else row["col_A"],
                    risk_area=row["risk_area"],
                    risk_area_specified=row.get("risk_area_specified", ""),
                    types_of_harm=row["types_of_harm"],
                    specific_harms=row["specific_harms"],
                    question_type=row["question_type"],
                    question=row["question"],
                )
            )
        except KeyError as err:
            raise ValueError(
                f"{xlsx_path} sheet {language!r} row {i + 2}: missing column {err}"
            ) from err
    return questions


def load_code_switched(csv_path: Path) -> list[CodeSwitchedQuestion]:
    """Load ``ru_kz_cs_questions.csv`` (500 code-switched probes)."""
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if CODE_SWITCHED_COLUMN not in fields:
            raise ValueError(
                f"{csv_path}: expected column {CODE_SWITCHED_COLUMN!r} "
                f"(upstream typo, verbatim), got {fields}"
            )
        return [
            CodeSwitchedQuestion(
                index=row["index"],
                original_index=row["original_index"],
                risk_area=row["risk_area"],
                types_of_harm=row["types_of_harm"],
                specific_harms=row["specific_harms"],
                question_type=row["question_type"],
                ru_version=row["ru_version"],
                kz_version=row["kz_version"],
                code_switched_version=row[CODE_SWITCHED_COLUMN],
            )
            for row in reader
        ]
