"""Doc-processing functions for the custom KazMMLU (Kazakh subset) lm-eval tasks.

KazMMLU rows have up to five options ("Option A".."Option E"); some rows have
only four. These helpers build the prompt and choice list per-document so the
task handles the variable option count correctly.
"""

OPTION_LETTERS = ("A", "B", "C", "D", "E")


def _valid_letters(doc: dict) -> list[str]:
    """Option letters that actually carry a non-empty value in this row."""
    letters: list[str] = []
    for letter in OPTION_LETTERS:
        val = doc.get(f"Option {letter}")
        if val is not None and str(val).strip() not in ("", "nan", "None"):
            letters.append(letter)
    return letters


def doc_to_text(doc: dict) -> str:
    """Question + lettered options + Kazakh answer cue (matches baseline protocol)."""
    parts = [doc["Question"]]
    for letter in _valid_letters(doc):
        parts.append(f"{letter}. {doc[f'Option {letter}']}")
    parts.append("Жауап:")
    return "\n".join(parts)


def doc_to_choice(doc: dict) -> list[str]:
    """Answer letters as continuation choices (target_delimiter adds the space)."""
    return _valid_letters(doc)


def doc_to_target(doc: dict) -> int:
    """Index of the gold letter within this row's valid choices."""
    return _valid_letters(doc).index(doc["Answer Key"].strip())
