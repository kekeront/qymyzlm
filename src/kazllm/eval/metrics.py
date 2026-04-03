"""Evaluation metrics: accuracy, chrF for translation."""


def accuracy(predictions: list, references: list) -> float:
    if not predictions:
        return 0.0
    correct = sum(p == r for p, r in zip(predictions, references))
    return correct / len(predictions)


def chrf_score(hypotheses: list[str], references: list[str]) -> float:
    """Compute corpus-level chrF++ score using sacrebleu."""
    try:
        from sacrebleu.metrics import CHRF
    except ImportError:
        raise ImportError("Install sacrebleu: pip install sacrebleu>=2.4.0")

    metric = CHRF(word_order=2)  # chrF++ uses word order 2
    return metric.corpus_score(hypotheses, [references]).score
