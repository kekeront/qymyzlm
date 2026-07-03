"""kazeval — Kazakh Frontier Lab benchmarking package.

Single source of truth for ALL lab numbers: no model claim exists until a runner
here reproduces it. Tracks: KazQAD retrieval/reranking (kk-MTEB), the hard-negatives
MRR protocol (planka re-measurement), KazMMLU (Kazakh subset), Qorgau guardrails.
"""

from kazeval.results import ResultRecord, load_records, save_record, validate_record

__version__ = "0.1.0"

__all__ = [
    "ResultRecord",
    "__version__",
    "load_records",
    "save_record",
    "validate_record",
]
