"""kk-MTEB task classes (written here first, then PR'd upstream to mteb)."""

from kazeval.tasks.kazqad_reranking import KazQADReranking
from kazeval.tasks.kazqad_retrieval import KazQADRetrieval

__all__ = ["KazQADReranking", "KazQADRetrieval"]
