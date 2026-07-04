"""QymyzLM testing lab: a local FastAPI + static UI for smoke-testing the
generative (Qwen3-0.6B, optionally Engram-grafted) and embedding (mE5-large)
sides of the campaign. No trained QymyzLM checkpoint exists yet — see
``kazllm.lab.server`` for the honesty banner this implies.
"""

from kazllm.lab.inference import EmbeddingLab, GenerativeLab

__all__ = ["GenerativeLab", "EmbeddingLab"]
