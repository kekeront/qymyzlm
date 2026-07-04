"""QymyzLM testing lab: lazy-loading generation + embedding backends for the local UI.

Honesty first: there is NO trained QymyzLM checkpoint yet. "Generative QymyzLM" in this lab
is base ``Qwen/Qwen3-0.6B`` (the fork-point and the KazMMLU target to beat), OPTIONALLY
wrapped with an UNTRAINED Engram graft (random memory tables). The gate is sigmoid-initialized
near 0.5 (NOT ~0), and an RMSNorm renormalizes the retrieved memory output to unit RMS before
residual injection — so toggling Engram on injects a full-strength, untrained random
perturbation at the wrapped layers. This is an architecture smoke-test — does the grafted
forward pass run and stay coherent — NOT a quality feature; expect VISIBLE DEGRADATION vs base,
not "~base or slightly worse". The embedding side (``intfloat/multilingual-e5-large``) IS a
real, meaningful model. server.py surfaces this distinction to the user via
GenerativeLab.info / EmbeddingLab.info; do not fake it here.

Both labs are lazy: constructing them must never trigger a download or a weight load. The
first call to ``ensure_loaded`` / ``stream_generate`` / ``embed`` does that, guarded by a
lock so concurrent requests don't double-load. ``GenerativeLab`` additionally serializes the
entire ``stream_generate`` call (load + decode loop) behind ``self._generate_lock``: the base
model's layers are mutated in place when Engram-grafted (``from_base`` wraps the SAME layer
objects and stashes per-call state like ``_current_input_ids`` on them), so two concurrent
generations would clobber each other's state. This is a single-user smoke lab — honest
serialization (second request blocks until the first finishes) beats silent corruption.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kazllm.model import EngramConfig, QymyzForCausalLM

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_EMBED_MODEL = "intfloat/multilingual-e5-large"
DEFAULT_DEVICE = "cpu"

try:
    from qymyz_embed.prefixes import add_passage_prefix, add_query_prefix
except ImportError:  # pragma: no cover - embed package not on the path in some contexts
    log.warning("qymyz_embed.prefixes not importable; falling back to hardcoded e5 prefixes")

    def add_query_prefix(text: str) -> str:
        return text if text.startswith("query: ") else f"query: {text}"

    def add_passage_prefix(text: str) -> str:
        return text if text.startswith("passage: ") else f"passage: {text}"


def _sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Sample one next-token id per row of ``logits`` ([B, V]) -> [B].

    temperature<=0 is treated as greedy (argmax). Otherwise applies temperature scaling
    followed by nucleus (top-p) filtering before multinomial sampling.
    """
    if temperature <= 0:
        return logits.argmax(dim=-1)

    scaled = logits / temperature
    sorted_logits, sorted_idx = torch.sort(scaled, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumprobs = torch.cumsum(probs, dim=-1)
    # Drop tokens once the cumulative probability *before* them already exceeds top_p,
    # so at least the single highest-probability token is always kept.
    remove_sorted = (cumprobs - probs) > top_p
    sorted_logits = sorted_logits.masked_fill(remove_sorted, float("-inf"))
    filtered = torch.full_like(scaled, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    filtered_probs = torch.softmax(filtered, dim=-1)
    return torch.multinomial(filtered_probs, num_samples=1)[:, 0]


def _incremental_delta(full_text: str, prev_text: str) -> tuple[str, str]:
    """Compute the safe-to-emit delta between decode steps (HF TextStreamer pattern).

    Qwen3's byte-level BPE can split a multi-byte UTF-8 character (e.g. Kazakh
    Cyrillic) across two tokens. When the trailing character is incomplete,
    ``tokenizer.decode`` renders it as the U+FFFD replacement character
    (``"�"``) — emitting that would show a visible mojibake glyph, and by the
    time the next token completes the character, length-slicing against the old
    ``prev_text`` would miss it entirely (same or shorter length).

    Withholds emission (returns ``""``, unchanged ``prev_text``) whenever:
      - ``full_text`` ends with the replacement character (char not yet complete), or
      - ``full_text`` does not extend ``prev_text`` (decode of a growing token list
        should be prefix-stable; guard against the case it isn't).

    Otherwise returns ``(full_text[len(prev_text):], full_text)`` — the delta to
    yield and the new ``prev_text`` to remember. Callers should only advance their
    own ``prev_text`` to the returned value (i.e. it must not advance when
    withholding).
    """
    if full_text.endswith("�"):
        return "", prev_text
    if not full_text.startswith(prev_text):
        return "", prev_text
    return full_text[len(prev_text) :], full_text


class GenerativeLab:
    """Lazy wrapper around the base Qwen model, optionally Engram-grafted."""

    def __init__(self, base_model: str | None = None, device: str | None = None) -> None:
        self._base_model_name = base_model or os.environ.get("QYMYZLM_BASE", DEFAULT_BASE_MODEL)
        self._device = device or os.environ.get("QYMYZLM_DEVICE", DEFAULT_DEVICE)
        self._lock = threading.Lock()
        # Serializes entire stream_generate() calls (load + decode loop), distinct from
        # self._lock (which only guards double-checked lazy loading). Grafted layers share
        # mutable per-call state (e.g. `_current_input_ids` stashed on base layers), so
        # concurrent generations would clobber each other once requests run in a threadpool.
        self._generate_lock = threading.Lock()
        self._tokenizer: AutoTokenizer | None = None
        self._base: torch.nn.Module | None = None
        self._grafted: QymyzForCausalLM | None = None
        self.last_stats: dict = {}

    @property
    def info(self) -> dict:
        loaded = self._base is not None
        engram_grafted = self._grafted is not None

        if engram_grafted:
            total_params = sum(p.numel() for p in self._grafted.parameters())
            table_params = sum(p.numel() for p in self._grafted.engram_table_parameters())
            active_params = total_params - table_params
        elif loaded:
            total_params = sum(p.numel() for p in self._base.parameters())
            active_params = total_params
        else:
            total_params = 0
            active_params = 0

        dtype = str(next(self._base.parameters()).dtype) if loaded else ""

        return {
            "loaded": loaded,
            "base_model": self._base_model_name,
            "engram_grafted": engram_grafted,
            "total_params": total_params,
            "active_params": active_params,
            "device": self._device,
            "dtype": dtype,
        }

    def ensure_loaded(self, engram: bool = False) -> None:
        """Load tokenizer + base model once; optionally graft Engram on top (also once)."""
        if self._base is None:
            with self._lock:
                if self._base is None:
                    log.info("loading base model %s on %s", self._base_model_name, self._device)
                    self._tokenizer = AutoTokenizer.from_pretrained(self._base_model_name)
                    model = AutoModelForCausalLM.from_pretrained(
                        self._base_model_name, dtype=torch.float32
                    )
                    model.to(self._device)
                    model.eval()
                    self._base = model

        if engram and self._grafted is None:
            with self._lock:
                if self._grafted is None:
                    log.info("grafting untrained Engram onto %s", self._base_model_name)
                    grafted = QymyzForCausalLM.from_base(self._base, engram_config=EngramConfig())
                    grafted.to(self._device)
                    grafted.eval()
                    self._grafted = grafted

    def _eos_token_ids(self) -> set[int]:
        eos_ids: set[int] = set()
        gen_config_eos = getattr(
            getattr(self._base, "generation_config", None), "eos_token_id", None
        )
        if isinstance(gen_config_eos, int):
            eos_ids.add(gen_config_eos)
        elif isinstance(gen_config_eos, list):
            eos_ids.update(gen_config_eos)
        if self._tokenizer.eos_token_id is not None:
            eos_ids.add(self._tokenizer.eos_token_id)
        return eos_ids

    def stream_generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
        engram: bool = False,
        chat: bool = True,
    ) -> Iterator[str]:
        """Manual sampling decode loop. Yields decoded token strings one at a time.

        engram=False runs the plain base model; engram=True runs the Engram-grafted
        module (built lazily on first use). Re-decodes the growing generated suffix each
        step (no KV cache), then runs it through ``_incremental_delta`` so multi-byte
        Cyrillic tokens always render as complete text (never a stray ``"�"``) — simple
        and correct, not optimized for throughput (this is a smoke-test lab, not a
        serving stack).

        The whole call (load + decode loop) is serialized on ``self._generate_lock``:
        the Engram-grafted module wraps the SAME base layer objects and stashes
        per-call state on them (e.g. ``_current_input_ids``), and ``last_stats`` is
        shared on ``self`` — concurrent generations would otherwise clobber each
        other's state. A second concurrent request blocks until the first finishes;
        that's the correct, honest behavior for a single-user smoke lab.
        """
        with self._generate_lock:
            self.ensure_loaded(engram=engram)
            model = self._grafted if engram else self._base
            tokenizer = self._tokenizer

            if chat and getattr(tokenizer, "chat_template", None):
                try:
                    text = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        add_generation_prompt=True,
                        tokenize=False,
                        enable_thinking=False,
                    )
                except TypeError:
                    # Tokenizer's chat template doesn't accept enable_thinking (not
                    # Qwen3-style thinking-mode template) — fall back gracefully.
                    text = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        add_generation_prompt=True,
                        tokenize=False,
                    )
            else:
                text = prompt

            encoded = tokenizer(text, return_tensors="pt")
            input_ids = encoded["input_ids"].to(self._device)
            attention_mask = encoded["attention_mask"].to(self._device)
            eos_ids = self._eos_token_ids()

            generated_ids: list[int] = []
            prev_text = ""
            start = time.perf_counter()

            with torch.no_grad():
                for _ in range(max_new_tokens):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    next_logits = outputs.logits[:, -1, :]
                    next_id = _sample_next_token(next_logits, temperature, top_p)
                    next_id_int = int(next_id.item())

                    if next_id_int in eos_ids:
                        break

                    generated_ids.append(next_id_int)
                    full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    delta, prev_text = _incremental_delta(full_text, prev_text)
                    if delta:
                        yield delta

                    input_ids = torch.cat([input_ids, next_id.view(1, 1)], dim=-1)
                    attention_mask = torch.cat(
                        [
                            attention_mask,
                            torch.ones((1, 1), dtype=attention_mask.dtype, device=input_ids.device),
                        ],
                        dim=-1,
                    )

            elapsed_s = time.perf_counter() - start
            n_tokens = len(generated_ids)
            self.last_stats = {
                "n_tokens": n_tokens,
                "elapsed_s": elapsed_s,
                "tok_per_s": (n_tokens / elapsed_s) if elapsed_s > 0 else 0.0,
                "engram": engram,
                "model": self._base_model_name,
            }


class EmbeddingLab:
    """Lazy wrapper around a sentence-transformers e5-family embedding model."""

    def __init__(self, model: str | None = None, device: str | None = None) -> None:
        self._model_name = model or os.environ.get("QYMYZLM_EMBED", DEFAULT_EMBED_MODEL)
        self._device = device or os.environ.get("QYMYZLM_DEVICE", DEFAULT_DEVICE)
        self._lock = threading.Lock()
        self._model: SentenceTransformer | None = None

    @property
    def info(self) -> dict:
        return {
            "loaded": self._model is not None,
            "model": self._model_name,
            "dim": self._model.get_sentence_embedding_dimension()
            if self._model is not None
            else None,
        }

    def _ensure_loaded(self) -> None:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    log.info("loading embedding model %s on %s", self._model_name, self._device)
                    self._model = SentenceTransformer(self._model_name, device=self._device)

    def embed(
        self, texts: list[str], mode: str = "query"
    ) -> tuple[list[list[float]], list[list[float]]]:
        """Embed ``texts`` with the e5 query/passage prefix; return (vectors, cosine_sim)."""
        if not texts:
            raise ValueError("texts must not be empty")
        if mode not in ("query", "passage"):
            raise ValueError(f"mode must be 'query' or 'passage', got {mode!r}")

        self._ensure_loaded()
        prefix_fn = add_query_prefix if mode == "query" else add_passage_prefix
        prefixed = [prefix_fn(t) for t in texts]

        vectors = self._model.encode(prefixed, normalize_embeddings=True)
        similarity = vectors @ vectors.T
        return vectors.tolist(), similarity.tolist()
