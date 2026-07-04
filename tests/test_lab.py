"""Offline tests for the QymyzLM testing lab HTTP surface (no model downloads).

Monkeypatches ``kazllm.lab.server.GEN`` / ``EMB`` with tiny stubs so ``/api/health``,
``/api/model``, ``/api/embed``, and ``/api/generate`` are exercised for correct shape
and status codes without ever constructing real torch/transformers/sentence-transformers
model objects.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from kazllm.lab import server
from kazllm.lab.inference import _incremental_delta

STUB_TOKENS = ["Са", "лем", "!"]


class StubGenerativeLab:
    """Stand-in for GenerativeLab: yields fixed tokens, touches no real model."""

    def __init__(self) -> None:
        self.last_stats: dict = {}
        self._loaded = False
        self._engram_grafted = False

    @property
    def info(self) -> dict:
        return {
            "loaded": self._loaded,
            "base_model": "stub/base",
            "engram_grafted": self._engram_grafted,
            "total_params": 100,
            "active_params": 90,
            "device": "cpu",
            "dtype": "torch.float32",
        }

    def ensure_loaded(self, engram: bool = False) -> None:
        self._loaded = True
        if engram:
            self._engram_grafted = True

    def stream_generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
        engram: bool = False,
        chat: bool = True,
    ) -> Iterator[str]:
        yield from STUB_TOKENS
        self.last_stats = {
            "n_tokens": len(STUB_TOKENS),
            "elapsed_s": 0.01,
            "tok_per_s": 300.0,
            "engram": engram,
            "model": "stub/base",
        }


class StubEmbeddingLab:
    """Stand-in for EmbeddingLab: deterministic tiny vectors, touches no real model."""

    def __init__(self) -> None:
        self._loaded = False

    @property
    def info(self) -> dict:
        return {"loaded": self._loaded, "model": "stub/embed", "dim": 3}

    def embed(
        self, texts: list[str], mode: str = "query"
    ) -> tuple[list[list[float]], list[list[float]]]:
        self._loaded = True
        vectors = [[float(i), 0.0, 1.0] for i in range(len(texts))]
        n = len(vectors)
        similarity = [[1.0 if i == j else 0.5 for j in range(n)] for i in range(n)]
        return vectors, similarity


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """TestClient wired to fresh stub singletons for each test (no weight loads)."""
    monkeypatch.setattr(server, "GEN", StubGenerativeLab())
    monkeypatch.setattr(server, "EMB", StubEmbeddingLab())
    return TestClient(server.app)


def test_health(client: TestClient) -> None:
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_model_info_shape(client: TestClient) -> None:
    res = client.get("/api/model")
    assert res.status_code == 200
    data = res.json()
    assert set(data.keys()) == {"generative", "embedding"}
    assert data["generative"]["base_model"] == "stub/base"
    assert data["embedding"]["model"] == "stub/embed"


def test_embed_returns_square_similarity_matrix(client: TestClient) -> None:
    res = client.post("/api/embed", json={"texts": ["a", "b", "c"], "mode": "query"})
    assert res.status_code == 200
    data = res.json()
    assert data["model"] == "stub/embed"
    assert data["dim"] == 3
    assert len(data["vectors"]) == 3

    sim = data["similarity"]
    assert len(sim) == 3
    for row in sim:
        assert len(row) == 3


def test_embed_empty_texts_returns_400(client: TestClient) -> None:
    res = client.post("/api/embed", json={"texts": []})
    assert res.status_code == 400


def test_embed_invalid_mode_returns_400(client: TestClient) -> None:
    res = client.post("/api/embed", json={"texts": ["a", "b"], "mode": "bogus"})
    assert res.status_code == 400


def test_generate_stream_ends_with_done_event_and_stats(client: TestClient) -> None:
    events: list[dict] = []
    with client.stream(
        "POST",
        "/api/generate",
        json={"prompt": "hello", "max_new_tokens": 8, "engram": False},
    ) as res:
        assert res.status_code == 200
        for line in res.iter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if not payload:
                continue
            events.append(json.loads(payload))

    assert events, "expected at least one SSE event"

    token_events = [e for e in events if "token" in e]
    assert len(token_events) == len(STUB_TOKENS)
    assert [e["token"] for e in token_events] == STUB_TOKENS

    done_events = [e for e in events if e.get("done")]
    assert len(done_events) == 1
    stats = done_events[0]["stats"]
    assert stats["n_tokens"] == len(STUB_TOKENS)
    assert stats["engram"] is False
    assert stats["model"] == "stub/base"


def test_incremental_delta_withholds_split_cyrillic_char() -> None:
    """Simulates a Cyrillic char split across two BPE tokens (decode -> '...�').

    Step 1: clean prefix decodes normally.
    Step 2: the next token lands mid-character; decode renders a trailing U+FFFD
        replacement char. Must withhold (delta == "", prev_text unchanged).
    Step 3: the following token completes the character; the full suffix
        ("қ") is now emitted in one delta and prev_text advances.
    """
    emitted: list[str] = []
    prev_text = ""

    delta, prev_text = _incremental_delta("Қаза", prev_text)
    assert delta == "Қаза"
    assert prev_text == "Қаза"
    emitted.append(delta)

    delta, prev_text = _incremental_delta("Қаза�", prev_text)
    assert delta == ""
    assert prev_text == "Қаза"  # unchanged while withholding
    emitted.append(delta)

    delta, prev_text = _incremental_delta("Қазақ", prev_text)
    assert delta == "қ"
    assert prev_text == "Қазақ"
    emitted.append(delta)

    assert all("�" not in piece for piece in emitted), "replacement char was emitted"
    assert "".join(emitted) == "Қазақ"


def test_generate_max_new_tokens_out_of_bounds_returns_422(client: TestClient) -> None:
    res = client.post(
        "/api/generate",
        json={"prompt": "hello", "max_new_tokens": 10_000_000},
    )
    assert res.status_code == 422


def test_generate_temperature_out_of_bounds_returns_422(client: TestClient) -> None:
    res = client.post(
        "/api/generate",
        json={"prompt": "hello", "temperature": 99},
    )
    assert res.status_code == 422
