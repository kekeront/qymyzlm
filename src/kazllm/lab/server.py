"""FastAPI app for the QymyzLM testing lab.

Serves two honesty-first smoke-test surfaces over HTTP:

- Generative: base ``Qwen/Qwen3-0.6B`` (the fork-point and KazMMLU target to
  beat), optionally wrapped with an UNTRAINED Engram graft. There is NO trained
  QymyzLM checkpoint — toggling Engram on is an architecture smoke-test (does
  the grafted forward pass run and stay coherent), not a quality feature.
- Embedding: ``intfloat/multilingual-e5-large``, a real, meaningful model.

Routes are mounted under ``/api``; a static UI is mounted at ``/`` last, so
API routes always win over the catch-all static handler.
"""

import json
import logging
from collections.abc import Iterator
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from kazllm.lab.inference import EmbeddingLab, GenerativeLab

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="QymyzLM Testing Lab")

# Lazy singletons: constructing these must NOT load any weights.
GEN = GenerativeLab()
EMB = EmbeddingLab()


class GenerateRequest(BaseModel):
    """Body for ``POST /api/generate``."""

    prompt: str
    max_new_tokens: int = Field(256, ge=1, le=2048)
    temperature: float = Field(0.8, ge=0.0, le=5.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    engram: bool = False
    chat: bool = True


class EmbedRequest(BaseModel):
    """Body for ``POST /api/embed``."""

    texts: list[str]
    mode: str = "query"


@app.get("/api/health")
def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


@app.get("/api/model")
def model_info() -> dict[str, dict]:
    """Report current load state and metadata for both labs."""
    return {"generative": GEN.info, "embedding": EMB.info}


def _generate_events(req: GenerateRequest) -> Iterator[str]:
    """Stream ``GEN.stream_generate`` output as SSE, then a final done+stats event.

    Deliberately a PLAIN SYNC generator, not ``async def``: generation drives a
    full fp32 forward pass per token with no natural await points, so an async
    generator here would block the whole event loop (freezing ``/api/health``,
    the UI, and other in-flight requests) for the entire run. Starlette's
    ``StreamingResponse`` runs sync iterators in a worker thread, keeping the
    event loop free.

    Any exception raised mid-stream is caught and surfaced as a single
    ``{"error": ...}`` event instead of crashing the response.
    """
    try:
        GEN.ensure_loaded(engram=req.engram)
        for token in GEN.stream_generate(
            req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            engram=req.engram,
            chat=req.chat,
        ):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield f"data: {json.dumps({'done': True, 'stats': GEN.last_stats})}\n\n"
    except Exception as exc:  # noqa: BLE001 - surface any failure as an SSE error event
        log.exception("generation stream failed")
        yield f"data: {json.dumps({'error': str(exc)})}\n\n"


@app.post("/api/generate")
def generate(req: GenerateRequest) -> StreamingResponse:
    """Stream generated tokens as ``text/event-stream``."""
    return StreamingResponse(_generate_events(req), media_type="text/event-stream")


@app.post("/api/embed")
def embed(req: EmbedRequest) -> dict:
    """Embed texts and return vectors plus their pairwise cosine similarity."""
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts must be non-empty")
    if req.mode not in ("query", "passage"):
        raise HTTPException(status_code=400, detail="mode must be 'query' or 'passage'")
    vectors, similarity = EMB.embed(req.texts, mode=req.mode)
    return {
        "model": EMB.info["model"],
        "dim": EMB.info["dim"],
        "vectors": vectors,
        "similarity": similarity,
    }


# Mounted LAST: StaticFiles is a catch-all for "/", so it must not shadow /api routes.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
