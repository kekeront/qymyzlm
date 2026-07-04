"""FastAPI app for the Campaign Mission Control cockpit.

Read-only dashboard: aggregates campaign state (focus, compute, claims, KB,
repos, ladder) from local files and light shell-outs (git, kaggle) via
``kazllm.cockpit.state.collect_all`` and serves it as JSON under ``/api``.
A static UI is mounted at ``/`` last, so API routes always win over the
catch-all static handler.

This tool never mutates anything it reads — no writes to SESSION.md,
CLAIM.md, nodes.jsonl, or any repo.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from kazllm.cockpit import state

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Campaign Mission Control")


@app.get("/api/health")
def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


@app.get("/api/status")
def status() -> dict:
    """Return the full campaign status snapshot.

    Deliberately a PLAIN SYNC route, not ``async def``: ``collect_all``
    shells out to ``git`` and ``kaggle`` and reads files from disk, with no
    natural await points. Starlette runs plain ``def`` routes in a worker
    thread, keeping the event loop (and ``/api/health``) free while a slow
    collector runs. ``collect_all`` never raises — every collector degrades
    gracefully and reports failures in the ``errors`` list instead.
    """
    return state.collect_all()


# Mounted LAST: StaticFiles is a catch-all for "/", so it must not shadow /api routes.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
