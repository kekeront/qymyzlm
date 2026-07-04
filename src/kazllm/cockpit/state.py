"""Read-only data layer for the campaign cockpit dashboard.

Collects the `/api/status` payload from a handful of plain-text/JSONL state
sources that already exist elsewhere in the Kazakh Frontier Lab campaign
(`SESSION.md`, `CLAIM.md`, the atlas KB, git repos, a Kaggle kernel). Every
collector is defensive: on any failure it appends a short string to an
``errors`` list and returns safe defaults (nulls/zeros/empty collections) —
this module must never raise up to the FastAPI layer because a file is
missing, malformed, or `kaggle`/`git` is unavailable.

Each collector returns ``(sub_dict, errors)`` so :func:`collect_all` can
merge everything into the exact `/api/status` shape without any collector
needing to know about the others.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# State sources (absolute defaults, each overridable via its env var).
# --------------------------------------------------------------------------

_DEFAULT_SESSION_MD = Path.home() / "projects/SLMs/basic/.claude/SESSION.md"
_DEFAULT_CLAIM_MD = Path.home() / "projects/SLMs/CLAIM.md"
_DEFAULT_NODES = Path.home() / "projects/SLMs/kazakh-nlp-atlas/kb/nodes.jsonl"
_DEFAULT_KAGGLE_KERNEL = "kekeront/kazeval-planka-me5-large-on-kazqad"

REPOS: dict[str, Path] = {
    "qymyzlm": Path.home() / "projects/SLMs/qymyzlm",
    "atlas": Path.home() / "projects/SLMs/kazakh-nlp-atlas",
}

# CLAIM.md lanes, in display order: (COUNTS key, header emoji).
_CLAIM_LANES: list[tuple[str, str]] = [
    ("LIVE", "\U0001f534"),  # 🔴
    ("PROMISED", "\U0001f3af"),  # 🎯
    ("VERIFIED", "✅"),  # ✅
    ("POSTED", "\U0001f4e2"),  # 📢
    ("RETRACTED", "⛔"),  # ⛔
]

_PLACEHOLDER_RE = re.compile(r"^_\(.*\)_$")
_MD_TOKEN_RE = re.compile(r"[*_`]")
_WHITESPACE_RE = re.compile(r"\s+")

# Kaggle status cache: kernel name -> (fetched_at_epoch, result_dict).
_KAGGLE_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_KAGGLE_TTL_SECONDS = 60.0


def _session_md_path() -> Path:
    return Path(os.environ.get("COCKPIT_SESSION_MD", str(_DEFAULT_SESSION_MD))).expanduser()


def _claim_md_path() -> Path:
    return Path(os.environ.get("COCKPIT_CLAIM_MD", str(_DEFAULT_CLAIM_MD))).expanduser()


def _nodes_path() -> Path:
    return Path(os.environ.get("COCKPIT_NODES", str(_DEFAULT_NODES))).expanduser()


def _kaggle_kernel_name() -> str:
    return os.environ.get("COCKPIT_KAGGLE_KERNEL", _DEFAULT_KAGGLE_KERNEL)


def _clean_text(text: str) -> str:
    """Strip markdown emphasis/code tokens and collapse whitespace."""
    text = _MD_TOKEN_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip(" :-—")


# --------------------------------------------------------------------------
# focus() — SESSION.md "## ⭐ ФОКУС" block
# --------------------------------------------------------------------------


def _derive_focus_status(primary: str) -> str:
    upper = primary.upper()
    for keyword, label in (
        ("BLOCKED", "blocked"),
        ("FAILED", "failed"),
        ("RUNNING", "running"),
        ("COMPLETE", "done"),
        ("DONE", "done"),
        ("PENDING", "pending"),
    ):
        if keyword in upper:
            return label
    return "in flight" if primary else "unknown"


def focus() -> tuple[dict[str, Any], list[str]]:
    """Parse the SESSION.md "## ⭐ ФОКУС" block into primary/status/parked."""
    errors: list[str] = []
    default: dict[str, Any] = {"primary": "", "status": "unknown", "parked": []}
    path = _session_md_path()

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"focus: cannot read SESSION.md ({path}): {exc}")
        return default, errors

    block_start = text.find("## ⭐")  # "## ⭐"
    if block_start == -1:
        errors.append("focus: '## ⭐ ФОКУС' block not found in SESSION.md")
        return default, errors
    block_end = text.find("\n## ", block_start + 4)
    block = text[block_start : block_end if block_end != -1 else len(text)]

    primary = ""
    primary_idx = block.find("**PRIMARY")
    if primary_idx == -1:
        errors.append("focus: '**PRIMARY' marker not found in FOCUS block")
    else:
        start = primary_idx + len("**PRIMARY")
        end = block.find("\n", start)  # primary is the one-liner ON the marker line
        raw = block[start : end if end != -1 else len(block)]
        if ":**" in raw:  # drop the "(in flight):**" qualifier prefix
            raw = raw.split(":**", 1)[1]
        primary = _clean_text(raw)

    parked: list[str] = []
    parked_idx = block.find("**PARKED")
    if parked_idx == -1:
        errors.append("focus: '**PARKED' marker not found in FOCUS block")
    else:
        start = parked_idx + len("**PARKED")
        end = block.find("\n\n", start)  # parked is a soft-wrapped multi-line paragraph
        raw = block[start : end if end != -1 else len(block)]
        if ":**" in raw:  # drop the "(не начинать...):**" qualifier prefix
            raw = raw.split(":**", 1)[1]
        raw = raw.replace("\n", " ")  # unwrap before splitting; items are ';'-separated
        for part in raw.split(";"):  # paths contain ':' and ',', so split on ';' only
            cleaned = _clean_text(part)
            if len(cleaned) >= 3:
                parked.append(cleaned)

    return {
        "primary": primary,
        "status": _derive_focus_status(primary),
        "parked": parked,
    }, errors


# --------------------------------------------------------------------------
# claims() — CLAIM.md lane tables
# --------------------------------------------------------------------------


def _table_first_column(section: str) -> list[str]:
    """Extract the first-column label of every data row in a markdown table."""
    rows = [ln for ln in section.splitlines() if ln.strip().startswith("|")]
    if len(rows) < 2:
        return []
    data_rows = rows[2:]  # rows[0]=header, rows[1]=separator
    labels: list[str] = []
    for row in data_rows:
        cells = row.strip().strip("|").split("|")
        if not cells:
            continue
        first = cells[0].strip()
        if not first or _PLACEHOLDER_RE.match(first):
            continue  # empty-lane placeholder row, e.g. "_(пусто)_"
        labels.append(_clean_text(first))
    return labels


def claims() -> tuple[dict[str, Any], list[str]]:
    """Parse CLAIM.md lane headers into per-lane counts + live/retracted labels."""
    errors: list[str] = []
    counts: dict[str, int] = {name: 0 for name, _ in _CLAIM_LANES}
    live: list[str] = []
    retracted: list[str] = []
    path = _claim_md_path()

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"claims: cannot read CLAIM.md ({path}): {exc}")
        return {"counts": counts, "live": live, "retracted": retracted}, errors

    for name, emoji in _CLAIM_LANES:
        marker = f"## {emoji} {name}"
        idx = text.find(marker)
        if idx == -1:
            errors.append(f"claims: lane '{name}' header not found in CLAIM.md")
            continue
        end = text.find("\n## ", idx + len(marker))
        section = text[idx : end if end != -1 else len(text)]
        labels = _table_first_column(section)
        counts[name] = len(labels)
        if name == "LIVE":
            live.extend(labels[:6])
        elif name == "RETRACTED":
            retracted.extend(labels[:6])

    return {"counts": counts, "live": live, "retracted": retracted}, errors


# --------------------------------------------------------------------------
# kb() — nodes.jsonl
# --------------------------------------------------------------------------


def kb() -> tuple[dict[str, Any], list[str]]:
    """Summarize the atlas KB: nodes, papers, sources, topics, claims, verified."""
    errors: list[str] = []
    default: dict[str, Any] = {
        "nodes": 0,
        "papers": 0,
        "sources": 0,
        "topics": 0,
        "claims": 0,
        "verified": 0,
    }
    path = _nodes_path()

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"kb: cannot read nodes.jsonl ({path}): {exc}")
        return default, errors

    nodes = 0
    papers = 0
    topics: set[str] = set()
    total_claims = 0
    verified = 0

    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            node = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"kb: line {lineno} invalid JSON: {exc}")
            continue
        if not isinstance(node, dict):
            errors.append(f"kb: line {lineno} is not a JSON object")
            continue

        nodes += 1
        abstract = node.get("abstract")
        if isinstance(abstract, str) and abstract.strip():
            papers += 1
        for topic in node.get("topics") or []:
            if isinstance(topic, str):
                topics.add(topic)
        for claim in node.get("claims") or []:
            if not isinstance(claim, dict):
                continue
            total_claims += 1
            if claim.get("verification"):
                verified += 1

    return {
        "nodes": nodes,
        "papers": papers,
        "sources": nodes - papers,
        "topics": len(topics),
        "claims": total_claims,
        "verified": verified,
    }, errors


# --------------------------------------------------------------------------
# repos() — git HEAD/dirty/unpushed per repo
# --------------------------------------------------------------------------


def _git_head(repo: Path) -> tuple[str, str, list[str]]:
    try:
        proc = subprocess.run(
            ["git", "log", "-1", "--pretty=%h\x1f%s"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return "", "", [f"git log failed in {repo}: {exc}"]
    if proc.returncode != 0:
        return "", "", [f"git log failed in {repo}: {proc.stderr.strip()[:200]}"]
    head, _, subject = proc.stdout.strip().partition("\x1f")
    return head, subject, []


def _git_dirty(repo: Path) -> tuple[bool, list[str]]:
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, [f"git status failed in {repo}: {exc}"]
    if proc.returncode != 0:
        return False, [f"git status failed in {repo}: {proc.stderr.strip()[:200]}"]
    return bool(proc.stdout.strip()), []


def _git_unpushed(repo: Path) -> tuple[int, list[str]]:
    try:
        verify = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", "origin/main"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 0, [f"git rev-parse origin/main failed in {repo}: {exc}"]
    if verify.returncode != 0:
        return 0, []  # no upstream — 0 unpushed by definition

    try:
        proc = subprocess.run(
            ["git", "rev-list", "--count", "origin/main..HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 0, [f"git rev-list failed in {repo}: {exc}"]
    if proc.returncode != 0:
        return 0, [f"git rev-list failed in {repo}: {proc.stderr.strip()[:200]}"]
    try:
        return int(proc.stdout.strip() or 0), []
    except ValueError as exc:
        return 0, [f"git rev-list output unparseable in {repo}: {exc}"]


def repos() -> tuple[list[dict[str, Any]], list[str]]:
    """Per-repo HEAD/subject/dirty/unpushed via `git` subprocess (list argv, no shell)."""
    errors: list[str] = []
    out: list[dict[str, Any]] = []

    for name, path in REPOS.items():
        entry: dict[str, Any] = {
            "name": name,
            "head": "",
            "subject": "",
            "dirty": False,
            "unpushed": 0,
        }
        if not path.exists() or not (path / ".git").exists():
            errors.append(f"repos: '{name}' not a git repo at {path}")
            out.append(entry)
            continue

        head, subject, head_errs = _git_head(path)
        dirty, dirty_errs = _git_dirty(path)
        unpushed, unpushed_errs = _git_unpushed(path)
        errors.extend(f"repos[{name}]: {e}" for e in (*head_errs, *dirty_errs, *unpushed_errs))

        entry.update(head=head, subject=subject, dirty=dirty, unpushed=unpushed)
        out.append(entry)

    return out, errors


# --------------------------------------------------------------------------
# kaggle() — cached `kaggle kernels status`
# --------------------------------------------------------------------------

_KAGGLE_STATUS_RE = re.compile(r'status\s+"(?:KernelWorkerStatus\.)?(\w+)"')
_KAGGLE_VERSION_RE = re.compile(r"[Vv]ersion\s+(\d+)")


def _fetch_kaggle_status(kernel: str) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    result: dict[str, Any] = {
        "kernel": kernel,
        "version": None,
        "status": "unknown",
        "running_for": None,  # not derivable from `kernels status` alone
        "fp16": None,  # only ever proven by a (deliberately skipped) log grep
        "note": "",
    }

    try:
        proc = subprocess.run(
            ["kaggle", "kernels", "status", kernel],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except FileNotFoundError:
        errors.append("kaggle: CLI not found on PATH")
        result["note"] = "kaggle CLI not installed"
        return result, errors
    except subprocess.TimeoutExpired:
        errors.append(f"kaggle: 'kernels status {kernel}' timed out")
        result["note"] = "kaggle status check timed out"
        return result, errors
    except OSError as exc:
        errors.append(f"kaggle: 'kernels status {kernel}' failed: {exc}")
        result["note"] = "kaggle status check failed"
        return result, errors

    if proc.returncode != 0:
        errors.append(f"kaggle: status check failed: {(proc.stderr or proc.stdout).strip()[:200]}")
        result["note"] = "kaggle CLI error (auth/network?)"
        return result, errors

    out = proc.stdout.strip()
    status_match = _KAGGLE_STATUS_RE.search(out)
    if status_match:
        result["status"] = status_match.group(1)
        result["note"] = f"kernel {status_match.group(1).lower()}"
    else:
        result["note"] = out[:120] or "unparseable kaggle status output"

    version_match = _KAGGLE_VERSION_RE.search(out)
    if version_match:
        result["version"] = int(version_match.group(1))

    return result, errors


def kaggle() -> tuple[dict[str, Any], list[str]]:
    """`kaggle kernels status <kernel>`, module-level cached for ~60s."""
    kernel = _kaggle_kernel_name()
    now = time.time()
    cached = _KAGGLE_CACHE.get(kernel)
    if cached is not None and (now - cached[0]) < _KAGGLE_TTL_SECONDS:
        return cached[1], []

    result, errors = _fetch_kaggle_status(kernel)
    _KAGGLE_CACHE[kernel] = (now, result)
    return result, errors


# --------------------------------------------------------------------------
# ladder() — the 5 reputation rungs
# --------------------------------------------------------------------------


def ladder(posted_count: int) -> list[dict[str, Any]]:
    """The 5 reputation rungs; rung 1 flips to done once a POSTED claim exists."""
    rungs: list[tuple[int, str, str]] = [
        (0, "Machine", "done"),
        (1, "First public proof", "done" if posted_count > 0 else "active"),
        (2, "Adoption artifact", "todo"),
        (3, "Upstream dependency", "todo"),
        (4, '"Learn from him to keep up"', "todo"),
    ]
    return [{"rung": rung, "title": title, "status": status} for rung, title, status in rungs]


# --------------------------------------------------------------------------
# collect_all() — full /api/status payload
# --------------------------------------------------------------------------


def collect_all() -> dict[str, Any]:
    """Assemble the exact `/api/status` payload; never raises."""
    errors: list[str] = []

    focus_d, e = focus()
    errors.extend(e)
    claims_d, e = claims()
    errors.extend(e)
    kb_d, e = kb()
    errors.extend(e)
    repos_d, e = repos()
    errors.extend(e)
    compute_d, e = kaggle()
    errors.extend(e)

    posted_count = claims_d["counts"].get("POSTED", 0)
    ladder_d = ladder(posted_count)

    return {
        "updated_at": datetime.now().isoformat(),
        "focus": focus_d,
        "compute": compute_d,
        "claims": claims_d,
        "kb": kb_d,
        "repos": repos_d,
        "ladder": ladder_d,
        "errors": errors,
    }
