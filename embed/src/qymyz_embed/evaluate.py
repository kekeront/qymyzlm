"""Evaluation shim — ALL numbers come from evallab/ runners (package `kazeval`).

embed/ deliberately contains NO evaluation logic (embed/CLAUDE.md hard constraint:
"Every eval number comes from evallab/ runners — never self-reported ad-hoc"). This shim
only forwards to kazeval so the embed workflow has a single entry point; results are the
committed JSONs in evallab/results/, and the leaderboard renders from those.

Usage (arguments are passed through to kazeval unchanged):
    python -m qymyz_embed.evaluate <kazeval args...>
"""

from __future__ import annotations

import runpy
import sys

_MISSING_KAZEVAL = (
    "kazeval (the evallab/ package) is not installed in this environment. Install the "
    "qymyzlm workspace (evallab/ is a member) and retry. All QymyzEmbed numbers MUST come "
    "from evallab runners — this shim has no fallback eval logic by design."
)
_NO_CLI_KAZEVAL = (
    "kazeval is installed but exposes neither kazeval.main() nor a kazeval.__main__ module "
    "yet — run its runner modules directly (e.g. python -m kazeval.hardneg) or update "
    "evallab. This shim intentionally has no fallback eval logic."
)


def main(argv: list[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    try:
        import kazeval
    except ImportError as exc:
        raise SystemExit(_MISSING_KAZEVAL) from exc

    entry = getattr(kazeval, "main", None)
    if callable(entry):
        return int(entry(args) or 0)
    # No kazeval.main — fall back to `python -m kazeval` semantics.
    sys.argv = ["kazeval", *args]
    try:
        runpy.run_module("kazeval", run_name="__main__", alter_sys=True)
    except ImportError as exc:  # no kazeval.__main__ either
        raise SystemExit(_NO_CLI_KAZEVAL) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
