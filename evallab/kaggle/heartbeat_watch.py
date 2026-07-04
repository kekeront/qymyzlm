#!/usr/bin/env python3
"""Watch a Kaggle run's heartbeat (see the beat() cell in the run notebooks).

Kaggle's API exposes only a status enum — no elapsed time, no mid-run logs, so
"RUNNING" cannot distinguish healthy from hung. Runs push a tiny JSON to the
private HF dataset repo below at every stage transition; this script polls it.

    python3 evallab/kaggle/heartbeat_watch.py            # one-shot
    python3 evallab/kaggle/heartbeat_watch.py --follow   # poll every 60s

Staleness is judged against the stage that is running: a beat is only expected
at the NEXT transition, so "no beat for 2h during eval" is normal; use
--stale-after to tune when to start worrying.
"""

from __future__ import annotations

import argparse
import calendar
import json
import sys
import time

HB_REPO = "kekeront/qymyzlm-run-heartbeat"
HB_FILE = "heartbeat.json"


def fetch() -> dict:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(HB_REPO, HB_FILE, repo_type="dataset", force_download=True)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def render(hb: dict, stale_after_s: int) -> str:
    age_s = None
    ts = hb.get("ts_utc")
    if ts:
        age_s = int(time.time() - calendar.timegm(time.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")))
    extras = {k: v for k, v in hb.items() if k not in ("run", "stage", "elapsed_s", "ts_utc")}
    line = (
        f"run={hb.get('run')} stage={hb.get('stage')} "
        f"run_elapsed={hb.get('elapsed_s', 0) / 60:.0f}m "
        f"beat_age={'?' if age_s is None else f'{age_s / 60:.0f}m'}"
    )
    if extras:
        line += f" {extras}"
    if age_s is not None and age_s > stale_after_s:
        line += f"  <-- STALE (> {stale_after_s / 60:.0f}m since last beat)"
    return line


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--follow", action="store_true", help="poll until interrupted")
    parser.add_argument("--interval", type=int, default=60, help="poll seconds (--follow)")
    parser.add_argument(
        "--stale-after",
        type=int,
        default=4 * 3600,
        help="seconds since last beat before flagging STALE (default: 4h — "
        "eval stages legitimately run hours between beats)",
    )
    args = parser.parse_args(argv)

    while True:
        try:
            print(render(fetch(), args.stale_after), flush=True)
        except Exception as exc:  # noqa: BLE001 — watcher reports, never crashes
            print(f"[watch] fetch failed: {exc}", file=sys.stderr, flush=True)
        if not args.follow:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
