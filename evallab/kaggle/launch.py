#!/usr/bin/env python3
"""Launch the kazeval Kaggle run via the Kaggle API (no browser needed).

Prereqs — evallab/kaggle/.env (gitignored) with:
  HF_TOKEN=hf_...            token of an account that accepted the gated issai/* datasets
  KAGGLE_API_TOKEN=KG...     kaggle.com -> Settings -> API (new-style access token;
                             written to ~/.kaggle/access_token if not already there —
                             requires Kaggle CLI >= 2.x, NOT the legacy 1.x python CLI,
                             whose basic auth 401s on the new api.kaggle.com endpoints).

What it does:
  1. Uploads/updates a PRIVATE Kaggle dataset <user>/qymyzlm-hf-token holding
     hf_token.txt (API-pushed kernels cannot read UserSecretsClient secrets;
     the notebook falls back to this attached dataset). Delete the dataset on
     kaggle.com whenever you rotate the token.
  2. Pushes the notebook as a private GPU+internet kernel (kernel-metadata.json)
     and prints the status/output commands.

Usage:
    python3 evallab/kaggle/launch.py
    kaggle kernels status kekeront/kazeval-planka-me5-large   # poll
    kaggle kernels output kekeront/kazeval-planka-me5-large -p /tmp/kazeval-out
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_env() -> dict[str, str]:
    env = HERE / ".env"
    if not env.exists():
        sys.exit(f"missing {env} (expected HF_TOKEN=... and KAGGLE_API_TOKEN=...)")
    values: dict[str, str] = {}
    for line in env.read_text().splitlines():
        key, _, value = line.strip().partition("=")
        if key and value:
            values[key] = value.strip().strip('"').strip("'")
    if "HF_TOKEN" not in values:
        sys.exit(f"{env} has no HF_TOKEN=... line")
    return values


def run(cmd: list[str], extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd))
    env = {**os.environ, **(extra_env or {})}
    env.pop("PYTHONPATH", None)  # ROS Humble pollutes it with py3.10 packages
    return subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)


def main() -> int:
    meta = json.loads((HERE / "kernel-metadata.json").read_text())
    user = meta["id"].split("/")[0]
    token_ds = f"{user}/qymyzlm-hf-token"
    assert token_ds in meta["dataset_sources"], "kernel metadata must attach the token dataset"

    values = read_env()
    kaggle_env: dict[str, str] = {}
    access_token = Path.home() / ".kaggle" / "access_token"
    if "KAGGLE_API_TOKEN" in values and not access_token.exists():
        access_token.parent.mkdir(exist_ok=True)
        access_token.write_text(values["KAGGLE_API_TOKEN"])
        access_token.chmod(0o600)

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        (tmpdir / "hf_token.txt").write_text(values["HF_TOKEN"] + "\n")
        (tmpdir / "dataset-metadata.json").write_text(
            json.dumps(
                {
                    "title": "qymyzlm-hf-token (private)",
                    "id": token_ds,
                    "licenses": [{"name": "other"}],
                }
            )
        )
        # create is idempotent-ish: if the dataset exists, push a new version
        created = run(["kaggle", "datasets", "create", "-p", str(tmpdir)], kaggle_env)
        print(created.stdout.strip() or created.stderr.strip())
        if "already exists" in (created.stdout + created.stderr):
            versioned = run(
                ["kaggle", "datasets", "version", "-p", str(tmpdir), "-m", "rotate token"],
                kaggle_env,
            )
            print(versioned.stdout.strip() or versioned.stderr.strip())

    # A kernel pushed before the dataset finishes processing starts WITHOUT the
    # attachment (observed: version 1 died on the missing hf_token.txt) — block
    # until Kaggle reports the dataset ready.
    for _ in range(36):
        status = run(["kaggle", "datasets", "status", token_ds], kaggle_env)
        if "ready" in status.stdout:
            break
        time.sleep(5)
    else:
        sys.exit(f"dataset {token_ds} never became ready; not pushing the kernel")

    pushed = run(["kaggle", "kernels", "push", "-p", str(HERE)], kaggle_env)
    print(pushed.stdout.strip() or pushed.stderr.strip())
    if pushed.returncode != 0:
        return pushed.returncode

    print()
    print("Kernel is queued. Poll / fetch results with:")
    print(f"  kaggle kernels status {meta['id']}")
    print(f"  kaggle kernels output {meta['id']} -p /tmp/kazeval-out")
    print("then copy /tmp/kazeval-out/results/*.json into evallab/results/ and run:")
    print("  python -m kazeval.leaderboard")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
