"""Atomic file I/O and shard path conventions."""

import hashlib
import json
import os
import tempfile
from pathlib import Path


def atomic_write_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, suffix=".tmp") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        tmp_path = f.name
    os.replace(tmp_path, path)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def shard_path(base_dir: str | Path, shard_idx: int) -> Path:
    return Path(base_dir) / f"shard-{shard_idx:05d}.bin"


def manifest_path(base_dir: str | Path) -> Path:
    return Path(base_dir) / "manifest.json"
