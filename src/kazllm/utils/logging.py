"""Structured logging with rank awareness for distributed training."""

import json
import logging
import os
import sys
from datetime import datetime


def get_rank() -> int:
    return int(os.environ.get("RANK", 0))


def is_main_process() -> bool:
    return get_rank() == 0


def setup_logging(log_file: str | None = None, level: int = logging.INFO) -> None:
    if not is_main_process():
        logging.basicConfig(level=logging.WARNING)
        return

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
    )


def log_metrics(metrics: dict, step: int, log_file: str) -> None:
    if not is_main_process():
        return
    record = {"step": step, "timestamp": datetime.utcnow().isoformat(), **metrics}
    with open(log_file, "a") as f:
        f.write(json.dumps(record) + "\n")
