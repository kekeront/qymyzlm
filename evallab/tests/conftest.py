"""Offline test harness: no network, kazeval importable from the src layout."""

import os
import sys
from pathlib import Path

# All tests run fully offline (contract: fixtures only, no dataset/model pulls).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# The shared venv may not have kazeval installed yet (orchestrator syncs later).
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
