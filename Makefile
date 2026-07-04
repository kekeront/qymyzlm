# QymyzLM engine — only targets whose scripts exist on disk.
# Workspace venv lives at .venv/ (uv workspace root; shared with embed/ and evallab/).

.PHONY: install lint test benchmark benchmark-quick qlora

PY := .venv/bin/python
RUFF := .venv/bin/ruff

TOKENS ?= 100_000_000

install:
	uv sync --all-packages --all-extras

lint:
	$(RUFF) check src scripts tests
	$(RUFF) format --check src scripts tests

# PYTHONPATH is cleared: system/ROS dist-packages on PYTHONPATH break pytest
# plugin autoload (e.g. ROS launch_testing pulls in a missing 'lark').
test:
	PYTHONPATH= $(PY) -m pytest tests -q

# --- Real pipelines ---

# KazMMLU 3-shot baseline benchmark (dev-limited: 3 exemplars/subject;
# downloads models + MBZUAI/KazMMLU)
benchmark:
	$(PY) scripts/benchmark_baselines.py

benchmark-quick:
	$(PY) scripts/benchmark_baselines.py --quick

# 4-bit QLoRA continual pretraining on streamed Kazakh data (Kaggle T4/P100 OK)
qlora:
	$(PY) scripts/qlora_continual.py --tokens $(TOKENS)
