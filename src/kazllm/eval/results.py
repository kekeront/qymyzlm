"""Evaluation result types and JSON serialisation."""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class BenchmarkResult:
    benchmark: str
    metric: str
    value: float
    num_fewshot: int
    num_examples: int


@dataclass
class EvalRun:
    model_path: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    results: list[BenchmarkResult] = field(default_factory=list)

    def add(self, result: BenchmarkResult) -> None:
        self.results.append(result)

    def save(self, output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "results.json"
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
        return path
