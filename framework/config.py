"""Configuration loading for the participant-side framework."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "benchmark.json"


@dataclass(frozen=True)
class BenchmarkConfig:
    data_dir: Path
    label: str
    development_splits: tuple[str, ...]
    primary_metric: str
    epsilon: float
    patience: int
    max_iterations: int
    max_wallclock_hours: float
    baseline_expected: dict[str, dict[str, float]]


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_benchmark_config(path: str | Path | None = None) -> BenchmarkConfig:
    """Load the JSON config without requiring a YAML dependency."""
    config_path = _resolve_repo_path(path or DEFAULT_CONFIG_PATH)
    with config_path.open(encoding="utf-8") as handle:
        raw: dict[str, Any] = json.load(handle)

    return BenchmarkConfig(
        data_dir=_resolve_repo_path(raw["data_dir"]),
        label=str(raw["label"]),
        development_splits=tuple(raw["development_splits"]),
        primary_metric=str(raw["primary_metric"]),
        epsilon=float(raw["convergence"]["epsilon"]),
        patience=int(raw["convergence"]["patience"]),
        max_iterations=int(raw["limits"]["max_iterations"]),
        max_wallclock_hours=float(raw["limits"]["max_wallclock_hours"]),
        baseline_expected={
            name: {metric: float(value) for metric, value in metrics.items()}
            for name, metrics in raw["baseline_expected"].items()
        },
    )

