"""Preflight checks and optional baseline reproduction for a new agent run."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .config import load_benchmark_config
from .doctor import diagnose
from .runner import execute_baseline
from .state import RunStore


def _within_tolerance(actual: dict[str, float], expected: dict[str, float], tolerance: float) -> bool:
    return all(abs(actual[metric] - target) <= tolerance for metric, target in expected.items())


def run_preflight(*, run_baselines: bool, tolerance: float, config_path: str | None = None) -> dict[str, Any]:
    config = load_benchmark_config(config_path)
    report: dict[str, Any] = {"environment": diagnose(config_path), "baselines": {}}
    if report["environment"]["missing_data_files"]:
        return report
    if not run_baselines:
        return report

    store = RunStore()
    for name in ("random", "pop", "fm"):
        record = execute_baseline(name, config=config, run_store=store, verbose=False)
        expected = config.baseline_expected[name]
        report["baselines"][name] = {
            "metrics": record["metrics"],
            "expected": expected,
            "within_tolerance": _within_tolerance(record["metrics"], expected, tolerance),
        }
    store.complete("preflight-completed")
    report["run_id"] = store.run_id
    report["passed"] = all(item["within_tolerance"] for item in report["baselines"].values())
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run framework preflight checks.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-baselines", action="store_true")
    parser.add_argument("--tolerance", type=float, default=0.02)
    args = parser.parse_args(argv)
    try:
        report = run_preflight(
            run_baselines=args.run_baselines,
            tolerance=args.tolerance,
            config_path=args.config,
        )
    except Exception as exc:
        print(f"framework preflight failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("passed", not report["environment"]["missing_data_files"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())

