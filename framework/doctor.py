"""Read-only environment and benchmark readiness diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

import numpy as np

from .config import load_benchmark_config
from .guardrails import verify_official_files


REQUIRED_DATA_FILES = (
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
    "video_features_basic_pure.csv",
)


def diagnose(config_path: str | Path | None = None) -> dict[str, object]:
    config = load_benchmark_config(config_path)
    missing = [name for name in REQUIRED_DATA_FILES if not (config.data_dir / name).is_file()]
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "data_dir": str(config.data_dir),
        "missing_data_files": missing,
        "official_files": verify_official_files(),
        "development_splits": list(config.development_splits),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check framework prerequisites without running training.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    try:
        report = diagnose(args.config)
    except Exception as exc:
        print(f"framework doctor failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not report["missing_data_files"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

