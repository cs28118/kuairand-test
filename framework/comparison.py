"""Compare completed supervised experiments without touching hidden test data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import REPO_ROOT
from .state import atomic_write_json, utc_now


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _provenance_confirmed(result: dict[str, Any]) -> bool:
    source = result.get("source_provenance")
    provenance = result.get("provenance")
    if result.get("status") != "completed" or not result.get("git_diff"):
        return False
    if not isinstance(source, dict) or not isinstance(provenance, dict):
        return False
    if not provenance.get("git_diff_sha256"):
        return False
    declared = source.get("declared_modified_files")
    changed = source.get("changed_files")
    original = source.get("original_file_sha256")
    patched = source.get("patched_file_sha256")
    if not all(isinstance(value, (list, dict)) for value in (declared, changed, original, patched)):
        return False
    declared_set = {str(path) for path in declared}
    changed_set = {str(path) for path in changed}
    return bool(declared_set) and declared_set <= changed_set and all(
        path in original and path in patched and original[path] != patched[path]
        for path in declared_set
    )


def _records(runs_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in runs_dir.glob("run-*") if path.is_dir()):
        for result_path in sorted(run_dir.glob("iter_*/result.json")):
            result = _read_json(result_path)
            if result is None:
                continue
            metrics = result.get("metrics")
            primary = metrics.get("primary") if isinstance(metrics, dict) else None
            if not isinstance(primary, (int, float)):
                continue
            artifacts = [str(item) for item in result.get("artifacts", []) if isinstance(item, str)]
            source = result.get("source_provenance")
            provenance = result.get("provenance")
            changed = bool(result.get("git_diff"))
            confirmed = _provenance_confirmed(result)
            checkpoint = next((item for item in artifacts if Path(item).name == "model.npz"), None)
            records.append(
                {
                    "run_id": run_dir.name,
                    "iteration": result_path.parent.name,
                    "experiment": result.get("description", ""),
                    "status": result.get("status"),
                    "primary": float(primary),
                    "metrics": metrics,
                    "seed": result.get("seed"),
                    "code_changed": changed,
                    "provenance_confirmed": confirmed,
                    "provenance": provenance if isinstance(provenance, dict) else {},
                    "checkpoint": checkpoint,
                    "result": str(result_path),
                }
            )
    return records


def summarize_supervised_runs(runs_dir: str | Path | None = None) -> dict[str, Any]:
    """Return validation-only results and select a confirmed best checkpoint."""
    root = Path(runs_dir) if runs_dir else REPO_ROOT / "runs"
    records = _records(root)
    completed = [record for record in records if record["status"] == "completed"]
    best_validation = max(completed, key=lambda record: record["primary"], default=None)
    eligible = [
        record for record in completed
        if record["provenance_confirmed"] and record["checkpoint"]
    ]
    confirmed = [record for record in completed if record["provenance_confirmed"]]
    best_checkpoint = max(eligible, key=lambda record: record["primary"], default=None)
    return {
        "generated_at": utc_now(),
        "scope": "train_validation_only",
        "runs_considered": len(records),
        "completed_runs": len(completed),
        "records": sorted(records, key=lambda record: (-record["primary"], record["run_id"], record["iteration"])),
        "best_validation_result": best_validation,
        "best_provenance_confirmed_result": max(
            confirmed, key=lambda record: record["primary"], default=None
        ),
        "best_provenance_confirmed_checkpoint": best_checkpoint,
    }


def write_summary(runs_dir: str | Path | None = None, output: str | Path | None = None) -> Path:
    root = Path(runs_dir) if runs_dir else REPO_ROOT / "runs"
    target = Path(output) if output else root / "experiment_summary.json"
    summary = summarize_supervised_runs(root)
    atomic_write_json(target, summary)
    selected = summary["best_provenance_confirmed_checkpoint"]
    atomic_write_json(
        root / "best_checkpoint.json",
        selected if selected is not None else {"status": "no_provenance_confirmed_checkpoint"},
    )
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize validation-only supervised runs and select a verified checkpoint.")
    parser.add_argument("--runs-dir", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    path = write_summary(args.runs_dir, args.output)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
