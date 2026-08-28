"""Run one human-supervised experiment in Docker and preserve its history."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

from .accounting import Budget, BudgetExceeded, CostLedger, TokenUsage
from .benchmark import load_development_splits, run_baseline
from .config import BenchmarkConfig, REPO_ROOT, load_benchmark_config
from .contracts import ExperimentResult, ExperimentSpec, load_experiment_spec
from .guardrails import GuardrailViolation, verify_official_files
from .isolation import DockerExecutor, DockerWorkspace, ExecutionOutcome, IsolationError
from .state import RunStore


def _budget(config: BenchmarkConfig) -> Budget:
    raw = config.budgets
    return Budget(
        max_run_tokens=raw.get("max_run_tokens"),
        max_total_tokens=raw.get("max_total_tokens"),
        max_run_cost=raw.get("max_run_cost"),
        max_total_cost=raw.get("max_total_cost"),
    )


def _usage(raw: dict[str, Any]) -> TokenUsage:
    return TokenUsage(
        model=raw.get("model"),
        input_tokens=int(raw.get("input_tokens", 0)),
        output_tokens=int(raw.get("output_tokens", 0)),
        cached_tokens=int(raw.get("cached_tokens", 0)),
        estimated_cost=float(raw.get("estimated_cost", 0.0)),
    )


def _baseline_primary(config: BenchmarkConfig) -> float:
    splits = load_development_splits(config.data_dir)
    return float(run_baseline("fm", splits, seed=0, verbose=False).metrics[config.primary_metric])


def _copy_artifacts(workspace: Path, output_dir: Path, names: list[str]) -> list[str]:
    copied: list[str] = []
    destination_root = output_dir / "artifacts"
    for name in names:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            continue
        source = workspace / relative
        if not source.is_file():
            continue
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(str(destination))
    return copied


def _failure(status: str, reason: str, *, modified_files: list[str] | None = None) -> ExperimentResult:
    return ExperimentResult(
        status=status,
        failure_reason=reason,
        modified_files=modified_files or [],
    )


def _attach_instruction(result: ExperimentResult, spec: ExperimentSpec) -> ExperimentResult:
    result.hypothesis = spec.hypothesis
    result.git_diff = spec.git_diff
    result.description = spec.description
    result.command = list(spec.command)
    result.seed = spec.seed
    result.result_compare = result.result_compare or spec.result_compare
    result.next_steps = result.next_steps or spec.next_steps
    return result


def run_pilot(
    spec: ExperimentSpec,
    *,
    config: BenchmarkConfig,
    run_store: RunStore,
    baseline_primary: float | None = None,
    repo_root: str | Path = REPO_ROOT,
) -> ExperimentResult:
    """Execute one instruction and append a durable supervised iteration."""
    evaluator_hashes = verify_official_files()
    run_store.initialize(
        {
            "framework_version": 2,
            "mode": "supervised_pilot",
            "data_dir": str(config.data_dir),
            "development_splits": list(config.development_splits),
            "evaluator_sha256": evaluator_hashes,
            "primary_metric": config.primary_metric,
            "experiment_spec": spec.to_dict(),
        }
    )
    started = time.perf_counter()
    iteration = int(run_store.read_state()["iterations"]) + 1
    artifact_dir = run_store.run_dir / f"iter_{iteration:03d}"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    workspace = DockerWorkspace(repo_root, artifact_dir / "isolation")
    outcome: ExecutionOutcome | None = None
    result = _attach_instruction(_failure("failed", "experiment did not start"), spec)
    try:
        existing = run_store.read_state().get("usage", {})
        ledger = CostLedger(
            _budget(config),
            total=TokenUsage(
                input_tokens=int(existing.get("input_tokens", 0)),
                output_tokens=int(existing.get("output_tokens", 0)),
                cached_tokens=int(existing.get("cached_tokens", 0)),
                estimated_cost=float(existing.get("estimated_cost", 0.0)),
            ),
        )
        ledger.require_capacity()
        workspace.prepare(spec)
        execution = config.execution
        executor = DockerExecutor(
            image=str(execution.get("docker_image", "python:3.12-slim")),
            timeout_seconds=float(execution.get("timeout_seconds", 1800)),
            memory_limit=str(execution.get("memory_limit", "4g")),
            cpus=float(execution.get("cpus", 2.0)),
            docker_executable=str(execution.get("docker_executable", "docker")),
        )
        outcome = executor.execute(spec, workspace)
        (artifact_dir / "stdout.txt").write_text(outcome.stdout, encoding="utf-8")
        (artifact_dir / "stderr.txt").write_text(outcome.stderr, encoding="utf-8")
        result_source = workspace.path / spec.result_file
        if outcome.timed_out:
            result = _failure("timed_out", outcome.failure_reason or "experiment timed out", modified_files=outcome.modified_files)
        elif not result_source.is_file():
            result = _failure(
                "failed",
                "experiment did not produce the required result file: " + spec.result_file,
                modified_files=outcome.modified_files,
            )
        else:
            try:
                result = ExperimentResult.from_dict(json.loads(result_source.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                result = _failure("failed", f"invalid experiment result: {exc}", modified_files=outcome.modified_files)
            if outcome.returncode != 0:
                result.status = "failed"
                result.failure_reason = outcome.failure_reason or "experiment failed"
            result.modified_files = sorted(set(result.modified_files) | set(outcome.modified_files))
            if result.status == "completed" and config.primary_metric not in result.metrics:
                result.status = "failed"
                result.failure_reason = f"result is missing required metric: {config.primary_metric}"

        usage = _usage(result.token_usage)
        try:
            ledger.record(usage)
            run_store.record_usage(usage.to_dict())
        except BudgetExceeded as exc:
            ledger.record_observed(usage)
            run_store.record_usage(usage.to_dict())
            result.status = "stopped"
            result.failure_reason = str(exc)
            result.stop_reason = "token or cost budget reached"

        result.result_compare = result.result_compare or spec.result_compare
        result.next_steps = result.next_steps or spec.next_steps
        result.hypothesis = spec.hypothesis
        result.git_diff = spec.git_diff
        result.command = list(spec.command)
        result.seed = spec.seed
        primary = result.metrics.get(config.primary_metric)
        if primary is not None and baseline_primary is not None:
            result.beats_fm = float(primary) > baseline_primary
            result.result_compare = (
                f"{config.primary_metric}={float(primary):.6f}; FM={baseline_primary:.6f}; "
                + ("beats FM" if result.beats_fm else "does not beat FM")
            )
        result.stdout = str(artifact_dir / "stdout.txt")
        result.stderr = str(artifact_dir / "stderr.txt")
        copied = _copy_artifacts(workspace.path, artifact_dir, [spec.result_file, *spec.artifacts, *result.artifacts])
        result.artifacts = sorted(set(copied))
    except BudgetExceeded as exc:
        modified = workspace.modified_files() if workspace.path.exists() else []
        result = _attach_instruction(_failure("stopped", str(exc), modified_files=modified), spec)
        result.stop_reason = "token or cost budget reached"
        (artifact_dir / "stderr.txt").write_text(str(exc) + "\n", encoding="utf-8")
        result.stderr = str(artifact_dir / "stderr.txt")
    except (IsolationError, GuardrailViolation, OSError, ValueError) as exc:
        modified = workspace.modified_files() if workspace.path.exists() else []
        result = _attach_instruction(_failure("failed", str(exc), modified_files=modified), spec)
        (artifact_dir / "stderr.txt").write_text(str(exc) + "\n", encoding="utf-8")
        result.stderr = str(artifact_dir / "stderr.txt")
    finally:
        if workspace.path.exists():
            final_modified = workspace.modified_files()
        else:
            final_modified = []
        run_store.append_audit(
            {
                "event": "supervised_experiment",
                "status": result.status,
                "command": list(spec.command),
                "modified_files": final_modified,
                "failure_reason": result.failure_reason,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "recovery_attempts": [],
            }
        )
        workspace.cleanup()

    result.write_json(artifact_dir / "result.json")
    record = run_store.append_iteration(
        {
            "experiment": spec.metadata.get("name", "supervised-experiment"),
            "kind": "supervised",
            "hypothesis": spec.hypothesis,
            "description": spec.description,
            "git_diff": spec.git_diff,
            "command": list(spec.command),
            "result_compare": result.result_compare,
            "next_steps": result.next_steps,
            "status": result.status,
            "metrics": result.metrics,
            "elapsed_seconds": time.perf_counter() - started,
            "seed": spec.seed,
            "failure_reason": result.failure_reason,
            "artifacts": result.artifacts + [str(artifact_dir / "result.json")],
            "modified_files": result.modified_files,
            "recovery_attempts": result.recovery_attempts,
            "token_usage": result.token_usage,
        }
    )
    run_store.append_audit({"event": "iteration_persisted", "iteration": record["iteration"], "status": result.status})
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one human-supervised experiment in Docker.")
    parser.add_argument("--spec", required=True, help="JSON ExperimentSpec instruction")
    parser.add_argument("--config", default=None)
    parser.add_argument("--runs-dir", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--baseline-primary", type=float, default=None)
    args = parser.parse_args(argv)
    try:
        config = load_benchmark_config(args.config)
        spec = load_experiment_spec(args.spec)
        store = RunStore(args.runs_dir, args.run_id)
        baseline = args.baseline_primary if args.baseline_primary is not None else _baseline_primary(config)
        result = run_pilot(spec, config=config, run_store=store, baseline_primary=baseline)
        store.complete("completed" if result.status == "completed" else result.status)
    except Exception as exc:
        print(f"framework pilot failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.status in {"completed", "stopped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
