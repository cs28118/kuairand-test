"""Prepare and validate AIDE research proposals without executing them.

This module deliberately has no patching, training, or submission capability.
Those actions belong to the later experiment runner and must consume a validated
hypothesis artifact first.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..contracts import (
    DEVELOPMENT_SPLITS,
    EPSILON,
    MAX_ITERATIONS,
    OFFICIAL_EVALUATOR_SHA256,
    PATIENCE_ITERATIONS,
    PRIMARY_METRIC,
    project_root,
)


class HypothesisValidationError(ValueError):
    """An AIDE response does not meet RankLab's safe proposal protocol."""


REQUIRED_HYPOTHESIS_FIELDS = frozenset({
    "hypothesis_id", "hypothesis", "evidence", "likely_cause", "proposed_experiment",
    "expected_outcome", "risk_assessment", "data_policy", "execution_authorized",
})
REQUIRED_EXPERIMENT_FIELDS = frozenset({"family", "target_split", "code_changes", "validation_plan", "rollback_plan"})
REQUIRED_CHANGE_FIELDS = frozenset({"path", "intent"})
FORBIDDEN_CHANGE_PATHS = {"evaluate.py", "submit.py"}


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HypothesisValidationError(f"Invalid JSON in {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise HypothesisValidationError(f"{label} must be a JSON object.")
    return value


def _load_prior_experiments(artifacts_dir: Path) -> list[dict[str, Any]]:
    ledger = artifacts_dir / "iterations.jsonl"
    if not ledger.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if line.strip():
            parsed = json.loads(line)
            # Keep the agent context focused and do not pass arbitrary local paths onward.
            records.append({
                key: parsed.get(key) for key in (
                    "run_id", "hypothesis", "configuration", "seed", "runtime_seconds",
                    "GAUC", "nDCG@5", "primary", "status", "error_recovery_note",
                )
            })
    return records


def build_aide_request(
    run_id: str, *, iteration_budget: int = MAX_ITERATIONS,
    wall_clock_budget_minutes: int = 360, artifacts_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build the complete, test-free context envelope for an AIDE proposal."""
    if iteration_budget < 1 or wall_clock_budget_minutes < 1:
        raise ValueError("Both AIDE budgets must be positive.")
    artifacts = Path(artifacts_dir) if artifacts_dir is not None else project_root() / "artifacts"
    diagnostics = _read_json(artifacts / "reports" / run_id / "diagnostics.json", "diagnostics report")
    if diagnostics.get("run_id") != run_id:
        raise HypothesisValidationError("Diagnostics run_id does not match the requested AIDE run_id.")
    if diagnostics.get("data_policy") != {"used_splits": ["train", "valid"], "test_accessed": False}:
        raise HypothesisValidationError("Diagnostics report does not prove the required train/validation-only policy.")
    prior_experiments = _load_prior_experiments(artifacts)
    return {
        "protocol_version": "ranklab-aide-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "baseline_run_id": run_id,
        "research_contract": {
            "allowed_splits": list(DEVELOPMENT_SPLITS),
            "forbidden": [
                "Do not access hidden test rows, labels, predictions, or scores.",
                "Do not modify evaluate.py or redefine the official metric.",
                "Do not use validation positives per user as a training feature.",
                "Return a hypothesis only; do not apply a patch or start a run.",
            ],
            "official_primary_metric": PRIMARY_METRIC,
            "evaluator_sha256": OFFICIAL_EVALUATOR_SHA256,
            "convergence": {"epsilon": EPSILON, "patience_iterations": PATIENCE_ITERATIONS},
        },
        "diagnostics": diagnostics,
        "prior_experiments": prior_experiments,
        "budget": {
            "iteration_cap": iteration_budget,
            "iterations_already_logged": len(prior_experiments),
            "iterations_remaining": max(0, iteration_budget - len(prior_experiments)),
            "wall_clock_budget_minutes": wall_clock_budget_minutes,
        },
        "required_response_schema": {
            "hypothesis_id": "non-empty string",
            "hypothesis": "non-empty string",
            "evidence": ["one or more diagnostic observations"],
            "likely_cause": "non-empty string",
            "proposed_experiment": {
                "family": "non-empty string",
                "target_split": "valid",
                "code_changes": [{"path": "relative source path", "intent": "non-empty description"}],
                "validation_plan": "non-empty string",
                "rollback_plan": "non-empty string",
            },
            "expected_outcome": "non-empty string",
            "risk_assessment": "non-empty string",
            "data_policy": {"used_splits": ["train", "valid"], "test_accessed": False},
            "execution_authorized": False,
        },
    }


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HypothesisValidationError(f"{field} must be a non-empty string.")
    return value


def _validate_change(change: Any, index: int) -> dict[str, str]:
    if not isinstance(change, dict) or REQUIRED_CHANGE_FIELDS.difference(change):
        raise HypothesisValidationError(f"proposed_experiment.code_changes[{index}] must contain path and intent.")
    path = _require_nonempty_string(change["path"], f"code_changes[{index}].path").replace("\\", "/")
    _require_nonempty_string(change["intent"], f"code_changes[{index}].intent")
    normalized = Path(path)
    if normalized.is_absolute() or ".." in normalized.parts or path in FORBIDDEN_CHANGE_PATHS:
        raise HypothesisValidationError(f"Unsafe proposed code-change path: {path!r}")
    if path.startswith("artifacts/") or path.startswith("KuaiRand-Pure/"):
        raise HypothesisValidationError(f"AIDE may not propose changing data or generated artifacts: {path!r}")
    return {"path": path, "intent": change["intent"].strip()}


def validate_hypothesis(response: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an AIDE proposal and return a normalized, execution-blocked object."""
    if not isinstance(response, Mapping):
        raise HypothesisValidationError("AIDE response must be a JSON object.")
    missing = REQUIRED_HYPOTHESIS_FIELDS.difference(response)
    if missing:
        raise HypothesisValidationError(f"AIDE response is missing fields: {sorted(missing)}")
    normalized = dict(response)
    for field in ("hypothesis_id", "hypothesis", "likely_cause", "expected_outcome", "risk_assessment"):
        normalized[field] = _require_nonempty_string(response[field], field).strip()
    evidence = response["evidence"]
    if not isinstance(evidence, list) or not evidence:
        raise HypothesisValidationError("evidence must be a non-empty list.")
    normalized["evidence"] = [_require_nonempty_string(item, "evidence entry").strip() for item in evidence]
    if response["data_policy"] != {"used_splits": ["train", "valid"], "test_accessed": False}:
        raise HypothesisValidationError("AIDE hypothesis must explicitly acknowledge train/validation-only data policy.")
    if response["execution_authorized"] is not False:
        raise HypothesisValidationError("AIDE responses must set execution_authorized to false; later runner approval is required.")
    experiment = response["proposed_experiment"]
    if not isinstance(experiment, Mapping):
        raise HypothesisValidationError("proposed_experiment must be an object.")
    missing_experiment = REQUIRED_EXPERIMENT_FIELDS.difference(experiment)
    if missing_experiment:
        raise HypothesisValidationError(f"proposed_experiment is missing fields: {sorted(missing_experiment)}")
    if experiment["target_split"] != "valid":
        raise HypothesisValidationError("AIDE proposals may target validation only.")
    changes = experiment["code_changes"]
    if not isinstance(changes, list) or not changes:
        raise HypothesisValidationError("proposed_experiment.code_changes must be a non-empty list.")
    normalized["proposed_experiment"] = {
        "family": _require_nonempty_string(experiment["family"], "proposed_experiment.family").strip(),
        "target_split": "valid",
        "code_changes": [_validate_change(change, index) for index, change in enumerate(changes)],
        "validation_plan": _require_nonempty_string(experiment["validation_plan"], "proposed_experiment.validation_plan").strip(),
        "rollback_plan": _require_nonempty_string(experiment["rollback_plan"], "proposed_experiment.rollback_plan").strip(),
    }
    normalized["data_policy"] = {"used_splits": ["train", "valid"], "test_accessed": False}
    normalized["execution_authorized"] = False
    normalized["validation_status"] = "validated_hypothesis_only"
    return normalized


def _dispatch(command: str, request: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
    """Send the request via stdin and accept exactly one JSON response on stdout."""
    result = subprocess.run(
        shlex.split(command, posix=False), input=json.dumps(request), text=True,
        capture_output=True, timeout=timeout_seconds, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"AIDE command failed ({result.returncode}): {result.stderr.strip()}")
    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise HypothesisValidationError("AIDE command stdout must contain exactly one JSON object.") from exc
    if not isinstance(response, dict):
        raise HypothesisValidationError("AIDE command response must be a JSON object.")
    return response


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create or validate an execution-blocked AIDE research hypothesis.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--iteration-budget", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--wall-clock-budget-minutes", type=int, default=360)
    parser.add_argument("--aide-command", help="Optional command: receives request JSON on stdin and emits proposal JSON on stdout.")
    parser.add_argument("--response-file", help="Existing AIDE JSON response to validate instead of dispatching a command.")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.aide_command and args.response_file:
        raise SystemExit("Use one of --aide-command or --response-file, not both.")
    artifacts = project_root() / "artifacts"
    request = build_aide_request(
        args.run_id, iteration_budget=args.iteration_budget,
        wall_clock_budget_minutes=args.wall_clock_budget_minutes, artifacts_dir=artifacts,
    )
    destination = artifacts / "aide" / args.run_id
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "request.json").write_text(json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.aide_command and not args.response_file:
        print(f"wrote AIDE request: {destination / 'request.json'}")
        return 0
    response = _dispatch(args.aide_command, request, args.timeout_seconds) if args.aide_command else _read_json(Path(args.response_file), "AIDE response")
    try:
        validated = validate_hypothesis(response)
    except HypothesisValidationError as exc:
        (destination / "rejected_hypothesis.json").write_text(json.dumps(response, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise SystemExit(f"AIDE hypothesis rejected: {exc}") from exc
    (destination / "validated_hypothesis.json").write_text(json.dumps(validated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"validated hypothesis only (no patch or run executed): {destination / 'validated_hypothesis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
