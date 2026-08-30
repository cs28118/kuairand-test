"""Command line entry point for a deterministic, non-executing proposal planner."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ranklab.contracts import project_root

from .catalogue import CATALOGUE
from .policy import (
    PlannerInputError, candidate_was_rejected, evidence_fingerprint, load_json, load_policy,
    priority, read_jsonl, remaining_budget,
)
from .validator import ProposalValidationError, validate_proposal


_WILDCARD = re.compile(r"^" + "{pattern}" + r"$")


def _matches(pattern: str, target_slice: str) -> bool:
    return re.fullmatch(re.escape(pattern).replace(r"\*", ".*"), target_slice) is not None


def _safe_run_id(run_id: str) -> str:
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise PlannerInputError("run_id must be a single, non-empty path component.")
    return run_id


def _proposal_id(run_id: str, target_slice: str, family: str) -> str:
    material = "\0".join((run_id, target_slice, family)).encode("utf-8")
    return "proposal-" + hashlib.sha256(material).hexdigest()[:16]


def _accepted_model_count(history: list[dict[str, Any]]) -> int:
    names = set()
    for record in history:
        if record.get("status") not in {"success", "accepted"}:
            continue
        configuration = record.get("configuration", {})
        if isinstance(configuration, dict) and configuration.get("model"):
            names.add(str(configuration["model"]))
    return len(names)


def _eligible(family: str, opportunity: dict[str, Any], overall: dict[str, Any], history: list[dict[str, Any]]) -> bool:
    target_slice = str(opportunity.get("weak_slice", ""))
    entry = CATALOGUE[family]
    if family == "ranking_loss":
        return float(overall.get("nDCG@5", 0.0)) < float(overall.get("GAUC", 0.0))
    if family == "ensemble" and _accepted_model_count(history) < 2:
        return False
    return any(_matches(pattern, target_slice) for pattern in entry["target_slice_patterns"])


def _planned_changes(family: str, target_slice: str) -> list[str]:
    return {
        "ranking_loss": [f"Add a validation-only ranking-loss challenger for {target_slice}.", "Compare it against the current pointwise baseline using validation primary."],
        "causal_features": [f"Add train-prefix causal features aimed at {target_slice}.", "Audit feature timestamps before validation."],
        "sequence": [f"Add causal user-history representation for {target_slice}.", "Bound history length and validate its primary-metric effect."],
        "multitask": [f"Add an auxiliary feedback objective for {target_slice}.", "Retain validation primary as the acceptance metric."],
        "ensemble": [f"Blend accepted complementary models for {target_slice}.", "Select weights using validation primary only."],
    }[family]


def _candidate(run_id: str, opportunity: dict[str, Any], family: str) -> dict[str, Any]:
    entry = CATALOGUE[family]
    target_slice = str(opportunity["weak_slice"])
    evidence = [str(opportunity["evidence"]), str(opportunity.get("likely_cause", ""))]
    return {
        "proposal_id": _proposal_id(run_id, target_slice, family),
        "run_id": run_id,
        "hypothesis": f"{entry['eligibility'].rstrip('.')}. Addressing {target_slice} may improve validation primary.",
        "target_slice": target_slice,
        "evidence": evidence,
        "experiment_family": family,
        "planned_changes": _planned_changes(family, target_slice),
        "allowed_files": entry["future_allowed_files"],
        "expected_cost_minutes": entry["expected_cost_minutes"],
        "success_criterion": entry["success_criterion"],
        "rollback_condition": entry["rollback_condition"],
        "risks": entry["risks"],
        "execution_authorized": False,
        "priority": priority(opportunity, entry["expected_cost_minutes"]),
    }


def build_proposal(
    run_id: str, *, artifacts_dir: str | Path | None = None, policy_path: str | Path | None = None,
) -> dict[str, Any]:
    """Read the three permitted inputs and return a proposal or no-proposal result."""
    run_id = _safe_run_id(run_id)
    artifacts = Path(artifacts_dir) if artifacts_dir is not None else project_root() / "artifacts"
    report = load_json(artifacts / "reports" / run_id / "diagnostics.json")
    if not isinstance(report, dict) or report.get("run_id") != run_id:
        raise PlannerInputError("Diagnostics report run_id does not match the requested run_id.")
    opportunities = report.get("top_opportunities")
    if not isinstance(opportunities, list):
        raise PlannerInputError("Diagnostics report lacks top_opportunities.")
    policy = load_policy(policy_path)
    iterations = read_jsonl(artifacts / "iterations.jsonl")
    planner_history = read_jsonl(artifacts / "planner_proposals.jsonl")
    history = iterations + planner_history
    budget = remaining_budget(policy, iterations)
    allowed_slices = [str(item.get("weak_slice", "")) for item in opportunities if isinstance(item, dict)]
    candidates: list[dict[str, Any]] = []
    for opportunity in opportunities:
        if not isinstance(opportunity, dict) or not opportunity.get("weak_slice") or not opportunity.get("evidence"):
            continue
        for family in CATALOGUE:
            if not _eligible(family, opportunity, report.get("overall_metrics", {}), history):
                continue
            candidate = _candidate(run_id, opportunity, family)
            if candidate_was_rejected(candidate, history):
                continue
            if candidate["expected_cost_minutes"] > budget["remaining_minutes"] or budget["remaining_iterations"] < 1:
                continue
            candidates.append(candidate)
    if not candidates:
        return {
            "status": "no_eligible_proposal",
            "run_id": run_id,
            "reason": "No eligible catalogue candidate remains within policy and budget.",
            "execution_authorized": False,
        }
    candidates.sort(key=lambda item: (-float(item["priority"]), item["target_slice"], item["experiment_family"]))
    proposal = candidates[0]
    validate_proposal(proposal, allowed_slices=allowed_slices, policy=policy, remaining=budget, history=history)
    return proposal


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if not isinstance(value, dict):
        raise PlannerInputError(f"Planner artifact must be a JSON object: {path}")
    return value


def propose(
    run_id: str, *, artifacts_dir: str | Path | None = None, policy_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build, validate, and persist a non-executable proposal plus an append-only event."""
    artifacts = Path(artifacts_dir) if artifacts_dir is not None else project_root() / "artifacts"
    result = build_proposal(run_id, artifacts_dir=artifacts, policy_path=policy_path)
    event: dict[str, Any] = {"event_type": "planner_proposal", "run_id": run_id, "status": result["status"] if "status" in result else "proposed", "execution_authorized": False}
    if result.get("status") != "no_eligible_proposal":
        proposal_dir = artifacts / "planner_proposals" / result["proposal_id"]
        proposal_path = proposal_dir / "proposal.json"
        validation_path = proposal_dir / "validation.json"
        if proposal_dir.exists():
            if not proposal_path.is_file() or not validation_path.is_file():
                raise PlannerInputError(f"Refusing to overwrite incomplete planner artifact: {proposal_dir}")
            existing = _read_json(proposal_path)
            validation = _read_json(validation_path)
            if existing != result or validation.get("status") != "valid" or validation.get("execution_authorized") is not False:
                raise PlannerInputError(f"Refusing to overwrite conflicting planner artifact: {proposal_dir}")
            return existing
        proposal_dir.mkdir(parents=True)
        _write_json(proposal_path, result)
        validation = {"proposal_id": result["proposal_id"], "status": "valid", "errors": [], "execution_authorized": False}
        _write_json(validation_path, validation)
        event.update({
            "proposal_id": result["proposal_id"], "target_slice": result["target_slice"],
            "experiment_family": result["experiment_family"],
            "evidence_fingerprint": evidence_fingerprint(result["evidence"]),
        })
    else:
        event["reason"] = result["reason"]
    ledger = artifacts / "planner_proposals.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8", newline="\n") as target:
        target.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create one safe, deterministic RankLab experiment proposal.")
    parser.add_argument("--run-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = propose(args.run_id)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
