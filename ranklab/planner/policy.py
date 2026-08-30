"""Policy loading, budget accounting, and deterministic candidate ranking."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from ranklab.contracts import project_root


class PlannerInputError(ValueError):
    """Raised when one of the three planner inputs is malformed."""


_IMPACT = re.compile(r"primary\s+[-+]?\d+(?:\.\d+)?,\s*([-+]?\d+(?:\.\d+)?)\s+versus", re.I)
_AFFECTED = re.compile(r"(\d+)\s+users\s+and\s+(\d+)\s+rows", re.I)


def default_policy_path() -> Path:
    return project_root() / "ranklab" / "config" / "research_policy.json"


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PlannerInputError(f"Required planner input is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PlannerInputError(f"Invalid JSON in planner input {path}: {exc.msg}") from exc


def load_policy(path: str | Path | None = None) -> dict[str, Any]:
    policy = load_json(Path(path) if path is not None else default_policy_path())
    required = {"max_iterations", "time_budget_minutes", "forbidden_files", "forbidden_terms", "minimum_validation_primary_improvement"}
    if not isinstance(policy, dict) or required.difference(policy):
        raise PlannerInputError("Research policy lacks required budget and safety fields.")
    if int(policy["max_iterations"]) < 0 or float(policy["time_budget_minutes"]) < 0:
        raise PlannerInputError("Research policy budgets must be non-negative.")
    return policy


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PlannerInputError(f"Invalid JSONL in {path} line {line_number}.") from exc
        if not isinstance(value, dict):
            raise PlannerInputError(f"Planner ledger {path} line {line_number} is not an object.")
        records.append(value)
    return records


def remaining_budget(policy: dict[str, Any], iteration_records: Iterable[dict[str, Any]]) -> dict[str, float | int]:
    """Account for every executed run; drafting a planner proposal consumes no budget."""
    executed = list(iteration_records)
    used_minutes = sum(float(record.get("runtime_seconds", 0) or 0) / 60.0 for record in executed)
    return {
        "remaining_iterations": max(0, int(policy["max_iterations"]) - len(executed)),
        "remaining_minutes": max(0.0, float(policy["time_budget_minutes"]) - used_minutes),
    }


def evidence_fingerprint(evidence: Iterable[str]) -> str:
    """Stable comparison key: reordered evidence is still the same evidence."""
    import hashlib

    material = "\n".join(sorted(str(item).strip() for item in evidence))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def opportunity_numbers(opportunity: dict[str, Any]) -> tuple[float, int]:
    evidence = str(opportunity.get("evidence", ""))
    impact_match, affected_match = _IMPACT.search(evidence), _AFFECTED.search(evidence)
    impact = abs(float(impact_match.group(1))) if impact_match else 0.0
    affected = max((int(affected_match.group(1)), int(affected_match.group(2)))) if affected_match else 0
    return impact, affected


def priority(opportunity: dict[str, Any], expected_cost_minutes: int | float) -> float:
    impact, affected = opportunity_numbers(opportunity)
    return impact * affected / float(expected_cost_minutes)


def candidate_was_rejected(candidate: dict[str, Any], history: Iterable[dict[str, Any]]) -> bool:
    """Block a failed family/slice pair until the diagnostic evidence changes."""
    fingerprint = evidence_fingerprint(candidate["evidence"])
    for record in history:
        if record.get("status") not in {"failed", "rejected"}:
            continue
        if record.get("experiment_family", record.get("family")) != candidate["experiment_family"]:
            continue
        if record.get("target_slice", record.get("slice")) != candidate["target_slice"]:
            continue
        # Older ledger entries have no evidence fingerprint, so fail closed.
        previous_evidence = record.get("evidence_fingerprint")
        if previous_evidence is None and record.get("evidence") is not None:
            raw_evidence = record["evidence"]
            previous_evidence = evidence_fingerprint(raw_evidence if isinstance(raw_evidence, list) else [str(raw_evidence)])
        if previous_evidence in {None, fingerprint}:
            return True
    return False
