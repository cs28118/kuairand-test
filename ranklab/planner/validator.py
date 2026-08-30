"""Fail-closed validation for untrusted RankLab planner proposals."""
from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any, Iterable

from .catalogue import CATALOGUE
from .policy import candidate_was_rejected


class ProposalValidationError(ValueError):
    """Raised when a proposal violates the RankLab planning contract."""


def validate_proposal(
    proposal: dict[str, Any], *, allowed_slices: Iterable[str], policy: dict[str, Any],
    remaining: dict[str, float | int], history: Iterable[dict[str, Any]] = (),
) -> None:
    """Validate a proposal without loading data, code, or model artifacts."""
    required = {
        "proposal_id", "run_id", "hypothesis", "target_slice", "evidence", "experiment_family",
        "planned_changes", "allowed_files", "expected_cost_minutes", "success_criterion",
        "rollback_condition", "risks", "execution_authorized",
    }
    missing = sorted(required.difference(proposal))
    if missing:
        raise ProposalValidationError(f"Proposal lacks required fields: {missing}")
    if proposal["target_slice"] not in set(allowed_slices):
        raise ProposalValidationError("Proposal target slice is absent from diagnostic opportunities.")
    family = proposal["experiment_family"]
    if family not in CATALOGUE:
        raise ProposalValidationError(f"Unknown experiment family: {family!r}.")
    if bool(proposal["execution_authorized"]):
        raise ProposalValidationError("Planner proposals cannot authorize execution.")
    cost = float(proposal["expected_cost_minutes"])
    if int(remaining["remaining_iterations"]) < 1 or cost > float(remaining["remaining_minutes"]):
        raise ProposalValidationError("Proposal exceeds the remaining research budget.")
    candidate = {
        "experiment_family": family,
        "target_slice": proposal["target_slice"],
        "evidence": proposal["evidence"],
    }
    if candidate_was_rejected(candidate, history):
        raise ProposalValidationError("Proposal repeats a failed experiment family and target slice.")
    text = "\n".join(_text_values(proposal)).lower()
    forbidden_terms = [str(term).lower() for term in policy["forbidden_terms"]]
    if any(term in text for term in forbidden_terms) or re.search(r"\blabels?\b", text):
        raise ProposalValidationError("Proposal mentions prohibited test, label, leaderboard, submission, or external-data material.")
    allowed_files = proposal["allowed_files"]
    if not isinstance(allowed_files, list) or not allowed_files or not all(isinstance(item, str) and item for item in allowed_files):
        raise ProposalValidationError("Proposal allowed_files must be a non-empty list of paths.")
    normalized_files = [item.replace("\\", "/") for item in allowed_files]
    expected_files = {str(item).replace("\\", "/") for item in CATALOGUE[family]["future_allowed_files"]}
    if len(normalized_files) != len(set(normalized_files)) or set(normalized_files) != expected_files:
        raise ProposalValidationError("Proposal allowed_files must exactly match the selected catalogue allowlist.")
    forbidden_files = {str(item).replace("\\", "/") for item in policy["forbidden_files"]}
    for normalized in normalized_files:
        path = PurePosixPath(normalized)
        if normalized in forbidden_files or ".." in path.parts or path.name in forbidden_files:
            raise ProposalValidationError(f"Proposal targets forbidden file {normalized!r}.")
    success, rollback = proposal["success_criterion"].lower(), proposal["rollback_condition"].lower()
    if "validation primary" not in success or "0.002" not in success:
        raise ProposalValidationError("Proposal needs a validation-primary improvement success criterion.")
    if "reject" not in rollback or "validation primary" not in rollback:
        raise ProposalValidationError("Proposal needs a validation-primary rollback condition.")


def _text_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _text_values(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _text_values(nested)
