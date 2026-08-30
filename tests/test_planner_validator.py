from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ranklab.planner.propose import build_proposal
from ranklab.planner.validator import ProposalValidationError, validate_proposal


def _proposal() -> dict:
    return {
        "proposal_id": "proposal-a", "run_id": "run-a", "hypothesis": "Safe validation-only change.",
        "target_slice": "content_tag=rare", "evidence": ["primary 0.4, -0.2 versus overall validation primary 0.6; 1 users and 2 rows."],
        "experiment_family": "causal_features", "planned_changes": ["Add causal feature."],
        "allowed_files": ["ranklab/features/causal_features.py"], "expected_cost_minutes": 30,
        "success_criterion": "Validation primary improves by at least 0.002.",
        "rollback_condition": "Reject if validation primary does not improve.", "risks": ["Causality audit."],
        "execution_authorized": False,
    }


def _policy() -> dict:
    return {
        "forbidden_files": ["evaluate.py", "data.py", "baseline.py", "submit.py"],
        "forbidden_terms": ["test data", "leaderboard", "submission scoring", "external data"],
    }


class PlannerValidatorTests(unittest.TestCase):
    def test_unknown_and_unsafe_proposals_are_rejected(self) -> None:
        for field, value in (("experiment_family", "unknown"), ("allowed_files", ["evaluate.py"]), ("allowed_files", ["ranklab/planner/propose.py"]), ("hypothesis", "Use external data")):
            proposal = _proposal()
            proposal[field] = value
            with self.subTest(field=field), self.assertRaises(ProposalValidationError):
                validate_proposal(proposal, allowed_slices={"content_tag=rare"}, policy=_policy(), remaining={"remaining_iterations": 1, "remaining_minutes": 30})

    def test_absent_slice_budget_repeat_and_missing_criteria_are_rejected(self) -> None:
        proposal = _proposal()
        with self.assertRaises(ProposalValidationError):
            validate_proposal(proposal, allowed_slices=set(), policy=_policy(), remaining={"remaining_iterations": 1, "remaining_minutes": 30})
        with self.assertRaises(ProposalValidationError):
            validate_proposal(proposal, allowed_slices={proposal["target_slice"]}, policy=_policy(), remaining={"remaining_iterations": 0, "remaining_minutes": 30})
        with self.assertRaises(ProposalValidationError):
            validate_proposal(proposal, allowed_slices={proposal["target_slice"]}, policy=_policy(), remaining={"remaining_iterations": 1, "remaining_minutes": 30}, history=[{"status": "rejected", "experiment_family": "causal_features", "target_slice": "content_tag=rare"}])
        proposal["success_criterion"] = "Improve the score."
        with self.assertRaises(ProposalValidationError):
            validate_proposal(proposal, allowed_slices={proposal["target_slice"]}, policy=_policy(), remaining={"remaining_iterations": 1, "remaining_minutes": 30})


if __name__ == "__main__":
    unittest.main()
