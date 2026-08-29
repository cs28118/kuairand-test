from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ranklab.aide.adapter import HypothesisValidationError, build_aide_request, validate_hypothesis


def _proposal() -> dict:
    return {
        "hypothesis_id": "pairwise-001",
        "hypothesis": "A pairwise objective will improve within-user ordering.",
        "evidence": ["Cold-item diagnostic primary is below overall validation primary."],
        "likely_cause": "Pointwise loss is mismatched to GAUC and nDCG.",
        "proposed_experiment": {
            "family": "within-user pairwise ranking",
            "target_split": "valid",
            "code_changes": [{"path": "ranklab/models/pairwise.py", "intent": "Add a train-only pairwise objective."}],
            "validation_plan": "Run validation and compare official metrics to the baseline.",
            "rollback_plan": "Revert the isolated experiment if it fails preflight or regresses.",
        },
        "expected_outcome": "Improve validation primary by at least epsilon.",
        "risk_assessment": "Pair sampling can increase runtime.",
        "data_policy": {"used_splits": ["train", "valid"], "test_accessed": False},
        "execution_authorized": False,
    }


class AideAdapterTests(unittest.TestCase):
    def test_build_request_requires_validation_only_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            artifacts = Path(temp)
            report = artifacts / "reports" / "run-1"
            report.mkdir(parents=True)
            (report / "diagnostics.json").write_text(json.dumps({"run_id": "run-1", "data_policy": {"used_splits": ["train", "valid"], "test_accessed": False}}), encoding="utf-8")
            request = build_aide_request("run-1", artifacts_dir=artifacts)
            self.assertEqual(request["research_contract"]["allowed_splits"], ["train", "valid"])
            self.assertEqual(request["budget"]["iterations_remaining"], 50)

    def test_valid_proposal_stays_execution_blocked(self) -> None:
        validated = validate_hypothesis(_proposal())
        self.assertFalse(validated["execution_authorized"])
        self.assertEqual(validated["validation_status"], "validated_hypothesis_only")

    def test_test_target_and_unsafe_path_are_rejected(self) -> None:
        test_target = _proposal()
        test_target["proposed_experiment"]["target_split"] = "test"
        with self.assertRaisesRegex(HypothesisValidationError, "validation only"):
            validate_hypothesis(test_target)
        unsafe_path = _proposal()
        unsafe_path["proposed_experiment"]["code_changes"][0]["path"] = "evaluate.py"
        with self.assertRaisesRegex(HypothesisValidationError, "Unsafe"):
            validate_hypothesis(unsafe_path)


if __name__ == "__main__":
    unittest.main()
