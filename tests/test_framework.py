import math
from pathlib import Path
import unittest

from framework.config import load_benchmark_config
from framework.guardrails import GuardrailViolation, reject_protected_paths, validate_scores, verify_official_files


class FrameworkTests(unittest.TestCase):
    def test_benchmark_contract(self) -> None:
        config = load_benchmark_config()
        self.assertEqual(config.label, "long_view")
        self.assertEqual(config.development_splits, ("train", "valid"))
        self.assertEqual(config.primary_metric, "primary")

    def test_official_evaluator_fingerprint(self) -> None:
        fingerprints = verify_official_files()
        self.assertIn("evaluate.py", fingerprints)

    def test_protected_evaluator_cannot_be_targeted(self) -> None:
        with self.assertRaises(GuardrailViolation):
            reject_protected_paths([Path("evaluate.py")])

    def test_scores_must_be_finite_and_aligned(self) -> None:
        validate_scores([0.0, 1.0], expected_length=2)
        with self.assertRaises(GuardrailViolation):
            validate_scores([0.0], expected_length=2)
        with self.assertRaises(GuardrailViolation):
            validate_scores([0.0, math.nan], expected_length=2)


if __name__ == "__main__":
    unittest.main()
