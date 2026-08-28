import math
from pathlib import Path
import tempfile
import unittest

from framework.accounting import Budget, BudgetExceeded, CostLedger, TokenUsage
from framework.config import load_benchmark_config
from framework.contracts import ExperimentResult, ExperimentSpec
from framework.dependencies import DependencyViolation, request_profile
from framework.guardrails import GuardrailViolation, reject_protected_paths, validate_scores, verify_official_files
from framework.isolation import DockerExecutor, DockerWorkspace, IsolationError
from framework.pilot import run_pilot
from framework.state import RunStore
from framework.stopping import StoppingPolicy


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

    def test_experiment_instruction_contract_round_trips(self) -> None:
        spec = ExperimentSpec.from_dict(
            {
                "hypothesis": "Try a pairwise loss.",
                "code_change": "diff --git a/train.py b/train.py",
                "description": "Train and score one candidate.",
                "result_comparison": "Compare primary with FM.",
                "what_to_do_next": "Keep it if it wins.",
                "command": ["python", "train.py"],
                "seed": 7,
            }
        )
        self.assertEqual(spec.git_diff, "diff --git a/train.py b/train.py")
        self.assertEqual(spec.to_dict()["command"], ["python", "train.py"])
        result = ExperimentResult.from_dict(
            {"status": "completed", "hypothesis": "h", "git_diff": "diff", "command": ["python"], "seed": 7, "metrics": {"primary": 0.61}}
        )
        self.assertAlmostEqual(result.to_dict()["metrics"]["primary"], 0.61)
        self.assertEqual(result.to_dict()["git_diff"], "diff")

    def test_command_must_not_be_a_shell_string(self) -> None:
        with self.assertRaises(ValueError):
            ExperimentSpec.from_dict(
                {
                    "hypothesis": "h", "description": "d", "result_compare": "r", "next_steps": "n",
                    "command": "python train.py", "seed": 1,
                }
            )

    def test_optional_budget_stops_future_usage(self) -> None:
        ledger = CostLedger(Budget(max_total_tokens=10))
        ledger.record(TokenUsage(input_tokens=4, output_tokens=4))
        with self.assertRaises(BudgetExceeded):
            ledger.record(TokenUsage(input_tokens=2, output_tokens=1))
        ledger.record_observed(TokenUsage(input_tokens=2, output_tokens=1))
        self.assertEqual(ledger.total.total_tokens, 11)

    def test_stopping_policy_uses_meaningful_improvement(self) -> None:
        policy = StoppingPolicy(epsilon=0.002, patience=3, max_iterations=50, max_wallclock_seconds=60)
        self.assertFalse(policy.observe(0.60, iteration=1, elapsed_seconds=1).stop)
        self.assertFalse(policy.observe(0.601, iteration=2, elapsed_seconds=2).stop)
        self.assertFalse(policy.observe(0.6005, iteration=3, elapsed_seconds=3).stop)
        self.assertTrue(policy.observe(0.6004, iteration=4, elapsed_seconds=4).stop)

    def test_unknown_dependency_profile_is_rejected(self) -> None:
        self.assertIn("torch", request_profile("pytorch").packages)
        with self.assertRaises(DependencyViolation):
            request_profile("arbitrary-package")

    def test_docker_workspace_excludes_protected_evaluator(self) -> None:
        spec = ExperimentSpec("h", "", "d", "r", "n", ("python", "-c", "pass"), 1)
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp) / "repo"
            (repo / "configs").mkdir(parents=True)
            (repo / "configs" / "official_files.json").write_text('{"sha256": {"evaluate.py": "ignored"}}')
            (repo / "evaluate.py").write_text("protected")
            (repo / "train.py").write_text("print('ok')")
            workspace = DockerWorkspace(repo, Path(temp) / "runs")
            workspace.prepare(spec)
            self.assertFalse((workspace.path / "evaluate.py").exists())
            workspace.cleanup()

    def test_docker_is_required_for_execution(self) -> None:
        spec = ExperimentSpec("h", "", "d", "r", "n", ("python", "-c", "pass"), 1)
        executor = DockerExecutor(docker_executable="definitely-not-installed")
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp) / "repo"
            (repo / "configs").mkdir(parents=True)
            (repo / "configs" / "official_files.json").write_text('{"sha256": {"evaluate.py": "ignored"}}')
            (repo / "evaluate.py").write_text("protected")
            workspace = DockerWorkspace(repo, Path(temp) / "runs")
            workspace.prepare(spec)
            with self.assertRaises(IsolationError):
                executor.execute(spec, workspace)
            workspace.cleanup()

    def test_pilot_persists_isolation_failure_and_audit(self) -> None:
        spec = ExperimentSpec("h", "", "d", "r", "n", ("python", "-c", "pass"), 1)
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp) / "repo"
            (repo / "configs").mkdir(parents=True)
            (repo / "configs" / "official_files.json").write_text('{"sha256": {"evaluate.py": "ignored"}}')
            (repo / "evaluate.py").write_text("protected")
            (repo / "train.py").write_text("print('ok')")
            config = load_benchmark_config()
            config = type(config)(
                data_dir=Path(temp), label=config.label, development_splits=config.development_splits,
                primary_metric=config.primary_metric, epsilon=config.epsilon, patience=config.patience,
                max_iterations=config.max_iterations, max_wallclock_hours=config.max_wallclock_hours,
                baseline_expected=config.baseline_expected,
                execution={"docker_executable": "definitely-not-installed"}, budgets={},
            )
            store = RunStore(Path(temp) / "runs")
            result = run_pilot(spec, config=config, run_store=store, baseline_primary=0.60, repo_root=repo)
            self.assertEqual(result.status, "failed")
            self.assertIn("Docker executable not found", result.failure_reason or "")
            self.assertEqual(result.hypothesis, "h")
            self.assertEqual(result.command, ["python", "-c", "pass"])
            self.assertTrue((store.run_dir / "audit.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
