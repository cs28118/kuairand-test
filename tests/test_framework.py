import math
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from framework.accounting import Budget, BudgetExceeded, CostLedger, TokenUsage
from framework.config import load_benchmark_config
from framework.contracts import ExperimentResult, ExperimentSpec
from framework.dependencies import DependencyViolation, request_profile
from framework.guardrails import GuardrailViolation, reject_protected_paths, validate_scores, verify_official_files
from framework.isolation import DockerExecutor, DockerWorkspace, IsolationError
from framework.llm import LLMRequest, LLMResponse, OpenAIResponsesClient, load_dotenv
from framework.pilot import run_pilot
from framework.proposal import ProposalViolation, build_proposal_prompt, parse_llm_experiment_spec
from framework.propose import approve_and_run, generate_proposal
from framework.state import RunStore, make_run_id
from framework.stopping import StoppingPolicy


class FrameworkTests(unittest.TestCase):
    @staticmethod
    def _llm_spec(**overrides):
        spec = {
            "hypothesis": "Test one validation-only variation.",
            "git_diff": "",
            "description": "Run an existing isolated validation experiment.",
            "result_compare": "Compare primary with the FM validation baseline.",
            "next_steps": "Keep it only if primary improves.",
            "command": ["python", "experiments/run_date_dow_fm.py"],
            "seed": 42,
            "dependency_profile": "base",
            "result_file": "experiment_result.json",
            "artifacts": [],
            "metadata": {"name": "llm-validation-proposal"},
        }
        spec.update(overrides)
        return json.dumps(spec)

    def test_benchmark_contract(self) -> None:
        config = load_benchmark_config()
        self.assertEqual(config.label, "long_view")
        self.assertEqual(config.development_splits, ("train", "valid"))
        self.assertEqual(config.primary_metric, "primary")

    def test_run_ids_are_persistent_and_sequential(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            runs_dir = Path(temp) / "runs"
            self.assertEqual(make_run_id(runs_dir=runs_dir), "run-1")
            self.assertEqual(make_run_id(runs_dir=runs_dir), "run-2")
            (runs_dir / "run-10").mkdir()
            self.assertEqual(make_run_id(runs_dir=runs_dir), "run-11")

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

    def test_llm_proposal_prompt_has_rules_baselines_and_allowed_files(self) -> None:
        prompt = build_proposal_prompt(load_benchmark_config(), "Try one ranking hypothesis.")
        self.assertIn("Never use hidden test data", prompt)
        self.assertIn("experiments/*.py", prompt)
        self.assertIn('"primary": 0.6016', prompt)
        self.assertIn("Return ONLY one JSON object", prompt)

    def test_llm_proposal_rejects_non_json_unsafe_commands_and_paths(self) -> None:
        with self.assertRaises(ProposalViolation):
            parse_llm_experiment_spec("```json\n{}\n```")
        with self.assertRaises(ProposalViolation):
            parse_llm_experiment_spec(self._llm_spec(command=["sh", "-c", "echo unsafe"]))
        with self.assertRaises(ProposalViolation):
            parse_llm_experiment_spec(self._llm_spec(git_diff="diff --git a/evaluate.py b/evaluate.py\n--- a/evaluate.py\n+++ b/evaluate.py"))

    def test_openai_payload_requests_strict_experiment_schema(self) -> None:
        payload = OpenAIResponsesClient._payload(LLMRequest("openai", "test-model", "prompt"))
        self.assertEqual(payload["text"]["format"]["type"], "json_schema")
        self.assertTrue(payload["text"]["format"]["strict"])
        self.assertIn("command", payload["text"]["format"]["schema"]["properties"])

    def test_openai_client_uses_configured_base_url(self) -> None:
        client = OpenAIResponsesClient(api_key="test-key", base_url="https://organization.example/v1/")
        self.assertEqual(client.endpoint, "https://organization.example/v1/responses")
        with self.assertRaises(Exception):
            OpenAIResponsesClient(api_key="test-key", base_url="organization.example/v1")

    def test_local_dotenv_loads_without_overriding_process_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            dotenv = Path(temp) / ".env"
            dotenv.write_text("LLM_MODEL=from-file\nNEW_SETTING='quoted value'\n", encoding="utf-8")
            with patch.dict("os.environ", {"LLM_MODEL": "from-process"}, clear=False):
                load_dotenv(dotenv)
                self.assertEqual(os.environ["LLM_MODEL"], "from-process")
                self.assertEqual(os.environ["NEW_SETTING"], "quoted value")

    def test_generated_proposal_is_audited_before_human_approval(self) -> None:
        class FakeClient:
            def generate(self, proposal_request):
                return LLMResponse(
                    text=FrameworkTests._llm_spec(), response_id="resp_123",
                    usage={"input_tokens": 3, "output_tokens": 4, "cached_tokens": 0, "total_tokens": 7}, raw={"id": "resp_123"},
                )

        with tempfile.TemporaryDirectory() as temp:
            store = RunStore(Path(temp) / "runs")
            spec = generate_proposal(
                client=FakeClient(), provider="openai", model="test-model", goal="one experiment",
                config=load_benchmark_config(), store=store,
            )
            self.assertEqual(spec.metadata["name"], "llm-validation-proposal")
            self.assertEqual(store.read_state()["status"], "awaiting_approval")
            self.assertTrue((store.run_dir / "proposal.json").is_file())
            events = [json.loads(line)["event"] for line in (store.run_dir / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(events, ["llm_request", "llm_response", "proposal_validated"])

    def test_approved_proposal_is_revalidated_and_audited_with_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = RunStore(Path(temp) / "runs")
            store.initialize({"mode": "llm_supervised_proposal"})
            (store.run_dir / "proposal.json").write_text(self._llm_spec(), encoding="utf-8")
            result = ExperimentResult(status="completed", metrics={"primary": 0.61})
            with patch("framework.propose.run_pilot", return_value=result) as pilot:
                actual = approve_and_run(
                    store=store, config=load_benchmark_config(), approval_note="reviewed", baseline_primary=0.60,
                )
            self.assertEqual(actual.status, "completed")
            pilot.assert_called_once()
            events = [json.loads(line)["event"] for line in (store.run_dir / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(events, ["human_approval", "llm_proposal_result"])

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
