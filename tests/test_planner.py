from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ranklab.planner.propose import build_proposal, propose


def _policy(root: Path, *, iterations: int = 50, minutes: int = 360) -> Path:
    path = root / "research_policy.json"
    path.write_text(json.dumps({
        "max_iterations": iterations,
        "time_budget_minutes": minutes,
        "forbidden_files": ["evaluate.py", "data.py", "baseline.py", "submit.py"],
        "forbidden_terms": ["test data", "test split", "test labels", "leaderboard", "submission scoring", "external data"],
        "minimum_validation_primary_improvement": 0.002,
    }), encoding="utf-8")
    return path


def _opportunity(slice_name: str, *, delta: float = -0.2, users: int = 100, rows: int = 200) -> dict[str, str]:
    return {
        "weak_slice": slice_name,
        "evidence": f"primary 0.400000, {delta:+.6f} versus overall validation primary 0.600000; {users} users and {rows} rows.",
        "likely_cause": "A diagnostic opportunity backed by validation-only evidence.",
        "candidate_experiment_family": "planner input only",
    }


def _write_inputs(root: Path, opportunities: list[dict[str, str]], *, gauc: float = 0.6, ndcg: float = 0.7) -> tuple[Path, Path]:
    artifacts = root / "artifacts"
    report_dir = artifacts / "reports" / "run-1"
    report_dir.mkdir(parents=True)
    (report_dir / "diagnostics.json").write_text(json.dumps({
        "run_id": "run-1", "overall_metrics": {"GAUC": gauc, "nDCG@5": ndcg, "primary": 0.6},
        "top_opportunities": opportunities,
    }), encoding="utf-8")
    (artifacts / "iterations.jsonl").write_text("", encoding="utf-8")
    return artifacts, _policy(root)


class PlannerTests(unittest.TestCase):
    def test_same_inputs_have_same_id_and_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            artifacts, policy = _write_inputs(Path(temp), [
                _opportunity("content_tag=rare", delta=-0.3, rows=300),
                _opportunity("item_popularity_train_exposures=tail", delta=-0.1, rows=100),
            ])
            first = build_proposal("run-1", artifacts_dir=artifacts, policy_path=policy)
            second = build_proposal("run-1", artifacts_dir=artifacts, policy_path=policy)
            self.assertEqual(first["proposal_id"], second["proposal_id"])
            self.assertEqual(first["experiment_family"], second["experiment_family"])
            self.assertEqual(first["target_slice"], "content_tag=rare")

    def test_tail_weakness_selects_causal_features(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            artifacts, policy = _write_inputs(Path(temp), [_opportunity("item_popularity_train_exposures=tail")])
            proposal = build_proposal("run-1", artifacts_dir=artifacts, policy_path=policy)
            self.assertEqual(proposal["experiment_family"], "causal_features")
            self.assertFalse(proposal["execution_authorized"])

    def test_low_ndcg_relative_to_gauc_selects_ranking_loss(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            artifacts, policy = _write_inputs(Path(temp), [_opportunity("context_tab=8")], gauc=0.7, ndcg=0.4)
            proposal = build_proposal("run-1", artifacts_dir=artifacts, policy_path=policy)
            self.assertEqual(proposal["experiment_family"], "ranking_loss")

    def test_rejected_experiment_is_not_repeated_without_new_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifacts, policy = _write_inputs(root, [_opportunity("content_tag=rare")])
            (artifacts / "planner_proposals.jsonl").write_text(json.dumps({
                "status": "rejected", "experiment_family": "causal_features", "target_slice": "content_tag=rare",
            }) + "\n", encoding="utf-8")
            result = build_proposal("run-1", artifacts_dir=artifacts, policy_path=policy)
            self.assertEqual(result["status"], "no_eligible_proposal")

    def test_budget_exhaustion_returns_no_eligible_proposal(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifacts, _ = _write_inputs(root, [_opportunity("content_tag=rare")])
            result = build_proposal("run-1", artifacts_dir=artifacts, policy_path=_policy(root, iterations=0))
            self.assertEqual(result["status"], "no_eligible_proposal")

    def test_persists_validated_non_executable_proposal_and_event(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            artifacts, policy = _write_inputs(Path(temp), [_opportunity("content_tag=rare")])
            proposal = propose("run-1", artifacts_dir=artifacts, policy_path=policy)
            destination = artifacts / "planner_proposals" / proposal["proposal_id"]
            self.assertTrue((destination / "proposal.json").is_file())
            self.assertEqual(json.loads((destination / "validation.json").read_text())["status"], "valid")
            event = json.loads((artifacts / "planner_proposals.jsonl").read_text())
            self.assertFalse(event["execution_authorized"])

    def test_repeated_identical_proposal_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            artifacts, policy = _write_inputs(Path(temp), [_opportunity("content_tag=rare")])
            first = propose("run-1", artifacts_dir=artifacts, policy_path=policy)
            second = propose("run-1", artifacts_dir=artifacts, policy_path=policy)
            self.assertEqual(second, first)
            events = (artifacts / "planner_proposals.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(events), 1)

    def test_planner_never_reads_dataset_or_executes_code(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            artifacts, policy = _write_inputs(Path(temp), [_opportunity("content_tag=rare")])
            reads: list[Path] = []
            original_read_text = Path.read_text

            def tracked_read_text(path: Path, *args, **kwargs):
                reads.append(path)
                return original_read_text(path, *args, **kwargs)

            with patch("subprocess.run", side_effect=AssertionError("must not execute")), patch("os.system", side_effect=AssertionError("must not execute")):
                with patch.object(Path, "read_text", tracked_read_text):
                    proposal = build_proposal("run-1", artifacts_dir=artifacts, policy_path=policy)
            self.assertEqual(proposal["target_slice"], "content_tag=rare")
            self.assertTrue(all("dataset" not in str(path).lower() and "test" not in str(path).lower() for path in reads))
            self.assertFalse(any("dataset" in str(value).lower() or "test" in str(value).lower() for value in proposal["allowed_files"]))


if __name__ == "__main__":
    unittest.main()
