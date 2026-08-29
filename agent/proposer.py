"""Generate one ExperimentSpec, wait for human approval, then run the pilot."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from framework.config import BenchmarkConfig, load_benchmark_config
from framework.contracts import ExperimentSpec
from framework.pilot import _baseline_primary, run_pilot
from framework.state import RunStore

from .llm import LLMRequest, ProposalClient, configured_client, load_dotenv
from .proposal import ProposalViolation, build_proposal_prompt, parse_llm_experiment_spec


def _proposal_path(store: RunStore) -> Path:
    return store.run_dir / "proposal.json"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def generate_proposal(*, client: ProposalClient, provider: str, model: str, goal: str, config: BenchmarkConfig, store: RunStore) -> ExperimentSpec:
    prompt = build_proposal_prompt(config, goal)
    proposal_request = LLMRequest(provider=provider, model=model, prompt=prompt)
    store.initialize({"framework_version": 3, "mode": "llm_supervised_proposal", "primary_metric": config.primary_metric, "development_splits": list(config.development_splits)})
    _write_json(store.run_dir / "llm_request.json", proposal_request.audit_dict())
    store.append_audit({"event": "llm_request", "request": proposal_request.audit_dict()})
    try:
        response = client.generate(proposal_request)
    except Exception as exc:
        store.append_audit({"event": "llm_failure", "failure_reason": str(exc)})
        store.complete("failed")
        raise
    _write_json(store.run_dir / "llm_response.json", response.audit_dict())
    (store.run_dir / "llm_output.txt").write_text(response.text, encoding="utf-8")
    store.append_audit({"event": "llm_response", "response": response.audit_dict()})
    try:
        spec = parse_llm_experiment_spec(response.text)
    except ProposalViolation as exc:
        store.append_audit({"event": "proposal_rejected", "reason": str(exc)})
        store.complete("rejected")
        raise
    _proposal_path(store).write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    store.append_audit({"event": "proposal_validated", "proposal": spec.to_dict()})
    store.complete("awaiting_approval")
    return spec


def approve_and_run(*, store: RunStore, config: BenchmarkConfig, approval_note: str, baseline_primary: float | None = None) -> Any:
    proposal_path = _proposal_path(store)
    if not proposal_path.is_file():
        raise ValueError(f"no validated proposal found for run {store.run_id}")
    spec = parse_llm_experiment_spec(proposal_path.read_text(encoding="utf-8"))
    store.append_audit({"event": "human_approval", "approved": True, "approval_note": approval_note, "proposal": spec.to_dict()})
    baseline = baseline_primary if baseline_primary is not None else _baseline_primary(config)
    result = run_pilot(spec, config=config, run_store=store, baseline_primary=baseline)
    store.append_audit({"event": "llm_proposal_result", "result": result.to_dict()})
    store.complete("completed" if result.status == "completed" else result.status)
    return result


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Generate one LLM ExperimentSpec and require human approval before Docker execution.")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--goal", help="Research goal for one validation-only proposal")
    actions.add_argument("--approve-run", help="Run ID of a previously validated proposal to approve")
    parser.add_argument(
        "--goal-file",
        default=str(Path(__file__).parent / "prompts" / "research_goal.md"),
        help="Markdown file containing the default research goal",
    )
    parser.add_argument("--provider", default=os.environ.get("LLM_PROVIDER", "openai"))
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL"))
    parser.add_argument("--approval-note", default="Reviewed by human via agent.proposer")
    parser.add_argument("--config", default=None)
    parser.add_argument("--runs-dir", default=None)
    parser.add_argument("--run-id", default=None, help="Optional run ID when generating a proposal")
    parser.add_argument("--baseline-primary", type=float, default=None)
    args = parser.parse_args(argv)
    try:
        config = load_benchmark_config(args.config)
        if args.approve_run:
            if args.goal:
                raise ValueError("do not combine --approve-run with --goal")
            store = RunStore(args.runs_dir, args.approve_run)
            result = approve_and_run(store=store, config=config, approval_note=args.approval_note, baseline_primary=args.baseline_primary)
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
            return 0 if result.status in {"completed", "stopped"} else 1
        if args.goal:
            goal = args.goal
        else:
            goal_path = Path(args.goal_file)
            if not goal_path.is_file():
                raise ValueError(f"research goal file not found: {goal_path}; provide --goal or --goal-file")
            goal = goal_path.read_text(encoding="utf-8").strip()
            if not goal:
                raise ValueError(f"research goal file is empty: {goal_path}")
        if not args.model or args.model == "replace-with-a-pinned-model":
            raise ValueError("set LLM_MODEL to a pinned model name or provide --model")
        store = RunStore(args.runs_dir, args.run_id)
        spec = generate_proposal(client=configured_client(args.provider), provider=args.provider, model=args.model, goal=goal, config=config, store=store)
    except Exception as exc:
        print(f"agent proposal failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"run_id": store.run_id, "status": "awaiting_approval", "proposal": spec.to_dict()}, indent=2, sort_keys=True))
    print(f"Review runs/{store.run_id}/proposal.json, then approve with: python -m agent.proposer --approve-run {store.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
