"""Prompt construction and safety validation for LLM experiment proposals."""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
from typing import Any

from framework.config import BenchmarkConfig, REPO_ROOT
from framework.contracts import ExperimentSpec
from framework.dependencies import APPROVED_PROFILES
from framework.guardrails import GuardrailViolation, reject_protected_paths
from framework.isolation import diff_paths


ALLOWED_FILE_PATTERNS = ("ablation_features.py", "baseline.py", "data.py", "experiments/*.py")
_SPEC_KEYS = {
    "hypothesis", "git_diff", "description", "result_compare", "next_steps", "command", "seed",
    "dependency_profile", "result_file", "artifacts", "metadata",
}


class ProposalViolation(ValueError):
    """Raised when a model response is not a safe ExperimentSpec."""


def experiment_spec_schema() -> dict[str, Any]:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["hypothesis", "git_diff", "description", "result_compare", "next_steps", "command", "seed", "dependency_profile", "result_file", "artifacts", "metadata"],
        "properties": {
            "hypothesis": {"type": "string", "minLength": 1},
            "git_diff": {"type": "string"},
            "description": {"type": "string", "minLength": 1},
            "result_compare": {"type": "string", "minLength": 1},
            "next_steps": {"type": "string", "minLength": 1},
            "command": {"type": "array", "minItems": 2, "items": {"type": "string", "minLength": 1}},
            "seed": {"type": "integer"},
            "dependency_profile": {"type": "string"},
            "result_file": {"type": "string", "minLength": 1},
            "artifacts": {"type": "array", "items": {"type": "string"}},
            "metadata": {"type": "object", "additionalProperties": False, "required": ["name"], "properties": {"name": {"type": "string", "minLength": 1}}},
        },
    }


def _is_allowed_file(path: str) -> bool:
    candidate = Path(path.replace("\\", "/"))
    if candidate.is_absolute() or ".." in candidate.parts:
        return False
    return bool(candidate.as_posix()) and any(candidate.match(pattern) for pattern in ALLOWED_FILE_PATTERNS)


def _parse_json_object(raw_response: str) -> dict[str, Any]:
    if not isinstance(raw_response, str):
        raise ProposalViolation("LLM response must be text containing an ExperimentSpec JSON object")
    stripped = raw_response.strip()
    try:
        value, consumed = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError as exc:
        raise ProposalViolation(f"LLM response is not valid JSON: {exc.msg}") from exc
    if consumed != len(stripped) or not isinstance(value, dict):
        raise ProposalViolation("LLM response must contain only one ExperimentSpec JSON object")
    return value


def parse_llm_experiment_spec(raw_response: str) -> ExperimentSpec:
    raw = _parse_json_object(raw_response)
    unknown = set(raw) - _SPEC_KEYS
    missing = _SPEC_KEYS - set(raw)
    if unknown or missing:
        details = []
        if missing:
            details.append("missing fields: " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unknown fields: " + ", ".join(sorted(unknown)))
        raise ProposalViolation("invalid ExperimentSpec schema; " + "; ".join(details))
    if not isinstance(raw["command"], list) or not isinstance(raw["artifacts"], list):
        raise ProposalViolation("command and artifacts must be JSON arrays")
    if not isinstance(raw["metadata"], dict) or set(raw["metadata"]) != {"name"}:
        raise ProposalViolation("metadata must contain only a non-empty name")
    if type(raw["seed"]) is not int:
        raise ProposalViolation("seed must be a JSON integer")
    try:
        spec = ExperimentSpec.from_dict(raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise ProposalViolation(f"invalid ExperimentSpec: {exc}") from exc
    validate_safe_spec(spec)
    return spec


def validate_safe_spec(spec: ExperimentSpec) -> None:
    command = list(spec.command)
    if command[0] not in {"python"}:
        raise ProposalViolation("command must start with python")
    if len(command) < 2 or not command[1].endswith(".py") or not _is_allowed_file(command[1]):
        raise ProposalViolation("command must run an allowed relative .py experiment file")
    if any("\x00" in part or "\n" in part or "\r" in part for part in command):
        raise ProposalViolation("command arguments may not contain control characters")
    if spec.dependency_profile not in APPROVED_PROFILES:
        raise ProposalViolation(f"dependency profile is not approved: {spec.dependency_profile}")
    if set(spec.metadata) != {"name"} or not isinstance(spec.metadata["name"], str) or not spec.metadata["name"].strip():
        raise ProposalViolation("metadata must contain only a non-empty name")
    changed = _validated_diff_paths(spec.git_diff)
    command_path = Path(command[1])
    if not (REPO_ROOT / command_path).is_file() and command[1].replace("\\", "/") not in changed:
        raise ProposalViolation(f"command target does not exist and is not created by git_diff: {command[1]}")
    for path in changed:
        if not _is_allowed_file(path):
            raise ProposalViolation(f"git diff targets a file outside the allowed list: {path}")
    try:
        reject_protected_paths(changed)
    except GuardrailViolation as exc:
        raise ProposalViolation(str(exc)) from exc


def _validated_diff_paths(git_diff: str) -> list[str]:
    if not git_diff:
        return []
    if any(marker in git_diff for marker in ("new file mode 120000", "rename from ", "rename to ", "Binary files ")):
        raise ProposalViolation("git diff may not create links, rename files, or contain binary changes")
    targets: list[str] = []
    for line in git_diff.splitlines():
        if not line.startswith("diff --git "):
            continue
        match = re.fullmatch(r"diff --git a/([^\s]+) b/([^\s]+)", line)
        if not match:
            raise ProposalViolation("git diff contains an unsupported or unsafe file header")
        targets.extend(match.groups())
    if not targets:
        raise ProposalViolation("non-empty git diff must contain ordinary diff --git file headers")
    check = subprocess.run(
        ["git", "apply", "--check", "--recount", "--unidiff-zero", "--whitespace=nowarn", "-"],
        cwd=REPO_ROOT,
        input=git_diff.encode("utf-8"),
        capture_output=True,
        check=False,
    )
    if check.returncode != 0:
        detail = check.stderr.decode("utf-8", errors="replace").strip()[:300]
        raise ProposalViolation(f"git diff cannot be applied cleanly: {detail or 'git apply check failed'}")
    return sorted(set(targets) | set(diff_paths(git_diff)))


def build_proposal_prompt(config: BenchmarkConfig, goal: str) -> str:
    prompt_dir = Path(__file__).parent / "prompts"
    prompt_parts = (
        "role.md",
        "project_rules.md",
        "repository_context.md",
        "research_context.md",
        "output_contract.md",
    )
    template = "\n\n".join(
        (prompt_dir / name).read_text(encoding="utf-8").strip()
        for name in prompt_parts
    )
    return template.format(
        development_splits=", ".join(config.development_splits),
        primary_metric=config.primary_metric,
        allowed_files="\n".join(f"- {pattern}" for pattern in ALLOWED_FILE_PATTERNS),
        profiles=", ".join(sorted(APPROVED_PROFILES)),
        baseline_metrics=json.dumps(config.baseline_expected, indent=2, sort_keys=True),
        goal=goal,
    )
