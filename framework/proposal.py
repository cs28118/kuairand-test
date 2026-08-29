"""Validation and prompt construction for one LLM-proposed experiment.

This module deliberately contains no iteration policy.  It turns one model
response into a reviewable :class:`ExperimentSpec`, or rejects it before a
Docker workspace is ever created.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from .config import BenchmarkConfig
from .contracts import ExperimentSpec
from .dependencies import APPROVED_PROFILES
from .guardrails import GuardrailViolation, reject_protected_paths
from .isolation import diff_paths


# A proposal may change an experiment implementation or these small baseline
# helpers.  It never needs the evaluator, submission code, framework, data,
# credentials, or Docker configuration.
ALLOWED_FILE_PATTERNS = (
    "ablation_features.py",
    "baseline.py",
    "data.py",
    "experiments/*.py",
)

_SPEC_KEYS = {
    "hypothesis",
    "git_diff",
    "description",
    "result_compare",
    "next_steps",
    "command",
    "seed",
    "dependency_profile",
    "result_file",
    "artifacts",
    "metadata",
}


class ProposalViolation(ValueError):
    """Raised when a model response is not a safe ExperimentSpec."""


def experiment_spec_schema() -> dict[str, Any]:
    """Return the JSON Schema supplied to providers that support structured output."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "hypothesis", "git_diff", "description", "result_compare", "next_steps",
            "command", "seed", "dependency_profile", "result_file", "artifacts", "metadata",
        ],
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
            "metadata": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name"],
                "properties": {"name": {"type": "string", "minLength": 1}},
            },
        },
    }


def _is_allowed_file(path: str) -> bool:
    candidate = Path(path.replace("\\", "/"))
    if candidate.is_absolute() or ".." in candidate.parts:
        return False
    normalized = candidate.as_posix()
    return any(candidate.match(pattern) for pattern in ALLOWED_FILE_PATTERNS) and normalized != ""


def _parse_json_object(raw_response: str) -> dict[str, Any]:
    if not isinstance(raw_response, str):
        raise ProposalViolation("LLM response must be text containing an ExperimentSpec JSON object")
    decoder = json.JSONDecoder()
    stripped = raw_response.strip()
    try:
        value, consumed = decoder.raw_decode(stripped)
    except json.JSONDecodeError as exc:
        raise ProposalViolation(f"LLM response is not valid JSON: {exc.msg}") from exc
    if consumed != len(stripped) or not isinstance(value, dict):
        raise ProposalViolation("LLM response must contain only one ExperimentSpec JSON object")
    return value


def parse_llm_experiment_spec(raw_response: str) -> ExperimentSpec:
    """Parse exact JSON, reject unknown keys, then validate command/path policy."""
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
    """Enforce the narrow execution and modification policy for LLM proposals."""
    command = list(spec.command)
    if command[0] not in {"python", "python3"}:
        raise ProposalViolation("command must start with python or python3")
    if len(command) < 2 or not command[1].endswith(".py") or not _is_allowed_file(command[1]):
        raise ProposalViolation("command must run an allowed relative .py experiment file")
    if any("\x00" in part or "\n" in part or "\r" in part for part in command):
        raise ProposalViolation("command arguments may not contain control characters")
    if spec.dependency_profile not in APPROVED_PROFILES:
        raise ProposalViolation(f"dependency profile is not approved: {spec.dependency_profile}")

    if set(spec.metadata) != {"name"} or not isinstance(spec.metadata["name"], str) or not spec.metadata["name"].strip():
        raise ProposalViolation("metadata must contain only a non-empty name")

    changed = _validated_diff_paths(spec.git_diff)
    for path in changed:
        if not _is_allowed_file(path):
            raise ProposalViolation(f"git diff targets a file outside the allowed list: {path}")
    try:
        reject_protected_paths(changed)
    except GuardrailViolation as exc:
        raise ProposalViolation(str(exc)) from exc


def _validated_diff_paths(git_diff: str) -> list[str]:
    """Accept only ordinary text patches with explicit, allowed file headers."""
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
    # Inspect both the diff header and +++/--- lines.  The latter protects
    # against a header/body mismatch that git apply might otherwise accept.
    return sorted(set(targets) | set(diff_paths(git_diff)))


def build_proposal_prompt(config: BenchmarkConfig, goal: str) -> str:
    """Build the complete, reviewable prompt used for exactly one proposal."""
    baseline_metrics = json.dumps(config.baseline_expected, indent=2, sort_keys=True)
    allowed_files = "\n".join(f"- {pattern}" for pattern in ALLOWED_FILE_PATTERNS)
    profiles = ", ".join(sorted(APPROVED_PROFILES))
    return f"""You are proposing exactly one small, validation-only KuaiRand experiment.

Project rules:
- Use only the development splits: {', '.join(config.development_splits)}. Never use hidden test data.
- Never modify, import, or execute evaluate.py. It is a protected judge file and is absent from Docker.
- Do not modify framework code, Docker files, dependency manifests, credentials, submissions, or dataset files.
- The Docker container has no network. Do not download packages or invoke a shell, package manager, or subprocess launcher.
- Use an argv command beginning with `python` or `python3`, followed by one allowed relative .py file. Never use `-c`, `-m`, a shell, or an inline script.
- The experiment must write `experiment_result.json` with `status`, numeric `metrics` including `{config.primary_metric}`, and any artifacts it declares.
- Propose a small, complete, reviewable unified git diff. An empty diff is allowed when an existing experiment script is sufficient; if you cannot write a complete `diff --git ...` patch, use `git_diff: ""` and run an existing script such as `experiments/run_date_dow_fm.py`. Never emit a partial or pseudo-diff.

Allowed files to modify or execute:
{allowed_files}

Approved dependency profiles: {profiles}
Baseline validation metrics:
{baseline_metrics}

Requested research goal:
{goal}

Output a DATA INSTANCE, not a schema. Return ONLY one JSON object with exactly these top-level keys:
`hypothesis`, `git_diff`, `description`, `result_compare`, `next_steps`, `command`, `seed`, `dependency_profile`, `result_file`, `artifacts`, `metadata`.
All five review fields (`hypothesis`, `description`, `result_compare`, and `next_steps`, plus the change description) are strings. `command` and `artifacts` are arrays of strings; `metadata` is {{"name": "short-name"}}. In particular, `result_compare` must be a plain string, never an object or array.
Do not output schema keywords such as `properties`, `additionalProperties`, or `required`. Do not wrap the JSON in Markdown, a code fence, or explanation.

Valid shape example (replace every example value with your proposal):
{{"hypothesis":"one testable idea","git_diff":"","description":"run one validation experiment","result_compare":"Compare primary with FM baseline 0.6016","next_steps":"Keep only if primary improves","command":["python","experiments/example.py"],"seed":42,"dependency_profile":"base","result_file":"experiment_result.json","artifacts":[],"metadata":{{"name":"one-experiment"}}}}
"""
