"""JSON contracts exchanged by the supervised research pilot.

The instruction fields are deliberately explicit.  They make a proposed
experiment reviewable by a human before it is executed and give a future LLM
adapter a stable, provider-neutral payload.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


ALLOWED_STATUSES = {"completed", "failed", "timed_out", "stopped", "rejected"}


def _empty_token_usage() -> dict[str, Any]:
    return {
        "model": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "total_tokens": 0,
        "estimated_cost": 0.0,
    }


def _non_empty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class ExperimentSpec:
    """A human-approved instruction for exactly one experiment.

    ``command`` is an argv vector, never a shell string.  ``git_diff`` is
    applied only inside the disposable workspace by the pilot.
    """

    hypothesis: str
    git_diff: str
    description: str
    result_compare: str
    next_steps: str
    command: tuple[str, ...]
    seed: int
    dependency_profile: str = "base"
    result_file: str = "experiment_result.json"
    artifacts: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("hypothesis", "description", "result_compare", "next_steps"):
            _non_empty(getattr(self, name), name)
        if not isinstance(self.git_diff, str):
            raise ValueError("git_diff must be a string")
        if not self.command or any(not isinstance(part, str) or not part for part in self.command):
            raise ValueError("command must be a non-empty argv tuple")
        if not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        result_path = Path(self.result_file)
        if result_path.is_absolute() or ".." in result_path.parts:
            raise ValueError("result_file must stay inside the experiment workspace")
        for artifact in self.artifacts:
            path = Path(artifact)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("artifacts must stay inside the experiment workspace")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentSpec":
        """Load both the canonical names and common instruction aliases."""
        command = raw.get("command")
        if isinstance(command, str):
            raise ValueError("command must be a JSON array, not a shell string")
        return cls(
            hypothesis=raw["hypothesis"],
            git_diff=raw.get("git_diff", raw.get("code_change", "")),
            description=raw["description"],
            result_compare=raw.get("result_compare", raw.get("result_comparison", "")),
            next_steps=raw.get("next_steps", raw.get("what_to_do_next", "")),
            command=tuple(command or ()),
            seed=int(raw["seed"]),
            dependency_profile=str(raw.get("dependency_profile", "base")),
            result_file=str(raw.get("result_file", "experiment_result.json")),
            artifacts=tuple(raw.get("artifacts", ())),
            metadata=dict(raw.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class ExperimentResult:
    """The durable outcome of one supervised experiment."""

    metrics: dict[str, float | int] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    status: str = "failed"
    failure_reason: str | None = None
    result_compare: str = ""
    next_steps: str = ""
    modified_files: list[str] = field(default_factory=list)
    stdout: str | None = None
    stderr: str | None = None
    recovery_attempts: list[dict[str, Any]] = field(default_factory=list)
    token_usage: dict[str, Any] = field(default_factory=_empty_token_usage)
    beats_fm: bool | None = None
    stop_reason: str | None = None
    hypothesis: str = ""
    git_diff: str = ""
    description: str = ""
    command: list[str] = field(default_factory=list)
    seed: int | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    source_provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in ALLOWED_STATUSES:
            raise ValueError(f"unsupported experiment status: {self.status}")
        for name, value in self.metrics.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"metric {name!r} must be numeric")
        if self.status == "failed" and not self.failure_reason:
            raise ValueError("failed results require failure_reason")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentResult":
        return cls(
            metrics=dict(raw.get("metrics", {})),
            artifacts=list(raw.get("artifacts", [])),
            status=str(raw.get("status", "completed")),
            failure_reason=raw.get("failure_reason"),
            result_compare=str(raw.get("result_compare", raw.get("result_comparison", ""))),
            next_steps=str(raw.get("next_steps", raw.get("what_to_do_next", ""))),
            modified_files=list(raw.get("modified_files", [])),
            stdout=raw.get("stdout"),
            stderr=raw.get("stderr"),
            recovery_attempts=list(raw.get("recovery_attempts", [])),
            token_usage=dict(raw.get("token_usage", _empty_token_usage())),
            beats_fm=raw.get("beats_fm"),
            stop_reason=raw.get("stop_reason"),
            hypothesis=str(raw.get("hypothesis", "")),
            git_diff=str(raw.get("git_diff", raw.get("code_change", ""))),
            description=str(raw.get("description", "")),
            command=list(raw.get("command", [])),
            seed=raw.get("seed"),
            provenance=dict(raw.get("provenance", {})),
            source_provenance=dict(raw.get("source_provenance", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target


def load_experiment_spec(path: str | Path) -> ExperimentSpec:
    with Path(path).open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("experiment spec must be a JSON object")
    return ExperimentSpec.from_dict(raw)
