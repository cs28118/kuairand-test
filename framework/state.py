"""Persistent run state and append-only iteration logs."""
from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path
import uuid
from typing import Any

from .config import REPO_ROOT


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def make_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write state atomically so interrupted runs retain the previous state."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)


class RunStore:
    """Owns one run directory and is the only writer of its canonical state."""

    def __init__(self, runs_dir: str | Path | None = None, run_id: str | None = None):
        self.runs_dir = Path(runs_dir) if runs_dir else REPO_ROOT / "runs"
        self.run_id = run_id or make_run_id()
        self.run_dir = self.runs_dir / self.run_id
        self.state_path = self.run_dir / "state.json"
        self.log_path = self.run_dir / "iterations.jsonl"

    def initialize(self, metadata: dict[str, Any]) -> dict[str, Any]:
        if self.state_path.exists():
            return self.read_state()
        self.run_dir.mkdir(parents=True, exist_ok=False)
        state = {
            "run_id": self.run_id,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "status": "running",
            "iterations": 0,
            "best": None,
            "usage": {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "total_tokens": 0, "estimated_cost": 0.0, "models": []},
            "metadata": metadata,
        }
        atomic_write_json(self.state_path, state)
        return state

    def read_state(self) -> dict[str, Any]:
        with self.state_path.open(encoding="utf-8") as handle:
            state = json.load(handle)
        state.setdefault("usage", {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "total_tokens": 0, "estimated_cost": 0.0, "models": []})
        state["usage"].setdefault("models", [])
        return state

    def append_audit(self, event: dict[str, Any]) -> None:
        with (self.run_dir / "audit.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"recorded_at": utc_now(), **event}, sort_keys=True) + "\n")

    def record_usage(self, usage: dict[str, Any]) -> dict[str, Any]:
        state = self.read_state()
        totals = state.setdefault("usage", {})
        for name in ("input_tokens", "output_tokens", "cached_tokens", "total_tokens"):
            totals[name] = int(totals.get(name, 0)) + int(usage.get(name, 0))
        totals["estimated_cost"] = float(totals.get("estimated_cost", 0.0)) + float(usage.get("estimated_cost", 0.0))
        models = list(totals.get("models", []))
        model = usage.get("model")
        if model and model not in models:
            models.append(str(model))
        totals["models"] = models
        state["updated_at"] = utc_now()
        atomic_write_json(self.state_path, state)
        return totals

    def append_iteration(self, record: dict[str, Any]) -> dict[str, Any]:
        state = self.read_state()
        iteration = int(state["iterations"]) + 1
        enriched = {"iteration": iteration, "recorded_at": utc_now(), **record}
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(enriched, sort_keys=True) + "\n")

        state["iterations"] = iteration
        metrics = enriched.get("metrics") or {}
        primary = metrics.get("primary")
        if primary is not None and (state["best"] is None or primary > state["best"]["primary"]):
            state["best"] = {
                "iteration": iteration,
                "primary": primary,
                "metrics": metrics,
                "checkpoint": enriched.get("checkpoint"),
                "experiment": enriched.get("experiment"),
            }
        state["updated_at"] = utc_now()
        atomic_write_json(self.state_path, state)
        return enriched

    def complete(self, status: str = "completed") -> dict[str, Any]:
        state = self.read_state()
        state["status"] = status
        state["updated_at"] = utc_now()
        atomic_write_json(self.state_path, state)
        return state
