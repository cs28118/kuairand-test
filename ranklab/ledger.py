"""Append-only experiment ledger."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import project_root

LEDGER_PATH = project_root() / "artifacts" / "iterations.jsonl"

REQUIRED_FIELDS = frozenset(
    {
        "run_id",
        "timestamp",
        "hypothesis",
        "configuration",
        "seed",
        "code_hash",
        "runtime_seconds",
        "GAUC",
        "nDCG@5",
        "primary",
        "status",
        "error_recovery_note",
    }
)


def append_run(record: Mapping[str, Any], path: Path | None = None) -> Path:
    """Validate and append exactly one JSON object as one ledger line."""
    missing = REQUIRED_FIELDS.difference(record)
    if missing:
        raise ValueError(f"Ledger record is missing required fields: {sorted(missing)}")
    destination = path or LEDGER_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dict(record), sort_keys=True, allow_nan=False, separators=(",", ":"))
    # A single append write keeps each completed record a single JSONL line.
    with destination.open("a", encoding="utf-8", newline="\n") as ledger:
        ledger.write(encoded + "\n")
    return destination
