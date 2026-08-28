"""Non-negotiable development safeguards for experiments."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

from .config import REPO_ROOT


OFFICIAL_FILES_PATH = REPO_ROOT / "configs" / "official_files.json"


class GuardrailViolation(RuntimeError):
    """Raised when an experiment violates the benchmark contract."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_official_files(path: str | Path = OFFICIAL_FILES_PATH) -> dict[str, str]:
    """Ensure protected judge files still match their committed fingerprints."""
    with Path(path).open(encoding="utf-8") as handle:
        protected: dict[str, str] = json.load(handle)["sha256"]

    actual: dict[str, str] = {}
    for relative_path, expected_hash in protected.items():
        target = REPO_ROOT / relative_path
        if not target.is_file():
            raise GuardrailViolation(f"Protected file is missing: {relative_path}")
        actual_hash = sha256_file(target)
        if actual_hash != expected_hash:
            raise GuardrailViolation(
                f"Protected file changed: {relative_path}. Restore the official version before running."
            )
        actual[relative_path] = actual_hash
    return actual


def require_development_split(split: str) -> None:
    if split not in {"train", "valid"}:
        raise GuardrailViolation(
            f"Development experiments may use only train/valid, not split={split!r}."
        )


def validate_scores(scores: Sequence[float], expected_length: int) -> None:
    if len(scores) != expected_length:
        raise GuardrailViolation(
            f"Prediction length {len(scores):,d} does not match expected {expected_length:,d}."
        )
    for index, score in enumerate(scores):
        if not math.isfinite(float(score)):
            raise GuardrailViolation(f"Prediction {index} is NaN or Inf.")


def reject_protected_paths(paths: Iterable[str | Path]) -> None:
    protected = {Path(name) for name in json.loads(OFFICIAL_FILES_PATH.read_text(encoding="utf-8"))["sha256"]}
    for candidate in paths:
        normalized = Path(candidate)
        try:
            relative = normalized.relative_to(REPO_ROOT)
        except ValueError:
            relative = normalized
        if relative in protected:
            raise GuardrailViolation(f"Experiments may not modify protected file: {relative}")

