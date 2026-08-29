"""Immutable benchmark rules shared by RankLab experiments."""
from __future__ import annotations

import hashlib
from pathlib import Path

TRAIN = "train"
VALID = "valid"
TEST = "test"
OFFICIAL_SPLITS = (TRAIN, VALID, TEST)
DEVELOPMENT_SPLITS = (TRAIN, VALID)
SPLITS = OFFICIAL_SPLITS

GAUC = "GAUC"
NDCG_AT_5 = "nDCG@5"
PRIMARY = "primary"
OFFICIAL_METRICS = (GAUC, NDCG_AT_5, PRIMARY)
METRICS = OFFICIAL_METRICS
PRIMARY_METRIC = PRIMARY

EPSILON = 0.002
PATIENCE_ITERATIONS = 3
MAX_ITERATIONS = 50
# Lower-case aliases mirror the research-controller terminology in the brief.
epsilon = EPSILON
patience_iterations = PATIENCE_ITERATIONS
max_iterations = MAX_ITERATIONS

# This is the SHA-256 of the supplied official evaluation script at project setup.
OFFICIAL_EVALUATOR_SHA256 = "ecfde28392eb14fec4f488083251df50624e1af2b86278b962daecfb42d195de"


class ContractViolation(RuntimeError):
    """Raised when an experiment would violate the benchmark safety contract."""


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def evaluator_path() -> Path:
    return project_root() / "evaluate.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_official_evaluator_unchanged(path: Path | None = None) -> str:
    """Fail closed if the official evaluator differs from its supplied checksum."""
    actual = sha256_file(path or evaluator_path())
    if actual != OFFICIAL_EVALUATOR_SHA256:
        raise ContractViolation(
            "evaluate.py checksum mismatch: the official evaluator must not be changed "
            f"(expected {OFFICIAL_EVALUATOR_SHA256}, got {actual})."
        )
    return actual


def reject_test_split(split: str) -> str:
    """Validate an official split name and explicitly forbid test experimentation."""
    normalized = split.strip().lower()
    if normalized not in OFFICIAL_SPLITS:
        raise ContractViolation(f"Unknown split {split!r}; expected one of {OFFICIAL_SPLITS}.")
    if normalized == TEST:
        raise ContractViolation("Test is held out and cannot be used for experiments or evaluation.")
    return normalized


def require_validation_split(split: str) -> str:
    """Allow only the official validation split for reported experiment metrics."""
    normalized = reject_test_split(split)
    if normalized != VALID:
        raise ContractViolation("RankLab experiment reporting is validation-only.")
    return normalized
