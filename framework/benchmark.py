"""Validation-only wrappers around the provided KuaiRand starter kit.

The official baseline helpers score both valid and test.  This module reuses
their FM implementation but intentionally loads and evaluates only train and
validation data for autonomous development.
"""
from __future__ import annotations

from dataclasses import dataclass
import collections
import time
from pathlib import Path
from typing import Any

import numpy as np

from baseline import FM
from data import encode, load
from evaluate import evaluate

from .guardrails import require_development_split, validate_scores


@dataclass
class BaselineOutcome:
    metrics: dict[str, float]
    history: list[dict[str, float]]
    checkpoint: dict[str, Any] | None = None


def _plain_metrics(metrics: dict[str, Any]) -> dict[str, float | int]:
    """Convert NumPy scalar metric values into JSON-safe Python primitives."""
    result: dict[str, float | int] = {}
    for name, value in metrics.items():
        if name in {"users", "rows"}:
            result[name] = int(value)
        else:
            result[name] = float(value)
    return result


def load_development_splits(data_dir: str | Path) -> dict[str, list[tuple[Any, ...]]]:
    """Return only the train/valid records to all framework callers."""
    splits = load(str(data_dir))
    return {name: splits[name] for name in ("train", "valid")}


def _valid_arrays(splits: dict[str, list[tuple[Any, ...]]]) -> tuple[list[str], list[int]]:
    require_development_split("valid")
    rows = splits["valid"]
    return [row[1] for row in rows], [row[6] for row in rows]


def run_random_validation(splits: dict[str, list[tuple[Any, ...]]], seed: int = 0) -> BaselineOutcome:
    users, labels = _valid_arrays(splits)
    scores = np.random.default_rng(seed).random(len(labels))
    validate_scores(scores, len(labels))
    return BaselineOutcome(_plain_metrics(evaluate(users, labels, scores)), history=[])


def run_pop_validation(splits: dict[str, list[tuple[Any, ...]]], prior: float = 20.0) -> BaselineOutcome:
    positives: collections.Counter[str] = collections.Counter()
    impressions: collections.Counter[str] = collections.Counter()
    for row in splits["train"]:
        impressions[row[2]] += 1
        positives[row[2]] += row[6]
    global_mean = sum(positives.values()) / sum(impressions.values())

    def score(video_id: str) -> float:
        if not impressions[video_id]:
            return global_mean
        return (positives[video_id] + prior * global_mean) / (impressions[video_id] + prior)

    users, labels = _valid_arrays(splits)
    scores = [score(row[2]) for row in splits["valid"]]
    validate_scores(scores, len(labels))
    return BaselineOutcome(_plain_metrics(evaluate(users, labels, scores)), history=[])


def run_fm_validation(
    splits: dict[str, list[tuple[Any, ...]]],
    *,
    k: int = 16,
    lr: float = 0.001,
    epochs: int = 40,
    batch_size: int = 8192,
    patience: int = 4,
    seed: int = 0,
    verbose: bool = True,
) -> BaselineOutcome:
    """Train the official FM formulation without loading or scoring test metrics."""
    encoded, dimension = encode(splits)
    X_train, y_train, _ = encoded["train"]
    X_valid, y_valid, users_valid = encoded["valid"]
    model = FM(dimension, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best_primary = -float("inf")
    best_state: tuple[np.ndarray, np.ndarray, np.float32] | None = None
    bad_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        indices = rng.permutation(len(y_train))
        losses = [
            model.step(X_train[indices[start : start + batch_size]], y_train[indices[start : start + batch_size]])
            for start in range(0, len(indices), batch_size)
        ]
        scores = model.predict(X_valid)
        validate_scores(scores, len(y_valid))
        metrics = _plain_metrics(evaluate(users_valid, y_valid, scores))
        epoch_record = {
            "epoch": float(epoch),
            "loss": float(np.mean(losses)),
            "elapsed_seconds": time.perf_counter() - started,
            **{key: float(value) for key, value in metrics.items() if key in {"GAUC", "nDCG@5", "primary"}},
        }
        history.append(epoch_record)
        if verbose:
            print(
                f"epoch {epoch:02d} | loss {epoch_record['loss']:.4f} | "
                f"GAUC {metrics['GAUC']:.4f} | nDCG@5 {metrics['nDCG@5']:.4f} | "
                f"primary {metrics['primary']:.4f} | {epoch_record['elapsed_seconds']:.1f}s"
            )

        if metrics["primary"] > best_primary + 1e-5:
            best_primary = metrics["primary"]
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(f"early stop at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("FM training produced no checkpoint.")
    model.V, model.W, model.b = best_state
    final_scores = model.predict(X_valid)
    validate_scores(final_scores, len(y_valid))
    final_metrics = _plain_metrics(evaluate(users_valid, y_valid, final_scores))
    checkpoint = {
        "V": model.V,
        "W": model.W,
        "b": float(model.b),
        "metadata": {
            "model": "fm",
            "k": k,
            "lr": lr,
            "batch_size": batch_size,
            "seed": seed,
            "best_validation_primary": final_metrics["primary"],
        },
    }
    return BaselineOutcome(final_metrics, history=history, checkpoint=checkpoint)


def run_baseline(
    experiment: str,
    splits: dict[str, list[tuple[Any, ...]]],
    *,
    seed: int = 0,
    epochs: int = 40,
    verbose: bool = True,
) -> BaselineOutcome:
    if experiment == "random":
        return run_random_validation(splits, seed=seed)
    if experiment == "pop":
        return run_pop_validation(splits)
    if experiment == "fm":
        return run_fm_validation(splits, seed=seed, epochs=epochs, verbose=verbose)
    raise ValueError(f"Unknown baseline experiment: {experiment}")
