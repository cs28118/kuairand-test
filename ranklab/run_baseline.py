"""Run the unmodified starter FM against validation only and record the outcome."""
from __future__ import annotations

import argparse
import hashlib
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from baseline import FM
from data import encode
from evaluate import evaluate

from .contracts import (
    GAUC,
    NDCG_AT_5,
    PRIMARY,
    assert_official_evaluator_unchanged,
    project_root,
    require_validation_split,
    sha256_file,
)
from .data import load_development_data
from .ledger import append_run


def code_hash() -> str:
    """Hash the RankLab code and the unmodified starter code it reuses."""
    files = [
        project_root() / "baseline.py",
        project_root() / "data.py",
        *sorted((project_root() / "ranklab").glob("*.py")),
    ]
    digest = hashlib.sha256()
    for source in files:
        digest.update(source.relative_to(project_root()).as_posix().encode("utf-8"))
        digest.update(sha256_file(source).encode("ascii"))
    return digest.hexdigest()


def run_validation_fm(
    development: dict[str, list[tuple]], *, k: int, lr: float, epochs: int,
    batch_size: int, patience: int, seed: int, verbose: bool,
) -> dict[str, Any]:
    """Equivalent FM training to the starter baseline, with no held-out access."""
    encoded, dimension = encode(development)
    train_x, train_y, _ = encoded["train"]
    valid_x, valid_y, valid_users = encoded["valid"]
    model = FM(dimension, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best_primary, best_state, stale_epochs = -float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(train_y))
        epoch_start = time.perf_counter()
        losses = [
            model.step(train_x[order[index:index + batch_size]], train_y[order[index:index + batch_size]])
            for index in range(0, len(order), batch_size)
        ]
        metrics = evaluate(valid_users, valid_y, model.predict(valid_x))
        if verbose:
            print(
                f"epoch {epoch:2d} | loss {np.mean(losses):.4f} | "
                f"valid {GAUC} {metrics[GAUC]:.4f} {NDCG_AT_5} {metrics[NDCG_AT_5]:.4f} "
                f"{PRIMARY} {metrics[PRIMARY]:.4f} | {time.perf_counter() - epoch_start:.1f}s"
            )
        if metrics[PRIMARY] > best_primary + 1e-5:
            best_primary, stale_epochs = metrics[PRIMARY], 0
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                if verbose:
                    print(f"early stop at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("FM did not produce a validation checkpoint.")
    model.V, model.W, model.b = best_state
    final_scores = model.predict(valid_x)
    final_metrics = evaluate(valid_users, valid_y, final_scores)
    return {key: float(final_metrics[key]) for key in (GAUC, NDCG_AT_5, PRIMARY)} | {
        "rows": int(final_metrics["rows"]),
        "users": int(final_metrics["users"]),
        "scores": np.asarray(final_scores, dtype=np.float64),
    }


def save_validation_predictions(
    run_id: str, development: dict[str, list[tuple]], scores: np.ndarray,
    *, artifacts_dir: Path | None = None,
) -> Path:
    """Persist the exact validation vector, including redundant alignment guards."""
    valid_rows = development["valid"]
    prediction_scores = np.asarray(scores, dtype=np.float64)
    if prediction_scores.ndim != 1 or len(prediction_scores) != len(valid_rows):
        raise ValueError("Validation prediction count does not match development validation rows.")
    if not np.isfinite(prediction_scores).all():
        raise ValueError("Validation predictions contain NaN or infinity.")
    destination = (artifacts_dir or project_root() / "artifacts") / "runs" / run_id
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "valid_predictions.npz"
    np.savez_compressed(
        path,
        row_id=np.arange(len(valid_rows), dtype=np.int64),
        user_id=np.asarray([row[1] for row in valid_rows]),
        video_id=np.asarray([row[2] for row in valid_rows]),
        label=np.asarray([row[6] for row in valid_rows], dtype=np.int8),
        score=prediction_scores,
    )
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validation-only RankLab starter FM baseline")
    parser.add_argument("--data-dir", default="./KuaiRand-Pure/data")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_id = f"fm-valid-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    started = time.perf_counter()
    configuration: dict[str, Any] = {
        "model": "starter_fm",
        "implementation": "baseline.FM (unmodified)",
        "evaluation_split": require_validation_split("valid"),
        "k": args.k,
        "lr": args.lr,
        "max_epochs": args.epochs,
        "batch_size": args.batch_size,
        "early_stopping_patience": args.patience,
    }
    record: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "The supplied FM is a reproducible validation-only baseline.",
        "configuration": configuration,
        "seed": args.seed,
        "code_hash": code_hash(),
        "runtime_seconds": None,
        GAUC: None,
        NDCG_AT_5: None,
        PRIMARY: None,
        "status": "failed",
        "error_recovery_note": "",
    }
    try:
        assert_official_evaluator_unchanged()
        development = load_development_data(args.data_dir)
        if not args.quiet:
            print(f"loaded train={len(development['train'])} valid={len(development['valid'])}")
        metrics = run_validation_fm(
            development, k=args.k, lr=args.lr, epochs=args.epochs,
            batch_size=args.batch_size, patience=args.patience, seed=args.seed,
            verbose=not args.quiet,
        )
        # Confirm the exact official implementation stayed intact for the full run.
        assert_official_evaluator_unchanged()
        prediction_path = save_validation_predictions(run_id, development, metrics["scores"])
        record.update({GAUC: metrics[GAUC], NDCG_AT_5: metrics[NDCG_AT_5], PRIMARY: metrics[PRIMARY]})
        record["status"] = "success"
        record["error_recovery_note"] = "none"
        print(f"validation | {GAUC} {metrics[GAUC]:.4f} | {NDCG_AT_5} {metrics[NDCG_AT_5]:.4f} | {PRIMARY} {metrics[PRIMARY]:.4f}")
        print(f"saved validation predictions: {prediction_path}")
        return_code = 0
    except Exception as exc:
        record["status"] = "rejected" if exc.__class__.__name__ == "ContractViolation" else "failed"
        record["error_recovery_note"] = f"{type(exc).__name__}: {exc}"
        print(f"baseline {record['status']}: {record['error_recovery_note']}")
        return_code = 1
    finally:
        record["runtime_seconds"] = round(time.perf_counter() - started, 6)
        try:
            append_run(record)
        except Exception:
            traceback.print_exc()
            return_code = 1
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
