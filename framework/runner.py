"""CLI and reusable function for validation-only baseline experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

from .benchmark import load_development_splits, run_baseline
from .checkpoints import save_fm_checkpoint, write_best_pointer
from .config import BenchmarkConfig, load_benchmark_config
from .guardrails import verify_official_files
from .state import RunStore


def execute_baseline(
    experiment: str,
    *,
    config: BenchmarkConfig,
    run_store: RunStore,
    seed: int = 0,
    epochs: int = 40,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run one baseline and persist all milestone artifacts."""
    evaluator_hashes = verify_official_files()
    run_store.initialize(
        {
            "framework_version": 1,
            "data_dir": str(config.data_dir),
            "development_splits": list(config.development_splits),
            "evaluator_sha256": evaluator_hashes,
            "primary_metric": config.primary_metric,
        }
    )
    started = time.perf_counter()
    splits = load_development_splits(config.data_dir)
    outcome = run_baseline(experiment, splits, seed=seed, epochs=epochs, verbose=verbose)
    elapsed = time.perf_counter() - started

    artifact_dir = run_store.run_dir / f"iter_{run_store.read_state()['iterations'] + 1:03d}"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    (artifact_dir / "metrics.json").write_text(
        json.dumps(outcome.metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (artifact_dir / "training_history.json").write_text(
        json.dumps(outcome.history, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    checkpoint_path: str | None = None
    if outcome.checkpoint is not None:
        checkpoint_path = str(
            save_fm_checkpoint(
                artifact_dir,
                V=outcome.checkpoint["V"],
                W=outcome.checkpoint["W"],
                b=outcome.checkpoint["b"],
                metadata=outcome.checkpoint["metadata"],
            )
        )

    record = run_store.append_iteration(
        {
            "experiment": experiment,
            "kind": "baseline",
            "hypothesis": f"Reproduce the validation-only {experiment} reference baseline.",
            "status": "completed",
            "metrics": outcome.metrics,
            "elapsed_seconds": elapsed,
            "seed": seed,
            "checkpoint": checkpoint_path,
            "artifacts": str(artifact_dir),
            "errors": [],
            "recovery": None,
            "llm_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }
    )
    state = run_store.read_state()
    if state["best"] is not None:
        write_best_pointer(run_store.run_dir, state["best"])
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a validation-only starter-kit baseline.")
    parser.add_argument("--experiment", required=True, choices=["random", "pop", "fm"])
    parser.add_argument("--config", default=None, help="Path to configs/benchmark.json")
    parser.add_argument("--runs-dir", default=None)
    parser.add_argument("--run-id", default=None, help="Append to an existing run when supplied")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    config = load_benchmark_config(args.config)
    store = RunStore(args.runs_dir, args.run_id)
    try:
        record = execute_baseline(
            args.experiment,
            config=config,
            run_store=store,
            seed=args.seed,
            epochs=args.epochs,
            verbose=not args.quiet,
        )
        store.complete()
    except Exception as exc:  # CLI boundary: persist a readable non-zero failure.
        print(f"framework runner failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
