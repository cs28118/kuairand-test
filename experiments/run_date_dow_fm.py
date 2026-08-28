"""One development-only FM candidate used by ExperimentSpec iteration 1."""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from safe_evaluate import evaluate

# baseline.py imports the module name ``evaluate``. Make it resolve to the
# non-protected copy above; evaluate.py is absent from the Docker workspace.
sys.modules["evaluate"] = sys.modules["safe_evaluate"]
from baseline import FM  # noqa: E402
from data import encode, load  # noqa: E402


def main() -> None:
    seed = int(os.environ.get("EXPERIMENT_SEED", "42"))
    data_dir = os.environ.get("KUAIRAND_DATA_DIR", "./KuaiRand-Pure/data")
    splits = load(data_dir)
    development = {name: splits[name] for name in ("train", "valid")}
    encoded, dimension = encode(development)
    x_train, y_train, _ = encoded["train"]
    x_valid, y_valid, users_valid = encoded["valid"]
    model = FM(dimension, k=16, lr=0.001, seed=seed)
    rng = np.random.default_rng(seed)
    best_primary = -float("inf")
    best_state = None
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, 41):
        indices = rng.permutation(len(y_train))
        losses = [
            model.step(x_train[indices[start:start + 8192]], y_train[indices[start:start + 8192]])
            for start in range(0, len(indices), 8192)
        ]
        metrics = evaluate(users_valid, y_valid, model.predict(x_valid))
        print(f"epoch {epoch:02d} loss {np.mean(losses):.4f} primary {metrics['primary']:.6f}", flush=True)
        if metrics["primary"] > best_primary + 1e-5:
            best_primary = metrics["primary"]
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
            stale = 0
        else:
            stale += 1
            if stale >= 4:
                break
    if best_state is None:
        raise RuntimeError("candidate produced no validation checkpoint")
    model.V, model.W, model.b = best_state
    final_metrics = evaluate(users_valid, y_valid, model.predict(x_valid))
    final_metrics = {
        key: float(value) if isinstance(value, (np.floating, float)) else int(value)
        for key, value in final_metrics.items()
    }
    result_path = os.environ["EXPERIMENT_RESULT_PATH"]
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "completed",
                "metrics": final_metrics,
                "artifacts": [],
                "token_usage": {
                    "model": None,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cached_tokens": 0,
                    "estimated_cost": 0.0,
                },
                "metadata": {"elapsed_seconds": time.perf_counter() - started, "epochs": epoch},
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
