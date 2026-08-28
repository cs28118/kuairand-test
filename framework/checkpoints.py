"""Checkpoint persistence for validation-best models."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def save_fm_checkpoint(
    directory: str | Path,
    *,
    V: np.ndarray,
    W: np.ndarray,
    b: float,
    metadata: dict[str, Any],
) -> Path:
    """Save the minimal state required to restore a NumPy FM."""
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "model.npz"
    np.savez_compressed(checkpoint_path, V=V, W=W, b=np.asarray(b, dtype=np.float32))
    (output_dir / "checkpoint.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return checkpoint_path


def write_best_pointer(run_dir: str | Path, best: dict[str, Any]) -> Path:
    path = Path(run_dir) / "best.json"
    path.write_text(json.dumps(best, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path

