"""Create an evidence report for one validation-only experiment run."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from evaluate import evaluate

from ..contracts import assert_official_evaluator_unchanged, project_root, require_validation_split
from .features import Impression, build_train_features
from .slices import build_slices

REQUIRED_PREDICTION_KEYS = frozenset({"row_id", "user_id", "video_id", "label", "score"})


class AlignmentError(ValueError):
    """Raised when a prediction artifact is not in validation source order."""


def load_validation_rows(data_dir: str | Path, *, split: str = "valid") -> tuple[list[Impression], list[Impression]]:
    """Load train/valid only; a test request is explicitly rejected before any log use."""
    require_validation_split(split)
    root = Path(data_dir)
    videos: dict[str, tuple[str, str, str, date | None]] = {}
    with (root / "video_features_basic_pure.csv").open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            try:
                uploaded = date.fromisoformat(row.get("upload_dt", ""))
            except ValueError:
                uploaded = None
            videos[row["video_id"]] = (row.get("author_id", "UNK"), row.get("video_type", "UNK"), row.get("tag", "UNK"), uploaded)

    train: list[Impression] = []
    valid: list[Impression] = []
    for filename in ("log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv"):
        with (root / filename).open(newline="", encoding="utf-8") as source:
            for raw in csv.DictReader(source):
                logged_date = int(raw["date"])
                if 20220408 <= logged_date <= 20220421:
                    target = train
                elif 20220422 <= logged_date <= 20220428:
                    target = valid
                else:
                    # Held-out rows (and importantly their labels) are not materialized.
                    continue
                author, video_type, tag, uploaded = videos.get(raw["video_id"], ("UNK", "UNK", "UNK", None))
                target.append(Impression(
                    date=logged_date, user_id=raw["user_id"], video_id=raw["video_id"], author_id=author,
                    tab=raw.get("tab", "unknown"), hour=int(raw.get("hourmin", "0")) // 100,
                    duration_ms=float(raw.get("duration_ms", 0.0)), label=1 if raw["long_view"] != "0" else 0,
                    video_type=video_type, tag=tag, upload_date=uploaded,
                ))
    return train, valid


def load_predictions(run_id: str, valid_rows: list[Impression], *, artifacts_dir: Path | None = None) -> np.ndarray:
    root = artifacts_dir or project_root() / "artifacts"
    path = root / "runs" / run_id / "valid_predictions.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing validation predictions for run {run_id!r}: {path}")
    with np.load(path, allow_pickle=False) as artifact:
        missing = REQUIRED_PREDICTION_KEYS.difference(artifact.files)
        if missing:
            raise AlignmentError(f"Prediction artifact is missing arrays: {sorted(missing)}")
        arrays = {key: np.asarray(artifact[key]) for key in REQUIRED_PREDICTION_KEYS}
    length = len(valid_rows)
    if any(values.ndim != 1 or len(values) != length for values in arrays.values()):
        raise AlignmentError("Each prediction array must be one-dimensional and match validation row count.")
    if not np.array_equal(arrays["row_id"], np.arange(length)):
        raise AlignmentError("Prediction row_id must be contiguous and match validation source order.")
    expected_users = np.asarray([row.user_id for row in valid_rows]).astype(str)
    expected_videos = np.asarray([row.video_id for row in valid_rows]).astype(str)
    expected_labels = np.asarray([row.label for row in valid_rows])
    if not np.array_equal(arrays["user_id"].astype(str), expected_users):
        raise AlignmentError("Prediction user_id does not align with validation source order.")
    if not np.array_equal(arrays["video_id"].astype(str), expected_videos):
        raise AlignmentError("Prediction video_id does not align with validation source order.")
    if not np.array_equal(arrays["label"].astype(np.int64), expected_labels):
        raise AlignmentError("Prediction label does not align with development validation labels.")
    scores = arrays["score"].astype(np.float64)
    if not np.isfinite(scores).all():
        raise AlignmentError("Prediction score contains NaN or infinity.")
    return scores


def _ledger_metrics(run_id: str, artifacts_dir: Path) -> dict[str, float] | None:
    ledger = artifacts_dir / "iterations.jsonl"
    if not ledger.is_file():
        return None
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("run_id") == run_id and record.get("status") == "success":
            return {key: float(record[key]) for key in ("GAUC", "nDCG@5", "primary")}
    return None


def _opportunity_metadata(grouping: str) -> tuple[str, str]:
    if "popularity" in grouping:
        return ("ID-only scoring has little signal for rare or unseen videos.", "causal author/tag/content features or history-to-content matching")
    if "content" in grouping:
        return ("The baseline does not use the available author, tag, type, or age information to generalize across content.", "causal author/tag/content features or history-to-content matching")
    if "familiarity" in grouping or "history" in grouping:
        return ("The model lacks an explicit causal representation of prior user-item or user-author behavior.", "timestamp-causal sequence/history features")
    if "context" in grouping or "video_age" in grouping:
        return ("Temporal or impression context is not explicitly represented in the current FM baseline.", "causal temporal-context and recency features")
    if "engagement" in grouping:
        return ("A pointwise objective can underfit differences in user engagement regimes.", "within-user pairwise/listwise ranking loss")
    return ("The validation metric exposes a difficult ranking regime rather than a trainable label feature.", "ranking-loss and calibration experiments; do not use this diagnostic as a feature")


def rank_opportunities(user_slices: list[dict], row_slices: list[dict], overall_primary: float) -> list[dict]:
    candidates = [
        row for row in user_slices + row_slices
        if row["support_status"] == "supported" and not row["slice_grouping"].startswith("evaluation_difficulty_")
    ]
    # Prefer distinct experiment families so the next three experiments probe different hypotheses.
    candidates.sort(key=lambda row: (row["delta_primary"], row["users"], row["rows"]))
    selected: list[dict] = []
    seen_families: set[str] = set()
    for row in candidates:
        cause, family = _opportunity_metadata(row["slice_grouping"])
        if family in seen_families:
            continue
        selected.append({
            "weak_slice": f"{row['slice_grouping']}={row['group']}",
            "evidence": f"primary {row['primary']:.6f}, {row['delta_primary']:+.6f} versus overall validation primary {overall_primary:.6f}; {row['users']} users and {row['rows']} rows.",
            "likely_cause": cause,
            "candidate_experiment_family": family,
        })
        seen_families.add(family)
        if len(selected) == 3:
            return selected
    # Extremely small validation sets can have fewer than three supported families. Keep the
    # output contract useful without inventing unsupported metrics.
    for row in candidates:
        if len(selected) == 3:
            break
        cause, family = _opportunity_metadata(row["slice_grouping"])
        selected.append({
            "weak_slice": f"{row['slice_grouping']}={row['group']}",
            "evidence": f"primary {row['primary']:.6f}, {row['delta_primary']:+.6f} versus overall validation primary {overall_primary:.6f}; {row['users']} users and {row['rows']} rows.",
            "likely_cause": cause,
            "candidate_experiment_family": family,
        })
    # The real validation set has many supported groups. For a deliberately high support
    # threshold, retain a structured three-slot result instead of making downstream AIDE
    # parsing depend on report sparsity.
    while len(selected) < 3:
        selected.append({
            "weak_slice": "insufficient_supported_slice_coverage",
            "evidence": f"Only {len(candidates)} slice groups met the configured support threshold; overall validation primary is {overall_primary:.6f}.",
            "likely_cause": "The support threshold prevents reliable slice comparison.",
            "candidate_experiment_family": "collect more development evidence before prioritizing a model change",
        })
    return selected


def _markdown(report: dict[str, Any]) -> str:
    overall = report["overall_metrics"]
    lines = [
        f"# Validation diagnostics: {report['run_id']}",
        "",
        "This report uses train and validation only. Row-level values are restricted within-slice diagnostic metrics, not components of the official primary score.",
        "",
        "## Overall (official evaluator)",
        "",
        "| rows | users | positives | GAUC | nDCG@5 | primary |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {overall['rows']} | {overall['users']} | {overall['positives']} | {overall['GAUC']:.6f} | {overall['nDCG@5']:.6f} | {overall['primary']:.6f} |",
    ]
    for title, slices in (("User slices", report["user_slices"]), ("Row slices (diagnostic)", report["row_slices"])):
        lines += ["", f"## {title}", "", "| grouping | group | status | rows | users | positives | GAUC | nDCG@5 | primary | Δ primary |", "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for row in slices:
            lines.append(f"| {row['slice_grouping']} | {row['group']} | {row['support_status']} | {row['rows']} | {row['users']} | {row['positives']} | {row['GAUC']:.6f} | {row['nDCG@5']:.6f} | {row['primary']:.6f} | {row['delta_primary']:+.6f} |")
    lines += ["", "## Top opportunities", ""]
    for index, opportunity in enumerate(report["top_opportunities"], start=1):
        lines += [f"### {index}. {opportunity['weak_slice']}", "", f"- Evidence: {opportunity['evidence']}", f"- Likely cause: {opportunity['likely_cause']}", f"- Candidate experiment family: {opportunity['candidate_experiment_family']}", ""]
    return "\n".join(lines)


def generate_report(run_id: str, data_dir: str | Path, *, support_threshold: int = 200, artifacts_dir: str | Path | None = None) -> dict[str, Any]:
    """Generate and persist a validation-only diagnostics report for ``run_id``."""
    if support_threshold < 1:
        raise ValueError("support_threshold must be at least one user.")
    assert_official_evaluator_unchanged()
    artifacts = Path(artifacts_dir) if artifacts_dir is not None else project_root() / "artifacts"
    train_rows, valid_rows = load_validation_rows(data_dir)
    scores = load_predictions(run_id, valid_rows, artifacts_dir=artifacts)
    overall = evaluate([row.user_id for row in valid_rows], [row.label for row in valid_rows], scores.tolist())
    overall_metrics = {key: float(overall[key]) for key in ("GAUC", "nDCG@5", "primary")} | {"rows": int(overall["rows"]), "users": int(overall["users"]), "positives": int(sum(row.label for row in valid_rows))}
    ledger = _ledger_metrics(run_id, artifacts)
    if ledger is not None and not all(np.isclose(overall_metrics[key], ledger[key], rtol=1e-7, atol=1e-9) for key in ledger):
        raise AlignmentError(f"Official evaluator metrics do not match the ledger for run {run_id!r}.")
    user_slices, row_slices = build_slices(valid_rows, scores, build_train_features(train_rows), overall_metrics["primary"], support_threshold)
    report: dict[str, Any] = {
        "run_id": run_id,
        "overall_metrics": overall_metrics,
        "user_slices": user_slices,
        "row_slices": row_slices,
        "top_opportunities": rank_opportunities(user_slices, row_slices, overall_metrics["primary"]),
        "data_policy": {"used_splits": ["train", "valid"], "test_accessed": False},
        "support_threshold_users": support_threshold,
    }
    destination = artifacts / "reports" / run_id
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "diagnostics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (destination / "diagnostics.md").write_text(_markdown(report), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate validation-only RankLab diagnostics.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-dir", default="./KuaiRand-Pure/data")
    parser.add_argument("--support-threshold", type=int, default=200)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = generate_report(args.run_id, args.data_dir, support_threshold=args.support_threshold)
    print(f"wrote artifacts/reports/{args.run_id}/diagnostics.json and diagnostics.md ({len(report['top_opportunities'])} opportunities)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
