# RankLab milestone checklist

## Milestone 1 — Foundation, safety contract, reproducible baseline

- [x] Created the `ranklab/` package and `artifacts/` directory.
- [x] Defined official splits (`train`, `valid`, `test`) and metrics (`GAUC`, `nDCG@5`, `primary`) in `ranklab/contracts.py`.
- [x] Added the official `evaluate.py` SHA-256 integrity check.
- [x] Reject use of the held-out split for experiment evaluation.
- [x] Defined convergence constants: `epsilon=0.002`, `patience_iterations=3`, and `max_iterations=50`.
- [x] Added a development-only data loader that returns train and validation rows in source order.
- [x] Prevented held-out rows and labels from being returned or persisted by RankLab.
- [x] Added an append-only JSONL experiment ledger at `artifacts/iterations.jsonl`.
- [x] Recorded run ID, timestamp, hypothesis, configuration, seed, code hash, runtime, metrics, status, and recovery note for each run.
- [x] Added `python -m ranklab.run_baseline --data-dir ./KuaiRand-Pure/data --seed 0` support.
- [x] Reused the starter `baseline.FM` class without modifying `baseline.py`.
- [x] Trained on train and scored validation only.
- [x] Verified `evaluate.py` still matches its original SHA-256 checksum.
- [x] Completed a canonical seed-0 run: validation GAUC `0.66713`, nDCG@5 `0.53581`, primary `0.60147`.
- [x] Confirmed the primary score is close to the published FM validation primary of `0.6016`.
- [x] Confirmed the ledger's latest record is valid JSON and contains no held-out score.

## Milestone 2 — Validation diagnostics and opportunity report

- [x] Added `ranklab/diagnostics/__init__.py`, `features.py`, `slices.py`, and `report.py`.
- [x] Persist aligned successful-run validation predictions in `artifacts/runs/<run_id>/valid_predictions.npz`.
- [x] Persist required arrays: `row_id`, `user_id`, `video_id`, `label`, and `score`.
- [x] Validate prediction row count, contiguous row IDs, validation source-order user/video IDs, labels, and finite scores.
- [x] Calculate overall GAUC, nDCG@5, and primary with the unmodified official evaluator.
- [x] Verify overall metrics match the successful ledger record within floating-point tolerance.
- [x] Build train-prefix user-history slices: `0`, `1-10`, `11-50`, `51-200`, and `200+` interactions.
- [x] Build train-prefix user-engagement quantiles from long-view rate.
- [x] Build train-prefix item-popularity groups: cold, tail, mid, and head.
- [x] Build train-prefix user familiarity slices for previously seen video and author.
- [x] Build context slices for tab, hour, and duration buckets.
- [x] Build content slices for author, tag, video type, and video age at the impression date.
- [x] Build validation-only difficulty descriptors for impressions and positives per user; never use them as training features or recommendations.
- [x] Evaluate user slices over every assigned user's complete validation impression list.
- [x] Clearly mark row slices as restricted within-slice diagnostic metrics, not official primary-score components.
- [x] Mark groups below the configurable user support threshold (default: 200 users).
- [x] Generate exactly three structured opportunities with weak slice, evidence, likely cause, and candidate experiment family.
- [x] Reject any test-split request before filesystem data access.
- [x] Write `artifacts/reports/<run_id>/diagnostics.json` and `diagnostics.md` with data policy `{used_splits: [train, valid], test_accessed: false}`.
- [x] Completed a real validation-only report for `fm-valid-20260829T152749Z-e36d5761`: GAUC `0.667133`, nDCG@5 `0.535805`, primary `0.601469`.
- [x] Reran the real report command with Python warnings promoted to errors; it completed cleanly.

### Automated test code

Canonical source: [`tests/test_diagnostics.py`](../tests/test_diagnostics.py). The tests use a temporary miniature KuaiRand-shaped dataset, so they never inspect the project test split.

```python
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ranklab.contracts import ContractViolation
from ranklab.diagnostics.features import Impression
from ranklab.diagnostics.report import AlignmentError, generate_report, load_predictions, load_validation_rows


def _write_csv(path: Path, header: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _make_data(root: Path) -> Path:
    data = root / "data"
    data.mkdir()
    _write_csv(data / "video_features_basic_pure.csv", ["video_id", "author_id", "video_type", "upload_dt", "tag"], [
        {"video_id": "v1", "author_id": "a1", "video_type": "NORMAL", "upload_dt": "2022-04-01", "tag": "1"},
        {"video_id": "v2", "author_id": "a2", "video_type": "AD", "upload_dt": "2022-04-20", "tag": "2"},
        {"video_id": "v3", "author_id": "a3", "video_type": "NORMAL", "upload_dt": "", "tag": ""},
    ])
    columns = ["user_id", "video_id", "date", "hourmin", "long_view", "duration_ms", "tab"]
    _write_csv(data / "log_standard_4_08_to_4_21_pure.csv", columns, [
        {"user_id": "u1", "video_id": "v1", "date": "20220410", "hourmin": "100", "long_view": "1", "duration_ms": "100", "tab": "1"},
        {"user_id": "u1", "video_id": "v2", "date": "20220411", "hourmin": "1200", "long_view": "0", "duration_ms": "200", "tab": "1"},
        {"user_id": "u2", "video_id": "v1", "date": "20220412", "hourmin": "1800", "long_view": "0", "duration_ms": "300", "tab": "2"},
    ])
    _write_csv(data / "log_standard_4_22_to_5_08_pure.csv", columns, [
        {"user_id": "u1", "video_id": "v1", "date": "20220422", "hourmin": "200", "long_view": "1", "duration_ms": "100", "tab": "1"},
        {"user_id": "u1", "video_id": "v3", "date": "20220422", "hourmin": "1000", "long_view": "0", "duration_ms": "400", "tab": "2"},
        {"user_id": "u2", "video_id": "v2", "date": "20220423", "hourmin": "1600", "long_view": "0", "duration_ms": "200", "tab": "1"},
        {"user_id": "u2", "video_id": "v3", "date": "20220423", "hourmin": "2300", "long_view": "0", "duration_ms": "400", "tab": "2"},
    ])
    return data


def _write_predictions(artifacts: Path, run_id: str, rows: list[Impression], scores: list[float]) -> None:
    destination = artifacts / "runs" / run_id
    destination.mkdir(parents=True)
    np.savez_compressed(
        destination / "valid_predictions.npz",
        row_id=np.arange(len(rows)), user_id=np.asarray([row.user_id for row in rows]),
        video_id=np.asarray([row.video_id for row in rows]), label=np.asarray([row.label for row in rows]),
        score=np.asarray(scores),
    )


class DiagnosticsTests(unittest.TestCase):
    def test_test_split_request_is_rejected_before_loading_data(self) -> None:
        with self.assertRaises(ContractViolation):
            load_validation_rows("this-path-does-not-need-to-exist", split="test")

    def test_misaligned_prediction_is_rejected(self) -> None:
        rows = [Impression(20220422, "u1", "v1", "a1", "1", 1, 10.0, 1)]
        with tempfile.TemporaryDirectory() as temp:
            artifacts = Path(temp) / "artifacts"
            _write_predictions(artifacts, "bad", rows, [0.1])
            path = artifacts / "runs" / "bad" / "valid_predictions.npz"
            np.savez_compressed(path, row_id=np.array([0]), user_id=np.array(["other"]), video_id=np.array(["v1"]), label=np.array([1]), score=np.array([0.1]))
            with self.assertRaisesRegex(AlignmentError, "user_id"):
                load_predictions("bad", rows, artifacts_dir=artifacts)

    def test_report_is_validation_only_and_handles_zero_positive_users(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = _make_data(root)
            artifacts = root / "artifacts"
            _, valid = load_validation_rows(data)
            _write_predictions(artifacts, "run-1", valid, [0.9, 0.1, 0.2, 0.3])
            report = generate_report("run-1", data, support_threshold=1, artifacts_dir=artifacts)
            self.assertEqual(report["data_policy"], {"used_splits": ["train", "valid"], "test_accessed": False})
            self.assertEqual(len(report["top_opportunities"]), 3)
            self.assertTrue(all(not item["weak_slice"].startswith("evaluation_difficulty_") for item in report["top_opportunities"]))
            self.assertTrue(any(row["positives"] == 0 for row in report["user_slices"]))
            self.assertTrue(all(row["metric_scope"] == "restricted_within_slice_diagnostic" for row in report["row_slices"]))
            self.assertTrue((artifacts / "reports" / "run-1" / "diagnostics.json").is_file())
            self.assertTrue((artifacts / "reports" / "run-1" / "diagnostics.md").is_file())

    def test_ledger_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = _make_data(root)
            artifacts = root / "artifacts"
            _, valid = load_validation_rows(data)
            _write_predictions(artifacts, "run-2", valid, [0.9, 0.1, 0.2, 0.3])
            artifacts.mkdir(exist_ok=True)
            (artifacts / "iterations.jsonl").write_text(json.dumps({"run_id": "run-2", "status": "success", "GAUC": 0.0, "nDCG@5": 0.0, "primary": 0.0}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(AlignmentError, "ledger"):
                generate_report("run-2", data, support_threshold=1, artifacts_dir=artifacts)


if __name__ == "__main__":
    unittest.main()
```

### Test commands and verified results

Run the Milestone 2 tests with warnings treated as errors:

```powershell
& 'C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe' -W error -m unittest tests/test_diagnostics.py -v
```

Run all local tests, including the separate Milestone 3 AIDE-adapter tests:

```powershell
& 'C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe' -W error -m unittest discover -s tests -v
```

Verified result: 7 tests passed with no warnings. The real diagnostics command was also rerun with warnings treated as errors and completed successfully:

```powershell
& 'C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe' -W error -m ranklab.diagnostics.report --run-id fm-valid-20260829T152749Z-e36d5761 --data-dir .\KuaiRand-Pure\data
```

## Next milestone

- [x] Milestone 2 — Build validation metric slices and a readable diagnostics report.
- [x] Milestone 3 — Build guarded AIDE integration with structured hypothesis validation.
- [ ] Milestone 4 — Create the standard isolated experiment runner.
