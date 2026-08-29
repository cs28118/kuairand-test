"""Development-data loading that never exposes test rows to RankLab."""
from __future__ import annotations

import csv
from pathlib import Path

from .contracts import TRAIN, VALID

LABEL = "long_view"
TRAIN_DATES = (20220408, 20220421)
VALID_DATES = (20220422, 20220428)


def _in_range(value: int, bounds: tuple[int, int]) -> bool:
    return bounds[0] <= value <= bounds[1]


def load_development_data(data_dir: str | Path) -> dict[str, list[tuple]]:
    """Load train and validation rows in their source order.

    The source log covers later dates too, but each row is classified immediately:
    only train and validation tuples are retained. Test rows and labels are neither
    returned nor written to disk by this function.
    """
    root = Path(data_dir)
    feature_file = root / "video_features_basic_pure.csv"
    if not feature_file.is_file():
        raise FileNotFoundError(f"Missing video feature file: {feature_file}")

    video_to_author: dict[str, str] = {}
    with feature_file.open("r", newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            video_to_author[row["video_id"]] = row["author_id"]

    development: dict[str, list[tuple]] = {TRAIN: [], VALID: []}
    log_files = (
        root / "log_standard_4_08_to_4_21_pure.csv",
        root / "log_standard_4_22_to_5_08_pure.csv",
    )
    for log_file in log_files:
        if not log_file.is_file():
            raise FileNotFoundError(f"Missing standard log file: {log_file}")
        with log_file.open("r", newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                date = int(row["date"])
                if _in_range(date, TRAIN_DATES):
                    split = TRAIN
                elif _in_range(date, VALID_DATES):
                    split = VALID
                else:
                    # Do not materialize held-out rows or their labels.
                    continue
                development[split].append(
                    (
                        date,
                        row["user_id"],
                        row["video_id"],
                        video_to_author.get(row["video_id"], "UNK"),
                        row["tab"],
                        float(row["duration_ms"]),
                        1 if row[LABEL] != "0" else 0,
                    )
                )
    return development

