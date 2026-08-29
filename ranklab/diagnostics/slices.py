"""Slice definitions and official/restricted metric calculation."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

import numpy as np

from evaluate import evaluate

from .features import (
    Impression,
    TrainFeatures,
    difficulty_bucket,
    duration_bucket,
    history_bucket,
    hour_bucket,
    popularity_bucket,
    quantile_bucket,
    validation_user_counts,
    video_age_bucket,
)


def _metric_record(
    *, grouping: str, group: str, rows: list[Impression], scores: np.ndarray,
    overall_primary: float, scope: str, support_threshold: int,
) -> dict:
    labels = [row.label for row in rows]
    users = [row.user_id for row in rows]
    metrics = evaluate(users, labels, scores.tolist()) if rows else {"GAUC": 0.5, "nDCG@5": 0.0, "primary": 0.25}
    user_count = len(set(users))
    supported = user_count >= support_threshold
    return {
        "slice_grouping": grouping,
        "group": group,
        "metric_scope": scope,
        "support_status": "supported" if supported else "below_support",
        "rows": len(rows),
        "users": user_count,
        "positives": int(sum(labels)),
        "GAUC": float(metrics["GAUC"]),
        "nDCG@5": float(metrics["nDCG@5"]),
        "primary": float(metrics["primary"]),
        "delta_primary": float(metrics["primary"] - overall_primary),
    }


def _group_rows(rows: list[Impression], groups: Iterable[str]) -> Mapping[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        grouped[group].append(index)
    return grouped


def _restricted_slices(
    definitions: Mapping[str, Iterable[str]], rows: list[Impression], scores: np.ndarray,
    overall_primary: float, support_threshold: int,
) -> list[dict]:
    output: list[dict] = []
    for name, groups in definitions.items():
        for group, indexes in sorted(_group_rows(rows, groups).items()):
            output.append(_metric_record(
                grouping=name, group=group, rows=[rows[i] for i in indexes], scores=scores[indexes],
                overall_primary=overall_primary, scope="restricted_within_slice_diagnostic",
                support_threshold=support_threshold,
            ))
    return output


def build_slices(
    valid_rows: list[Impression], scores: np.ndarray, features: TrainFeatures,
    overall_primary: float, support_threshold: int,
) -> tuple[list[dict], list[dict]]:
    """Build user full-list slices and restricted row-level diagnostic slices."""
    valid_exposures, valid_positives = validation_user_counts(valid_rows)
    exposure_edges = tuple(int(x) for x in np.quantile(list(valid_exposures.values()), (0.25, 0.5, 0.75))) if valid_exposures else ()
    # Include zero-positive users in the distribution; they are part of official nDCG.
    positive_edges = tuple(int(x) for x in np.quantile(
        [valid_positives.get(user_id, 0) for user_id in valid_exposures], (0.25, 0.5, 0.75)
    )) if valid_exposures else ()
    user_definitions = {
        "user_history_train_interactions": [history_bucket(features.user_counts.get(row.user_id, 0)) for row in valid_rows],
        "user_engagement_train_long_view_rate": [quantile_bucket(features.user_long_view_rates.get(row.user_id), features.engagement_edges) for row in valid_rows],
        # These two are validation-label diagnostics only, never candidate features.
        "evaluation_difficulty_validation_exposures_per_user": [difficulty_bucket(valid_exposures[row.user_id], exposure_edges) for row in valid_rows],
        "evaluation_difficulty_validation_positives_per_user": [difficulty_bucket(valid_positives.get(row.user_id, 0), positive_edges) for row in valid_rows],
    }
    user_slices: list[dict] = []
    for name, per_row_groups in user_definitions.items():
        by_user: dict[str, str] = {}
        for row, group in zip(valid_rows, per_row_groups):
            by_user.setdefault(row.user_id, group)
        for group in sorted(set(by_user.values())):
            indexes = [i for i, row in enumerate(valid_rows) if by_user[row.user_id] == group]
            user_slices.append(_metric_record(
                grouping=name, group=group, rows=[valid_rows[i] for i in indexes], scores=scores[indexes],
                overall_primary=overall_primary, scope="full_validation_impression_lists_for_assigned_users",
                support_threshold=support_threshold,
            ))

    row_definitions = {
        "item_popularity_train_exposures": [popularity_bucket(features.video_exposures.get(row.video_id, 0), features.popularity_edges) for row in valid_rows],
        "item_familiarity_train_saw_video": ["yes" if (row.user_id, row.video_id) in features.user_videos else "no" for row in valid_rows],
        "item_familiarity_train_saw_author": ["yes" if (row.user_id, row.author_id) in features.user_authors else "no" for row in valid_rows],
        "context_tab": [row.tab or "unknown" for row in valid_rows],
        "context_hour_bucket": [hour_bucket(row.hour) for row in valid_rows],
        "context_duration_bucket": [duration_bucket(row.duration_ms, features.duration_edges) for row in valid_rows],
        "content_author": [row.author_id or "unknown" for row in valid_rows],
        "content_tag": [row.tag or "unknown" for row in valid_rows],
        "content_video_type": [row.video_type or "unknown" for row in valid_rows],
        "content_video_age": [video_age_bucket(row) for row in valid_rows],
    }
    return user_slices, _restricted_slices(row_definitions, valid_rows, scores, overall_primary, support_threshold)
