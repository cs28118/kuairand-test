"""Train-prefix feature construction for validation diagnostics.

All aggregates in this module are fitted exclusively on the development train
split.  They are diagnostic attributes, never model inputs.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class Impression:
    date: int
    user_id: str
    video_id: str
    author_id: str
    tab: str
    hour: int
    duration_ms: float
    label: int
    video_type: str = "UNK"
    tag: str = "UNK"
    upload_date: date | None = None


@dataclass(frozen=True)
class TrainFeatures:
    user_counts: Mapping[str, int]
    user_long_view_rates: Mapping[str, float]
    video_exposures: Mapping[str, int]
    user_videos: frozenset[tuple[str, str]]
    user_authors: frozenset[tuple[str, str]]
    duration_edges: tuple[float, ...]
    engagement_edges: tuple[float, ...]
    popularity_edges: tuple[float, float]


def _quantile_edges(values: Iterable[float], quantiles: tuple[float, ...]) -> tuple[float, ...]:
    materialized = list(values)
    if not materialized:
        return ()
    return tuple(float(x) for x in np.quantile(np.asarray(materialized), quantiles))


def build_train_features(train_rows: Iterable[Impression]) -> TrainFeatures:
    """Fit all diagnostics aggregates using train impressions only."""
    rows = list(train_rows)
    user_counts = Counter(row.user_id for row in rows)
    user_positives = Counter(row.user_id for row in rows if row.label)
    rates = {user: user_positives[user] / count for user, count in user_counts.items()}
    exposures = Counter(row.video_id for row in rows)
    nonzero_popularity = list(exposures.values())
    return TrainFeatures(
        user_counts=dict(user_counts),
        user_long_view_rates=rates,
        video_exposures=dict(exposures),
        user_videos=frozenset((row.user_id, row.video_id) for row in rows),
        user_authors=frozenset((row.user_id, row.author_id) for row in rows),
        duration_edges=_quantile_edges((row.duration_ms for row in rows), (0.25, 0.5, 0.75)),
        engagement_edges=_quantile_edges(rates.values(), (0.25, 0.5, 0.75)),
        popularity_edges=tuple(_quantile_edges(nonzero_popularity, (0.5, 0.9))),  # tail / mid / head
    )


def history_bucket(count: int) -> str:
    if count == 0:
        return "0"
    if count <= 10:
        return "1-10"
    if count <= 50:
        return "11-50"
    if count <= 200:
        return "51-200"
    return "200+"


def quantile_bucket(value: float | None, edges: tuple[float, ...], prefix: str = "Q") -> str:
    if value is None:
        return "unknown"
    return f"{prefix}{int(np.searchsorted(edges, value, side='right')) + 1}"


def popularity_bucket(count: int, edges: tuple[float, float]) -> str:
    if count == 0:
        return "cold"
    if not edges:
        return "tail"
    if count <= edges[0]:
        return "tail"
    if count <= edges[1]:
        return "mid"
    return "head"


def duration_bucket(value: float, edges: tuple[float, ...]) -> str:
    return quantile_bucket(value, edges, prefix="Q")


def hour_bucket(hour: int) -> str:
    if hour < 6:
        return "00-05"
    if hour < 12:
        return "06-11"
    if hour < 18:
        return "12-17"
    return "18-23"


def video_age_bucket(row: Impression) -> str:
    if row.upload_date is None:
        return "unknown"
    logged = date(row.date // 10000, (row.date // 100) % 100, row.date % 100)
    days = (logged - row.upload_date).days
    if days < 0:
        return "future_or_invalid"
    if days <= 7:
        return "0-7d"
    if days <= 30:
        return "8-30d"
    if days <= 90:
        return "31-90d"
    return "91d+"


def validation_user_counts(rows: Iterable[Impression]) -> tuple[Mapping[str, int], Mapping[str, int]]:
    """Return validation-only difficulty descriptors; never use these for training."""
    row_list = list(rows)
    return (
        dict(Counter(row.user_id for row in row_list)),
        dict(Counter(row.user_id for row in row_list if row.label)),
    )


def difficulty_bucket(value: int, edges: tuple[int, ...]) -> str:
    if not edges:
        return "all"
    return f"Q{int(np.searchsorted(edges, value, side='right')) + 1}"
