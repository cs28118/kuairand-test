"""The fixed, dependency-free RankLab experiment catalogue."""
from __future__ import annotations

from typing import Any


CATALOGUE: dict[str, dict[str, Any]] = {
    "ranking_loss": {
        "family": "ranking_loss",
        "eligibility": "Overall validation nDCG@5 is weaker than GAUC.",
        "target_slice_patterns": ["*"],
        "expected_cost_minutes": 20,
        "future_allowed_files": ["ranklab/losses/ranking_loss.py", "ranklab/configs/ranking_loss.json"],
        "success_criterion": "Validation primary improves by at least 0.002.",
        "rollback_condition": "Reject if validation primary does not improve.",
        "risks": ["Within-user objectives can reduce global calibration."],
    },
    "causal_features": {
        "family": "causal_features",
        "eligibility": "A supported cold/tail video or author/content weakness is present.",
        "target_slice_patterns": [
            "item_popularity_train_exposures=cold",
            "item_popularity_train_exposures=tail",
            "content_author=*",
            "content_tag=*",
            "content_video_type=*",
            "item_familiarity_train_saw_author=yes",
        ],
        "expected_cost_minutes": 30,
        "future_allowed_files": ["ranklab/features/causal_features.py", "ranklab/configs/causal_features.json"],
        "success_criterion": "Validation primary improves by at least 0.002.",
        "rollback_condition": "Reject if validation primary does not improve.",
        "risks": ["Feature timestamps must remain train-prefix causal."],
    },
    "sequence": {
        "family": "sequence",
        "eligibility": "A weak user slice has substantial train-prefix history.",
        "target_slice_patterns": [
            "user_history_train_interactions=11-50",
            "user_history_train_interactions=51-200",
            "user_history_train_interactions=200+",
        ],
        "expected_cost_minutes": 60,
        "future_allowed_files": ["ranklab/models/sequence.py", "ranklab/configs/sequence.json"],
        "success_criterion": "Validation primary improves by at least 0.002.",
        "rollback_condition": "Reject if validation primary does not improve.",
        "risks": ["Long histories may add cost without improving ranking."],
    },
    "multitask": {
        "family": "multitask",
        "eligibility": "A sparse-history or deep-feedback/engagement weakness is present.",
        "target_slice_patterns": [
            "user_history_train_interactions=0",
            "user_history_train_interactions=1-10",
            "user_engagement_train_long_view_rate=*",
            "engagement_*",
        ],
        "expected_cost_minutes": 50,
        "future_allowed_files": ["ranklab/models/multitask.py", "ranklab/configs/multitask.json"],
        "success_criterion": "Validation primary improves by at least 0.002.",
        "rollback_condition": "Reject if validation primary does not improve.",
        "risks": ["Auxiliary objectives can distract from the primary metric."],
    },
    "ensemble": {
        "family": "ensemble",
        "eligibility": "Complementary accepted models exist and an ensemble opportunity is reported.",
        "target_slice_patterns": ["ensemble_*", "model_complementarity=*"],
        "expected_cost_minutes": 20,
        "future_allowed_files": ["ranklab/ensembles/blend.py", "ranklab/configs/ensemble.json"],
        "success_criterion": "Validation primary improves by at least 0.002.",
        "rollback_condition": "Reject if validation primary does not improve.",
        "risks": ["Blending can overfit a single validation result."],
    },
}


def catalogue_entry(family: str) -> dict[str, Any] | None:
    """Return a catalogue entry without exposing a mutable global object."""
    entry = CATALOGUE.get(family)
    return dict(entry) if entry is not None else None
