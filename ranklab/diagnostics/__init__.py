"""Validation-only diagnostics for RankLab experiment runs."""

__all__ = ["generate_report"]


def generate_report(*args, **kwargs):
    """Lazily expose the report builder without interfering with ``-m`` execution."""
    from .report import generate_report as _generate_report

    return _generate_report(*args, **kwargs)
