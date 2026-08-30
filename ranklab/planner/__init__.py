"""Deterministic, validation-only experiment proposal planning."""

__all__ = ["build_proposal", "propose", "validate_proposal"]


def build_proposal(*args, **kwargs):
    """Lazily expose the builder without interfering with ``python -m``."""
    from .propose import build_proposal as _build_proposal

    return _build_proposal(*args, **kwargs)


def propose(*args, **kwargs):
    """Lazily expose the writer without interfering with ``python -m``."""
    from .propose import propose as _propose

    return _propose(*args, **kwargs)


def validate_proposal(*args, **kwargs):
    """Lazily expose the validator while retaining a small public API."""
    from .validator import validate_proposal as _validate_proposal

    return _validate_proposal(*args, **kwargs)
