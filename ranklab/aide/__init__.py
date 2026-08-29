"""Guarded AIDE handoff for validation-only research hypotheses."""

__all__ = ["HypothesisValidationError", "build_aide_request", "validate_hypothesis"]


def __getattr__(name):
    if name in __all__:
        from . import adapter

        return getattr(adapter, name)
    raise AttributeError(name)
