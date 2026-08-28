"""Bounded stopping decisions for validation-driven experiment loops."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StopDecision:
    stop: bool
    reason: str | None = None


class StoppingPolicy:
    def __init__(self, *, epsilon: float, patience: int, max_iterations: int, max_wallclock_seconds: float):
        if epsilon < 0 or patience < 1 or max_iterations < 1 or max_wallclock_seconds <= 0:
            raise ValueError("invalid stopping limits")
        self.epsilon = epsilon
        self.patience = patience
        self.max_iterations = max_iterations
        self.max_wallclock_seconds = max_wallclock_seconds
        self.best_primary: float | None = None
        self.stale_iterations = 0

    def before_iteration(self, iteration: int, elapsed_seconds: float) -> StopDecision:
        if iteration > self.max_iterations:
            return StopDecision(True, "maximum iterations reached")
        if elapsed_seconds >= self.max_wallclock_seconds:
            return StopDecision(True, "maximum wall-clock time reached")
        return StopDecision(False)

    def observe(self, primary: float, *, iteration: int, elapsed_seconds: float) -> StopDecision:
        if self.best_primary is None or primary - self.best_primary > self.epsilon:
            self.best_primary = primary if self.best_primary is None else max(self.best_primary, primary)
            self.stale_iterations = 0
        else:
            self.stale_iterations += 1
        if iteration >= self.max_iterations:
            return StopDecision(True, "maximum iterations reached")
        if elapsed_seconds >= self.max_wallclock_seconds:
            return StopDecision(True, "maximum wall-clock time reached")
        if self.stale_iterations >= self.patience:
            return StopDecision(True, "no meaningful validation improvement")
        return StopDecision(False)
