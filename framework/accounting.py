"""Provider-neutral token and cost accounting.

No model prices or limits are assumed.  A caller can configure them later;
the ledger still records usage now and exposes one gate for future LLM calls.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


class BudgetExceeded(RuntimeError):
    """Raised when a new model call would exceed a configured budget."""


@dataclass
class TokenUsage:
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    estimated_cost: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __post_init__(self) -> None:
        for name in ("input_tokens", "output_tokens", "cached_tokens"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.cached_tokens > self.input_tokens:
            raise ValueError("cached_tokens cannot exceed input_tokens")
        if self.estimated_cost < 0:
            raise ValueError("estimated_cost must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "total_tokens": self.total_tokens}


@dataclass(frozen=True)
class Budget:
    max_run_tokens: int | None = None
    max_total_tokens: int | None = None
    max_run_cost: float | None = None
    max_total_cost: float | None = None


class CostLedger:
    """Tracks totals and enforces optional per-run and cumulative limits."""

    def __init__(self, budget: Budget | None = None, *, total: TokenUsage | None = None):
        self.budget = budget or Budget()
        self.total = total or TokenUsage()
        self.run = TokenUsage()

    def can_start(self, estimate: TokenUsage | None = None) -> bool:
        estimate = estimate or TokenUsage()
        return not self._violations(estimate)

    def require_capacity(self, estimate: TokenUsage | None = None) -> None:
        violations = self._violations(estimate or TokenUsage())
        if violations:
            raise BudgetExceeded("; ".join(violations))

    def record(self, usage: TokenUsage) -> None:
        self.require_capacity(usage)
        self.run = self._add(self.run, usage)
        self.total = self._add(self.total, usage)

    def record_observed(self, usage: TokenUsage) -> None:
        """Record usage that already happened, even if it crossed a limit."""
        self.run = self._add(self.run, usage)
        self.total = self._add(self.total, usage)

    def _violations(self, usage: TokenUsage) -> list[str]:
        run_tokens = self.run.total_tokens + usage.total_tokens
        total_tokens = self.total.total_tokens + usage.total_tokens
        run_cost = self.run.estimated_cost + usage.estimated_cost
        total_cost = self.total.estimated_cost + usage.estimated_cost
        violations: list[str] = []
        if self.budget.max_run_tokens is not None and run_tokens > self.budget.max_run_tokens:
            violations.append("per-run token budget reached")
        if self.budget.max_total_tokens is not None and total_tokens > self.budget.max_total_tokens:
            violations.append("total token budget reached")
        if self.budget.max_run_cost is not None and run_cost > self.budget.max_run_cost:
            violations.append("per-run cost budget reached")
        if self.budget.max_total_cost is not None and total_cost > self.budget.max_total_cost:
            violations.append("total cost budget reached")
        return violations

    @staticmethod
    def _add(left: TokenUsage, right: TokenUsage) -> TokenUsage:
        return TokenUsage(
            model=right.model or left.model,
            input_tokens=left.input_tokens + right.input_tokens,
            output_tokens=left.output_tokens + right.output_tokens,
            cached_tokens=left.cached_tokens + right.cached_tokens,
            estimated_cost=left.estimated_cost + right.estimated_cost,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget": asdict(self.budget),
            "run": self.run.to_dict(),
            "total": self.total.to_dict(),
            "exhausted": not self.can_start(),
        }
