"""metric_usage_validator — v1.1 architectural scaffold.

Design intent: confirms that when the agent computes a named metric in its
answer, the computation matches the stored ``metric_definition`` for that
name. E.g. if ``metric_definition[revenue]`` says ``sum(price * quantity)
where status = 'paid'`` and the agent's code does ``sum(price * quantity)``
without the status filter, the validator flags the mismatch.

v1.1 trigger: when the eval surfaces the LLM using a metric name without
honoring the stored formula — the most consequential failure mode for the
metric-definition category, since it silently produces wrong numbers.

Status: STUB — interface only, no behavior. See docs/architecture.md.
"""
from __future__ import annotations

from typing import Any


class MetricUsageValidator:
    """Confirms metric computations in agent output match stored definitions."""

    def validate(self, artifacts: Any) -> list[str]:
        """Return findings where a named metric's usage diverges from its stored definition."""
        raise NotImplementedError(
            "metric_usage_validator is a v1.1 scaffold; not yet implemented."
        )


__all__ = ["MetricUsageValidator"]
