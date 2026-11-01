"""metric_extractor — v1.1 architectural scaffold.

Design intent: mines ``metric_definition`` candidates from analyst utterances
(the user's question text plus the agent's final answer). Looks for phrases
like "let's define revenue as ...", "active user means ...", "30-day
retention = ..." and proposes a metric_definition row for the user to
confirm. The point is to catch implicit definitions the LLM wouldn't have
thought to call ``add_metric_definition`` for.

v1.1 trigger: after observed gap-closure rate <30% in the with-memory eval —
i.e. once we have evidence the LLM consistently fails to write the metrics
it just used.

Status: STUB — interface only, no behavior. See docs/architecture.md.
"""
from __future__ import annotations

from typing import Any


class MetricExtractor:
    """Mines candidate metric_definition rows from analyst transcripts."""

    def extract(self, transcript: str) -> list[dict[str, Any]]:
        """Return a list of candidate metric_definition payloads.

        Inputs: the concatenated user question + agent answer + tool-call
        log for a single turn. Output: zero or more candidate payloads;
        each is intended to be presented to the user for confirmation
        before persisting (no silent writes).
        """
        raise NotImplementedError(
            "metric_extractor is a v1.1 scaffold; not yet implemented."
        )


__all__ = ["MetricExtractor"]
