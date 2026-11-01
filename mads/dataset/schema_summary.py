"""schema_summary — v1.1 architectural scaffold.

Design intent: render a human-readable, prompt-friendly schema summary from
the structured profile produced by :mod:`mads.dataset.profiler` plus any
matching ``dataset_memory`` / ``column_memory`` recalls. The output is a
markdown-style block intended to be prepended to agent prompts by
:mod:`mads.router.context_builder` so the LLM sees both ground-truth schema
facts AND the user's prior notes about each column ("``customer_id`` is the
PK", "``signup_date`` is in UTC", ...) on every turn.

v1.1 trigger: lands alongside ``context_builder`` (same observed gap — the
LLM doesn't read its own dataset memory tools, so the facts need to be
pushed into the prompt).

Status: STUB — interface only, no behavior. See docs/architecture.md.
"""
from __future__ import annotations

from typing import Any


class SchemaSummary:
    """Renders a markdown schema summary from a profile + memory recalls."""

    def render(
        self,
        profile: dict[str, Any],
        *,
        dataset_recall: dict[str, Any] | None = None,
        column_recalls: list[dict[str, Any]] | None = None,
    ) -> str:
        """Return a markdown summary string suitable for prompt injection.

        Inputs: the structured profile from :class:`DatasetProfiler`, plus
        optional ``dataset_memory`` and ``column_memory`` rows for the same
        fingerprint. Output: a compact markdown block (~200-400 tokens
        target) the context builder can splice into the agent prompt.
        """
        raise NotImplementedError(
            "schema_summary is a v1.1 scaffold; not yet implemented."
        )


__all__ = ["SchemaSummary"]
