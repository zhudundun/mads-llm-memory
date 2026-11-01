"""result_validator — v1.1 architectural scaffold.

Design intent: post-agent output sanity checks. Runs after the wrapper
returns a :class:`TurnArtifacts`; validates the agent's final answer for
gross failures the agent itself wouldn't catch — e.g. the loader claims
to have loaded a file but no df was produced; the viz agent emitted code
that imports a forbidden module; the EDA agent's answer references a column
that doesn't exist in the bound df.

v1.1 trigger: when the with-memory eval surfaces categories of answer-level
errors that the per-store tool-arg validation can't catch (because they're
about the answer text, not the tool call).

Status: STUB — interface only, no behavior. See docs/architecture.md.
"""
from __future__ import annotations

from typing import Any


class ResultValidator:
    """Validates an agent's final answer for gross sanity failures."""

    def validate(self, artifacts: Any) -> list[str]:
        """Return a list of human-readable validation findings (empty = ok).

        Inputs: the :class:`TurnArtifacts` returned by the wrapper. Output:
        zero or more findings; non-empty implies the caller should surface
        them to the user or trigger a retry.
        """
        raise NotImplementedError(
            "result_validator is a v1.1 scaffold; not yet implemented."
        )


__all__ = ["ResultValidator"]
