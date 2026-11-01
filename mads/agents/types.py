"""Dataclasses shared between :mod:`mads.agents.wrapper` and
:mod:`mads.composer` so the wrapper-composer pair has no circular import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

from mads.adapters.ads import AgentResult


@dataclass
class TurnArtifacts:
    """One memory-aware analysis turn.

    Produced by a wrapped agent's ``invoke(prompt)``; consumed by
    :func:`mads.composer.compose` to format the final answer.

    The ``memories_injected`` / ``warnings`` fields are populated by the
    proactive :class:`mads.wrappers.memory_aware.MemoryAwareEDA` wrapper;
    legacy reactive wrappers (``WrappedLoader`` / ``WrappedEDA`` /
    ``WrappedViz``) leave them at the default empty list.
    """

    question: str
    agent_name: str  # "loader" | "eda" | "viz"
    agent_result: AgentResult
    memory_reads: List[dict] = field(default_factory=list)
    memory_writes: List[dict] = field(default_factory=list)
    memories_injected: List[Any] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


__all__ = ["AgentResult", "TurnArtifacts"]
