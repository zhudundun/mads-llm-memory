"""Memory-aware wrappers around :mod:`mads.adapters.ads` handles.

A wrapper does three things over the bare adapter:

1. Injects the manager's memory ``@tool`` callables into the underlying ADS
   agent's tool list (see :mod:`mads.adapters.ads` for the patch mechanism).
2. Observes which memory tools were called during ``invoke(prompt)`` by
   inspecting ``result.tool_calls`` and the manager's known tool-name set.
3. Optionally appends a system-prompt nudge that asks the LLM to record
   relevant facts before answering. The nudge is ON by default and can be
   disabled via ``nudge_memory=False`` at wrap time.
"""

from __future__ import annotations

from typing import List, Optional, Set

import pandas as pd

from mads.adapters.ads import (
    AgentResult,
    EDAHandle,
    LoaderHandle,
    VizHandle,
    make_eda,
    make_loader,
    make_viz,
)
from mads.agents.types import TurnArtifacts
from mads.manager import MemoryManager


# The default nudge that, per V1_DESIGN_BRIEF §8 (option A2), is appended to
# the user prompt when ``nudge_memory=True``. It lives at the *prompt* layer
# rather than the system layer because ADS bakes its own system prompts into
# the agent graphs.
_DEFAULT_NUDGE = (
    "\n\n[Memory hint]: When the user states a domain fact, schema rule, "
    "metric definition, dataset preference, or analysis conclusion you should "
    "remember for next time, call the appropriate memory tool "
    "(register_dataset, register_column, add_metric_definition, "
    "record_analysis_episode, ...) BEFORE giving your final answer."
)


def _classify_tool_calls(
    tool_calls: list,
    *,
    memory_names: Set[str],
    write_names: Set[str],
    read_names: Set[str],
) -> tuple[list, list]:
    """Split ``tool_calls`` into ``(memory_reads, memory_writes)`` lists.

    A tool call is identified by name; if the name matches the manager's
    known memory tools and is also classified as write, it goes into
    ``memory_writes``; otherwise into ``memory_reads``.
    """
    reads: list = []
    writes: list = []
    for rec in tool_calls or []:
        name = _extract_name(rec)
        if not name or name not in memory_names:
            continue
        if name in write_names:
            writes.append(_normalize(rec, name))
        elif name in read_names:
            reads.append(_normalize(rec, name))
        else:
            # Conservatively log unknown classification under reads.
            reads.append(_normalize(rec, name))
    return reads, writes


def _extract_name(rec) -> Optional[str]:
    if isinstance(rec, str):
        return rec
    if isinstance(rec, dict):
        fn = rec.get("function")
        fn_name = fn.get("name") if isinstance(fn, dict) else None
        return (
            rec.get("name")
            or rec.get("tool")
            or rec.get("tool_name")
            or fn_name
        )
    return getattr(rec, "name", None) or getattr(rec, "tool", None)


def _normalize(rec, name: str) -> dict:
    """Coerce a tool-call record to ``{"name": ..., "args": ...}``."""
    if isinstance(rec, dict):
        args = rec.get("args") or rec.get("arguments") or {}
        return {"name": name, "args": args}
    return {"name": name, "args": {}}


# -----------------------------------------------------------------------------
# Memory-tool classification (write vs read)
# -----------------------------------------------------------------------------
# Authoritative source for tool names is the factory in mads/memory/. These
# constants mirror them. The tests confirm they are in sync with the actual
# factory output by counting against ``manager.all_tools(...)``.
_METRIC_WRITES = {"add_metric_definition"}
_METRIC_READS = {
    "get_metric_definition",
    "list_metric_definitions",
    "search_metric_definitions",
    "get_metric_history",
}
_DATASET_WRITES = {"register_dataset", "register_column"}
_DATASET_READS = {
    "recall_dataset",
    "list_datasets",
    "recall_column",
    "list_columns",
}
_EPISODE_WRITES = {"record_analysis_episode", "update_user_feedback"}
_EPISODE_READS = {
    "get_analysis_episode",
    "list_recent_analyses",
    "find_similar_analyses",
    "search_analyses",
}
# Preference tool names are authored in mads/memory/preference_tools.py.
# ``remember_preference`` and ``forget_preference`` mutate state →  writes;
# ``recall_preference`` and ``list_preferences`` are read-only.
_PREFERENCE_WRITES = {"remember_preference", "forget_preference"}
_PREFERENCE_READS = {"recall_preference", "list_preferences"}

_MEMORY_WRITE_NAMES = (
    _METRIC_WRITES | _PREFERENCE_WRITES | _DATASET_WRITES | _EPISODE_WRITES
)
_MEMORY_READ_NAMES = (
    _METRIC_READS | _PREFERENCE_READS | _DATASET_READS | _EPISODE_READS
)
_MEMORY_ALL_NAMES = _MEMORY_WRITE_NAMES | _MEMORY_READ_NAMES


# -----------------------------------------------------------------------------
# Wrapped agents
# -----------------------------------------------------------------------------
class _BaseWrapped:
    """Shared logic: nudge, invoke, classify tool calls."""

    agent_name: str = "?"

    def __init__(
        self,
        handle,
        *,
        manager: MemoryManager,
        nudge_memory: bool,
        memory_names: Set[str],
    ) -> None:
        self._handle = handle
        self._manager = manager
        self._nudge_memory = nudge_memory
        self._memory_names = memory_names

    def invoke(self, prompt: str) -> TurnArtifacts:
        effective = prompt
        if self._nudge_memory:
            effective = prompt + _DEFAULT_NUDGE
        result: AgentResult = self._handle.invoke(effective)
        reads, writes = _classify_tool_calls(
            list(result.tool_calls or []),
            memory_names=self._memory_names,
            write_names=_MEMORY_WRITE_NAMES,
            read_names=_MEMORY_READ_NAMES,
        )
        return TurnArtifacts(
            question=prompt,
            agent_name=self.agent_name,
            agent_result=result,
            memory_reads=reads,
            memory_writes=writes,
        )


class WrappedLoader(_BaseWrapped):
    agent_name = "loader"


class WrappedEDA(_BaseWrapped):
    agent_name = "eda"


class WrappedViz(_BaseWrapped):
    agent_name = "viz"


# -----------------------------------------------------------------------------
# Factory functions
# -----------------------------------------------------------------------------
def wrap_loader(
    manager: MemoryManager,
    *,
    nudge_memory: bool = True,
) -> WrappedLoader:
    """Build a :class:`WrappedLoader`.

    The loader's *first* turn has no df, so dataset tools are skipped;
    metric + preference + episode tools are injected (15 total: 5 + 4 + 6).
    Per v1 brief §4, the loader does one turn and is done — v1.1 will
    rewire it post-load.
    """
    extra_tools = manager.all_tools(include_dataset=False)
    handle = make_loader(extra_tools=extra_tools)
    memory_names = {t.name for t in extra_tools}
    return WrappedLoader(
        handle,
        manager=manager,
        nudge_memory=nudge_memory,
        memory_names=memory_names,
    )


def wrap_eda(
    manager: MemoryManager,
    df: pd.DataFrame,
    *,
    dataset_name: str,
    table_name: str,
    nudge_memory: bool = True,
) -> WrappedEDA:
    """Build a :class:`WrappedEDA`.

    Binds the df on the manager (so dataset tools fingerprint THIS df), then
    injects all 21 tools (5 metric + 4 preference + 6 dataset + 6 episode).
    """
    if df is None:
        raise ValueError("wrap_eda requires a non-None DataFrame")
    manager.bind_dataframe(df, dataset_name=dataset_name, table_name=table_name)
    extra_tools = manager.all_tools(include_dataset=True)
    handle = make_eda(df, extra_tools=extra_tools)
    memory_names = {t.name for t in extra_tools}
    return WrappedEDA(
        handle,
        manager=manager,
        nudge_memory=nudge_memory,
        memory_names=memory_names,
    )


def wrap_viz(
    manager: MemoryManager,
    df: pd.DataFrame,
    *,
    dataset_name: str,
    table_name: str,
    nudge_memory: bool = True,
) -> WrappedViz:
    """Build a :class:`WrappedViz`.

    Note: as of v1, the underlying ADS visualization agent does not call
    tools (it generates code directly), so injected memory tools are inert.
    The wrapper still binds the df and exposes a stable :class:`TurnArtifacts`
    return shape so callers don't need a viz-specific code path.
    """
    if df is None:
        raise ValueError("wrap_viz requires a non-None DataFrame")
    manager.bind_dataframe(df, dataset_name=dataset_name, table_name=table_name)
    extra_tools = manager.all_tools(include_dataset=True)
    handle = make_viz(df, extra_tools=extra_tools)
    memory_names = {t.name for t in extra_tools}
    return WrappedViz(
        handle,
        manager=manager,
        nudge_memory=nudge_memory,
        memory_names=memory_names,
    )


__all__ = [
    "WrappedLoader",
    "WrappedEDA",
    "WrappedViz",
    "wrap_loader",
    "wrap_eda",
    "wrap_viz",
]
