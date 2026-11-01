"""Proactive memory-injection wrapper for the ADS EDA agent (Plan B).

Where the reactive :class:`mads.agents.wrapper.WrappedEDA` only *nudges* the
LLM to call memory tools, :class:`MemoryAwareEDA` builds and injects a
retrieved-memory block into the prompt *before* the agent runs. The 8-step
flow (with auto-extraction explicitly out of scope) is:

1. Profile the df → ``DatasetContext`` with semantic fingerprint.
2. Retrieve top-k memories across metric/dataset/episode via the manager's
   hybrid retriever, filtered by the current dataset fingerprint.
3. Format retrieved memories as citation-tagged blocks
   (``[memory:CATEGORY:ID] (score=S.SS)``).
4. Build an enriched prompt: user question + dataset profile_summary +
   memory block + one-line instructions about citing/respecting memories.
5. Bind the df on the manager, build the ADS EDA handle with memory tools
   still injected (so the LLM CAN call them if it wants, even though it
   doesn't need to), invoke.
6. Run all three :mod:`mads.validation` validators (warn-only).
7. SKIPPED — no auto-extraction in this Plan B build.
8. Classify any tool calls the LLM made into reads/writes (reusing the
   classifier from the legacy wrapper) and return a
   :class:`TurnArtifacts`.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from mads.adapters.ads import AgentResult, make_eda
from mads.agents.types import TurnArtifacts
from mads.agents.wrapper import (
    _DEFAULT_NUDGE,
    _MEMORY_READ_NAMES,
    _MEMORY_WRITE_NAMES,
    _classify_tool_calls,
)
from mads.dataset.profiler import profile as _profile_df
from mads.manager import MemoryManager
from mads.memory.retrievers import RetrievedMemory


# Threshold below which retrieved memories are dropped before injection.
# Calibrated against the per-category retrievers' actual scoring:
#   * metric retriever: substring match floors at 0.5
#   * dataset retriever: semantic match is 0.7, exact is 1.0
#   * episode retriever: 1.0 - clamp(distance, 0, 2)/2.0 — so a HashEmbedder
#     cosine "miss" can easily score < 0.3
# 0.3 trims junk without throwing out borderline substring metric matches.
_MIN_SCORE = 0.3


_INSTRUCTIONS = (
    "When you reference a memory above, cite it inline as [memory:CATEGORY:ID]. "
    "Respect any known issues. State assumptions before computing metrics. "
    "If a memory gives you a filter rule, metric definition, or predicate, "
    "USE `run_pandas_code` to compute the result against the loaded dataframe. "
    "Do not estimate from prior knowledge. "
    "If a memory is marked `APPLICABLE: false`, do NOT silently substitute. "
    "Either explain the mismatch to the user and ask for clarification, or pick "
    "a clearly-flagged proxy and disclose what you did."
)


class MemoryAwareEDA:
    """Proactive memory-injection wrapper over :class:`EDAHandle`.

    Construction is cheap; the real work happens in :meth:`run`, which both
    profiles the df and builds the underlying ADS handle. ``nudge_memory``
    defaults to ``False`` because the injected memory block replaces the
    static nudge — keeping both on would double-prompt the LLM.
    """

    agent_name: str = "eda"

    def __init__(
        self,
        manager: MemoryManager,
        *,
        nudge_memory: bool = False,
    ) -> None:
        self._manager = manager
        self._nudge_memory = nudge_memory

    def run(
        self,
        user_input: str,
        df: pd.DataFrame,
        *,
        user_id: str,
        project_id: str,
        dataset_name: str,
        table_name: str,
        k: int = 5,
    ) -> TurnArtifacts:
        """Execute the 8-step flow and return a :class:`TurnArtifacts`."""
        if df is None:
            raise ValueError("MemoryAwareEDA.run requires a non-None DataFrame")

        # Step 1: profile.
        ctx = _profile_df(df, dataset_name=dataset_name)

        # Step 2: retrieve. We retrieve *before* binding the df so the
        # dataset sub-retriever sees ctx.semantic_fingerprint via the bound
        # df (we bind below for tool wiring); thread the semantic
        # fingerprint as the episode-side filter so we only see episodes
        # that referenced the same logical dataset.
        self._manager.bind_dataframe(
            df, dataset_name=dataset_name, table_name=table_name
        )
        retrieved = self._manager.retrieve(
            query=user_input,
            user_id=user_id,
            project_id=project_id,
            dataset_fingerprint=ctx.semantic_fingerprint,
            k=k,
        )
        memories: List[RetrievedMemory] = [
            rm for rm in retrieved if rm.score >= _MIN_SCORE
        ]

        # Step 3: format.
        memory_block = self._manager.format_context(memories)

        # Step 4: build the enriched prompt.
        enriched_prompt = self._build_prompt(
            user_input=user_input,
            profile_summary=ctx.profile_summary,
            memory_block=memory_block,
        )
        if self._nudge_memory:
            enriched_prompt = enriched_prompt + _DEFAULT_NUDGE

        # Step 5: build the ADS EDA handle (memory tools still injected so
        # the LLM can fetch more if it wants) and invoke.
        extra_tools = self._manager.all_tools(include_dataset=True)
        memory_names = {t.name for t in extra_tools}
        handle = make_eda(df, extra_tools=extra_tools)
        result: AgentResult = handle.invoke(enriched_prompt)

        # Step 6: validate (warn-only; never blocks).
        warnings = self._manager.validate(
            result, memories_used=memories
        )

        # Step 7: SKIPPED — no auto-extraction in Plan B.

        # Step 8: classify tool calls (LLM may still have called memory tools).
        reads, writes = _classify_tool_calls(
            list(result.tool_calls or []),
            memory_names=memory_names,
            write_names=_MEMORY_WRITE_NAMES,
            read_names=_MEMORY_READ_NAMES,
        )

        return TurnArtifacts(
            question=user_input,
            agent_name=self.agent_name,
            agent_result=result,
            memory_reads=reads,
            memory_writes=writes,
            memories_injected=list(memories),
            warnings=warnings,
        )

    # Convenience for unit tests / callers that want to introspect the prompt
    # without invoking the ADS handle.
    @staticmethod
    def _build_prompt(
        *,
        user_input: str,
        profile_summary: str,
        memory_block: str,
    ) -> str:
        return (
            f"{user_input}\n\n"
            f"## Dataset profile\n{profile_summary}\n\n"
            f"## Relevant memories\n{memory_block}\n"
            f"## Instructions\n{_INSTRUCTIONS}\n"
        )


__all__ = ["MemoryAwareEDA"]
