"""context_builder — v1.1 architectural scaffold.

Design intent: assembles agent prompts from memory recalls. After the router
picks an agent and BEFORE the wrapper invokes that agent, the context builder
queries the memory layer for the top-k relevant recalls — applicable user
preferences, prior dataset/schema knowledge for the bound df, matching metric
definitions for any metric tokens in the question, similar prior analysis
episodes — and weaves them into the prompt as compact, citation-tagged
context blocks. This is the read-side mirror of the wrapper's write-side
nudge: rather than asking the LLM to remember, it pre-loads what we already
know so the LLM doesn't have to ask.

v1.1 trigger: after the with-memory eval shows the LLM consistently fails to
*read* its own memory tools (recall_dataset, search_metric_definitions, ...)
even when the relevant facts exist. Observed signal: write tools fire but
read tools don't, so the model never benefits from prior turns.

Status: STUB — interface only, no behavior. See docs/architecture.md.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mads.manager import MemoryManager


class ContextBuilder:
    """Assembles agent prompts from memory recalls."""

    def build(
        self,
        question: str,
        *,
        manager: "MemoryManager",
        agent_name: str,
    ) -> str:
        """Return ``question`` augmented with relevant memory recalls.

        Inputs: the raw user question, the manager (for tenancy-scoped store
        access), and the routed agent name (loader/eda/viz) so the builder
        can apply per-agent recall budgets.

        Output: a single prompt string with prepended context blocks tagged
        for the LLM to cite when it answers (e.g. ``[memory:metric:revenue]``).
        """
        raise NotImplementedError(
            "context_builder is a v1.1 scaffold; not yet implemented."
        )


__all__ = ["ContextBuilder"]
