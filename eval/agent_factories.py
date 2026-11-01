"""ADS agent factories + a per-agent invocation dispatcher.

This module is the eval-side bridge between the harness and the ADS agents.
Per V1_DESIGN_BRIEF §3, eval no longer imports ai_data_science_team directly;
it routes through :mod:`mads.adapters.ads` so the adapter remains the single
exempt callsite for ADS imports.

Public API:
    make_loader_factory()                -> Callable[[], LoaderHandle]
    make_eda_factory(df)                 -> Callable[[], EDAHandle]
    make_viz_factory(df)                 -> Callable[[], VizHandle]
    invoke_for_agent(name, agent, prompt, df=None) -> dict

    invoke_dispatcher(name, df=None)     -> (prompt, agent) -> dict
        (curried form that plugs straight into harness.run_scenario_pair.)

Why factories (not just instances): the eval harness creates a fresh agent
per session. That's the entire point — we want the baseline-without-memory
contract to be the literal absence of shared state across A and B.

API contract reference: /Users/wang/LLM_memory/baseline/SETUP_REPORT.md §3.

LLM provider: Anthropic Claude Sonnet 4.6, configured inside the adapter.
Token usage is captured via :class:`mads.adapters.ads._UsageCallback`.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from mads.adapters.ads import (
    ANTHROPIC_MODEL,
    AgentResult,
    EDAHandle,
    LoaderHandle,
    VizHandle,
    make_eda,
    make_loader,
    make_viz,
)


# -----------------------------------------------------------------------------
# Factories
# -----------------------------------------------------------------------------
def make_loader_factory(
    *, extra_tools: Optional[list] = None
) -> Callable[[], LoaderHandle]:
    """Return a zero-arg callable that builds a fresh LoaderHandle."""

    def _factory() -> LoaderHandle:
        return make_loader(extra_tools=extra_tools)

    return _factory


def make_eda_factory(
    df, *, extra_tools: Optional[list] = None
) -> Callable[[], EDAHandle]:
    """Return a zero-arg callable that builds a fresh EDAHandle.

    The dataframe is captured in the closure; the handle passes it to ADS at
    invoke time.
    """

    def _factory() -> EDAHandle:
        return make_eda(df, extra_tools=extra_tools)

    return _factory


def make_viz_factory(
    df, *, extra_tools: Optional[list] = None
) -> Callable[[], VizHandle]:
    """Return a zero-arg callable that builds a fresh VizHandle."""

    def _factory() -> VizHandle:
        return make_viz(df, extra_tools=extra_tools)

    return _factory


# -----------------------------------------------------------------------------
# Per-agent invocation dispatcher
# -----------------------------------------------------------------------------
def _result_to_payload(r: AgentResult) -> Dict[str, Any]:
    return {
        "response": r.response,
        "tool_calls": r.tool_calls,
        "viz_code": r.viz_code,
        "prompt_tokens": r.prompt_tokens,
        "completion_tokens": r.completion_tokens,
        "total_tokens": r.total_tokens,
    }


def invoke_for_agent(
    name: str,
    agent: Any,
    prompt: str,
    df: Optional[Any] = None,
) -> Dict[str, Any]:
    """Single-shot invoke of one ADS agent (handle) and normalize its outputs.

    ``agent`` is one of :class:`LoaderHandle`, :class:`EDAHandle`,
    :class:`VizHandle`.

    Returns:
        {
            "response":  str | None,         # the AI's natural-language reply
            "tool_calls": Any,               # raw ADS tool-call records
            "viz_code":   str | None,        # viz agent's generated Python code
            "prompt_tokens": int,
            "completion_tokens": int,
            "total_tokens": int,
        }
    """
    if name not in ("loader", "eda", "viz"):
        raise ValueError(
            f"unknown agent name: {name!r} (want one of: loader, eda, viz)"
        )
    result = agent.invoke(prompt)
    return _result_to_payload(result)


def invoke_dispatcher(
    name: str, df: Optional[Any] = None
) -> Callable[[str, Any], Dict[str, Any]]:
    """Curry invoke_for_agent for `harness.run_scenario_pair(invoke_fn=...)`."""

    def _fn(prompt: str, agent: Any) -> Dict[str, Any]:
        return invoke_for_agent(name, agent, prompt, df=df)

    return _fn


__all__ = [
    "ANTHROPIC_MODEL",
    "make_loader_factory",
    "make_eda_factory",
    "make_viz_factory",
    "invoke_for_agent",
    "invoke_dispatcher",
]
