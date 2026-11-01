"""Adapter over ``ai_data_science_team`` (ADS) sub-agents.

This module is the **only** place outside ``eval/`` allowed to
``import ai_data_science_team``. Every other ``mads`` module talks to ADS
through :class:`AgentResult` and the three handle classes.

Tool-injection mechanism
------------------------

ADS sub-agents do not expose a ``tools=`` constructor kwarg (as of the version
pinned at install time). They build their react agent from module-level tool
lists baked into the package:

* ``ai_data_science_team.agents.data_loader_tools_agent.tools``
* ``ai_data_science_team.ds_agents.eda_tools_agent.EDA_TOOLS``

The adapter therefore monkey-patches those module-level lists during
:meth:`LoaderHandle.__init__` / :meth:`EDAHandle.__init__`, then restores
their original contents in a ``try/finally``. This patch is local to the
adapter; callers see only the handle objects.

:class:`DataVisualizationAgent` does **not** consume tools — it is a
code-generation pipeline, not a react tool-calling agent. ``extra_tools``
passed to :func:`make_viz` is therefore stored on the handle for inspection
but not wired into the underlying agent. This is documented as a known v1
limitation; the visualization agent cannot record/recall memory via tool
calls.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import BaseTool


# -----------------------------------------------------------------------------
# Default model
# -----------------------------------------------------------------------------
ANTHROPIC_MODEL = "claude-sonnet-4-6"


# -----------------------------------------------------------------------------
# AgentResult — uniform return shape
# -----------------------------------------------------------------------------
@dataclass
class AgentResult:
    """Uniform return for every ADS sub-agent invocation."""

    response: Optional[str]
    tool_calls: List[Any] = field(default_factory=list)
    viz_code: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# -----------------------------------------------------------------------------
# Token-usage callback
# -----------------------------------------------------------------------------
class _UsageCallback(BaseCallbackHandler):
    """Accumulates token usage across every LLM call during one session."""

    def __init__(self) -> None:
        super().__init__()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def on_llm_end(self, response, **kwargs) -> None:  # noqa: D401
        for gen_list in (getattr(response, "generations", None) or []):
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                um = (
                    getattr(msg, "usage_metadata", None)
                    if msg is not None
                    else None
                )
                if um:
                    self.prompt_tokens += int(um.get("input_tokens", 0) or 0)
                    self.completion_tokens += int(
                        um.get("output_tokens", 0) or 0
                    )
                    self.total_tokens += int(um.get("total_tokens", 0) or 0)
                    continue
                gi = getattr(gen, "generation_info", None) or {}
                tu = gi.get("token_usage") or gi.get("usage") or {}
                if tu:
                    self.prompt_tokens += int(
                        tu.get("input_tokens", tu.get("prompt_tokens", 0)) or 0
                    )
                    self.completion_tokens += int(
                        tu.get(
                            "output_tokens", tu.get("completion_tokens", 0)
                        )
                        or 0
                    )
                    self.total_tokens += int(tu.get("total_tokens", 0) or 0)


def _make_llm_with_usage(model: Optional[Any] = None):
    """Build a fresh ``ChatAnthropic`` (unless ``model`` already supplied) plus
    a per-session :class:`_UsageCallback`.

    Returns ``(llm, callback)``. The callback is also stashed on the agent by
    the calling handle.
    """
    cb = _UsageCallback()
    if model is not None:
        # Caller supplied a fully-formed LLM; attach the callback if supported.
        # ChatAnthropic accepts a ``callbacks`` list; we register via the
        # langchain callback manager pattern. We don't refuse weird models —
        # users may pass a fake for tests.
        try:
            existing = list(getattr(model, "callbacks", None) or [])
            existing.append(cb)
            try:
                model.callbacks = existing
            except Exception:
                pass
        except Exception:
            pass
        return model, cb

    from langchain_anthropic import ChatAnthropic

    llm = ChatAnthropic(
        model=ANTHROPIC_MODEL, temperature=0, callbacks=[cb]
    )
    return llm, cb


# -----------------------------------------------------------------------------
# Helpers (coercion lifted from the prior eval/agent_factories module)
# -----------------------------------------------------------------------------
def _coerce_ai_message(agent: Any) -> Optional[str]:
    try:
        msg = agent.get_ai_message(markdown=False)
    except Exception:
        try:
            msg = agent.get_ai_message()
        except Exception:
            return None
    if msg is None:
        return None
    return msg if isinstance(msg, str) else str(msg)


def _coerce_tool_calls(agent: Any) -> Any:
    try:
        return agent.get_tool_calls()
    except Exception:
        return []


def _read_usage(cb: _UsageCallback) -> Dict[str, int]:
    return {
        "prompt_tokens": int(getattr(cb, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(cb, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(cb, "total_tokens", 0) or 0),
    }


# -----------------------------------------------------------------------------
# Tool injection: monkey-patch the ADS module-level tool lists.
# -----------------------------------------------------------------------------
class _PatchedTools:
    """Context manager that swaps a module attribute (a list) and restores it.

    Used to inject memory tools into ADS's module-level tool lists during
    react-agent construction, then put the original list back.
    """

    def __init__(self, module, attr: str, replacement: list) -> None:
        self.module = module
        self.attr = attr
        self.replacement = list(replacement)
        self._original: Optional[list] = None

    def __enter__(self):
        self._original = list(getattr(self.module, self.attr))
        setattr(self.module, self.attr, self.replacement)
        return self

    def __exit__(self, *exc):
        if self._original is not None:
            setattr(self.module, self.attr, self._original)
        return False


# -----------------------------------------------------------------------------
# Handles
# -----------------------------------------------------------------------------
class LoaderHandle:
    """Thin wrapper around :class:`DataLoaderToolsAgent`.

    ``extra_tools`` is appended to the loader's module-level tool list during
    construction (see module docstring for the mechanism). Calling
    :meth:`invoke(prompt)` returns an :class:`AgentResult`.
    """

    def __init__(
        self,
        *,
        extra_tools: Optional[List[BaseTool]] = None,
        model: Optional[Any] = None,
    ) -> None:
        self._extra_tools = list(extra_tools or [])
        from ai_data_science_team.agents import DataLoaderToolsAgent
        from ai_data_science_team.agents import data_loader_tools_agent as _mod

        llm, cb = _make_llm_with_usage(model)
        self._cb = cb

        base = list(_mod.tools)
        with _PatchedTools(_mod, "tools", base + self._extra_tools):
            agent = DataLoaderToolsAgent(model=llm)
        self._agent = agent

    @property
    def extra_tools(self) -> List[BaseTool]:
        return list(self._extra_tools)

    def invoke(self, prompt: str) -> AgentResult:
        self._agent.invoke_agent(user_instructions=prompt)
        return AgentResult(
            response=_coerce_ai_message(self._agent),
            tool_calls=_coerce_tool_calls(self._agent),
            viz_code=None,
            **_read_usage(self._cb),
        )


class EDAHandle:
    """Thin wrapper around :class:`EDAToolsAgent`."""

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        extra_tools: Optional[List[BaseTool]] = None,
        model: Optional[Any] = None,
    ) -> None:
        self._df = df
        self._extra_tools = list(extra_tools or [])
        from ai_data_science_team.ds_agents import EDAToolsAgent
        from ai_data_science_team.ds_agents import eda_tools_agent as _mod

        llm, cb = _make_llm_with_usage(model)
        self._cb = cb

        base = list(_mod.EDA_TOOLS)
        with _PatchedTools(_mod, "EDA_TOOLS", base + self._extra_tools):
            agent = EDAToolsAgent(model=llm)
        self._agent = agent

    @property
    def extra_tools(self) -> List[BaseTool]:
        return list(self._extra_tools)

    def invoke(self, prompt: str) -> AgentResult:
        self._agent.invoke_agent(user_instructions=prompt, data_raw=self._df)
        return AgentResult(
            response=_coerce_ai_message(self._agent),
            tool_calls=_coerce_tool_calls(self._agent),
            viz_code=None,
            **_read_usage(self._cb),
        )


class VizHandle:
    """Thin wrapper around :class:`DataVisualizationAgent`.

    Note: ``DataVisualizationAgent`` is a code-generation pipeline that does
    not call tools, so ``extra_tools`` is stored on the handle for inspection
    but not wired into the underlying agent. This is a v1 limitation.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        extra_tools: Optional[List[BaseTool]] = None,
        model: Optional[Any] = None,
    ) -> None:
        self._df = df
        self._extra_tools = list(extra_tools or [])
        from ai_data_science_team.agents import DataVisualizationAgent

        llm, cb = _make_llm_with_usage(model)
        self._cb = cb
        self._agent = DataVisualizationAgent(
            model=llm,
            n_samples=30,
            log=False,
            bypass_recommended_steps=False,
            bypass_explain_code=False,
        )

    @property
    def extra_tools(self) -> List[BaseTool]:
        return list(self._extra_tools)

    def invoke(self, prompt: str) -> AgentResult:
        self._agent.invoke_agent(
            data_raw=self._df,
            user_instructions=prompt,
            max_retries=3,
        )
        try:
            viz_code = self._agent.get_data_visualization_function(
                markdown=False
            )
        except Exception:
            viz_code = None
        response = _coerce_ai_message(self._agent)
        if not response:
            try:
                response = self._agent.get_log_summary(markdown=False)
            except Exception:
                response = None
        return AgentResult(
            response=response,
            tool_calls=_coerce_tool_calls(self._agent),
            viz_code=viz_code,
            **_read_usage(self._cb),
        )


# -----------------------------------------------------------------------------
# Factory functions
# -----------------------------------------------------------------------------
def make_loader(
    *,
    extra_tools: Optional[List[BaseTool]] = None,
    model: Optional[Any] = None,
) -> LoaderHandle:
    """Construct a :class:`LoaderHandle` with memory tools optionally injected."""
    return LoaderHandle(extra_tools=extra_tools, model=model)


def make_eda(
    df: pd.DataFrame,
    *,
    extra_tools: Optional[List[BaseTool]] = None,
    model: Optional[Any] = None,
) -> EDAHandle:
    """Construct an :class:`EDAHandle` with memory tools optionally injected."""
    return EDAHandle(df, extra_tools=extra_tools, model=model)


def make_viz(
    df: pd.DataFrame,
    *,
    extra_tools: Optional[List[BaseTool]] = None,
    model: Optional[Any] = None,
) -> VizHandle:
    """Construct a :class:`VizHandle`. Note: ``extra_tools`` is currently
    inert for viz (the underlying ADS agent does not call tools).
    """
    return VizHandle(df, extra_tools=extra_tools, model=model)


# -----------------------------------------------------------------------------
# Pandas-code execution (subprocess sandbox)
# -----------------------------------------------------------------------------
# Cheap regex pre-check: confirms the code defines the requested function name
# at top-level before we pay subprocess overhead. We only require the literal
# `def NAME(` token to appear at line start (allowing leading whitespace). This
# is not a full parser — the sandbox runner re-parses the code with ast — but
# it catches the obvious "LLM forgot to define compute()" case fast.
_DEF_FUNC_RE_TEMPLATE = r"(?m)^\s*def\s+{name}\s*\("


def _sanitize_jsonable(obj: Any) -> Any:
    """Recursively coerce NaN/Inf to None (so ``allow_nan=False`` succeeds).

    Tuples are preserved as lists. Non-finite floats — ``nan``, ``+inf``,
    ``-inf`` — become ``None`` because the LLM consumer can't interpret JSON
    ``NaN`` (it's not strict-JSON). Other primitives pass through unchanged.
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_jsonable(v) for v in obj]
    return obj


def execute_pandas(
    code: str,
    df: pd.DataFrame,
    *,
    function_name: str = "compute",
    timeout_s: int = 30,
) -> Tuple[Optional[dict], Optional[str]]:
    """Run LLM-generated pandas code in the ADS subprocess sandbox.

    The sandbox runner (``ai_data_science_team.utils.sandbox``) expects a
    ``data`` payload it converts via ``pd.DataFrame.from_dict(data)``. We
    serialize ``df`` with ``to_dict()`` (column-oriented), which is the
    exact inverse of ``DataFrame.from_dict`` on the sandbox side — so the
    df the user's ``compute`` sees has the same columns and dtypes (modulo
    json-round-trippable types) as the bound df. Datetime columns survive
    via ISO-format strings; non-jsonable dtypes (e.g. complex) will raise.

    Parameters
    ----------
    code : str
        Python source that defines ``def {function_name}(df) -> dict``.
    df : pd.DataFrame
        The currently bound dataframe (passed by value into the sandbox).
    function_name : str, optional
        Name of the function the runner will invoke. Default ``"compute"``.
    timeout_s : int, optional
        Subprocess timeout in seconds. Default 30.

    Returns
    -------
    tuple
        ``(result_dict, None)`` on success; ``(None, error_message)`` on
        failure. Error messages are short, single-line, and safe to surface
        directly to the LLM ("ERROR: ..." framing is applied by the caller).
    """
    # Cheap pre-checks BEFORE paying subprocess startup cost.
    if not isinstance(code, str) or not code.strip():
        return None, "code is empty"
    if not isinstance(function_name, str) or not function_name.isidentifier():
        return None, f"invalid function_name: {function_name!r}"
    if not re.search(
        _DEF_FUNC_RE_TEMPLATE.format(name=re.escape(function_name)), code
    ):
        return None, (
            f"code must define a top-level function `def {function_name}(df) -> dict`"
        )
    if df is None or not isinstance(df, pd.DataFrame):
        return None, "df must be a pandas DataFrame"

    # Serialize the df. ``to_dict()`` defaults to column-oriented
    # ({col: {index: value}}) which round-trips through
    # ``pd.DataFrame.from_dict`` on the sandbox side back to the same shape.
    try:
        data = df.to_dict()
    except Exception as exc:  # noqa: BLE001
        return None, f"df serialization failed: {exc}"

    # Import the ADS sandbox lazily so test environments that mock ADS still
    # work. This is the ONE place in `mads/` allowed to import from ADS.
    try:
        from ai_data_science_team.utils.sandbox import (
            run_code_sandboxed_subprocess,
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"sandbox unavailable: {exc}"

    try:
        result, err = run_code_sandboxed_subprocess(
            code_snippet=code,
            function_name=function_name,
            data=data,
            timeout=timeout_s,
            data_format="dataframe",
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"sandbox raised: {exc}"

    if err is not None:
        # Normalize timeout phrasing so callers can pattern-match on "timed
        # out". The ADS sandbox uses "Sandbox timed out after Ns." — keep that
        # but pass it through unmodified; callers wrap with "ERROR: " framing.
        return None, str(err)

    # Sandbox returns the function's return value directly. We REQUIRE a dict.
    if result is None:
        return None, "compute() returned None; expected a dict"
    if not isinstance(result, dict):
        return None, (
            f"compute() must return a dict, got {type(result).__name__}"
        )

    # Sanitize NaN/Inf so the upstream ``json.dumps(..., allow_nan=False)``
    # doesn't fail. We coerce silently — the LLM gets ``null`` instead.
    sanitized = _sanitize_jsonable(result)
    if not isinstance(sanitized, dict):
        # Defensive: _sanitize_jsonable shouldn't change the top-level type
        # for a dict input, but guard anyway.
        return None, "sanitization yielded non-dict result"
    return sanitized, None


__all__ = [
    "ANTHROPIC_MODEL",
    "AgentResult",
    "LoaderHandle",
    "EDAHandle",
    "VizHandle",
    "execute_pandas",
    "make_loader",
    "make_eda",
    "make_viz",
]
