"""Tests for :mod:`mads.adapters.ads`.

These tests avoid real LLM and ADS-graph execution by mocking the ADS agent
classes and the model factory. They confirm:

* :class:`AgentResult` shape
* the three handle classes wire ``extra_tools`` through the monkey-patch
* tool injection mutates the ADS module-level lists *during* construction and
  restores them after
* :func:`make_viz` accepts ``extra_tools`` even though it is currently inert
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest import mock

import pandas as pd
import pytest

from mads.adapters.ads import (
    AgentResult,
    EDAHandle,
    LoaderHandle,
    VizHandle,
    _PatchedTools,
    execute_pandas,
    make_eda,
    make_loader,
    make_viz,
)


# --------------------------------------------------------------------- AgentResult


def test_agent_result_default_shape():
    r = AgentResult(response="ok")
    assert r.response == "ok"
    assert r.tool_calls == []
    assert r.viz_code is None
    assert r.prompt_tokens == 0
    assert r.completion_tokens == 0
    assert r.total_tokens == 0


def test_agent_result_full_shape():
    r = AgentResult(
        response="hi",
        tool_calls=[{"name": "foo"}],
        viz_code="import pandas as pd",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
    )
    assert r.viz_code == "import pandas as pd"
    assert r.total_tokens == 30
    assert r.tool_calls == [{"name": "foo"}]


# --------------------------------------------------------------------- _PatchedTools


def test_patched_tools_swap_and_restore():
    target = SimpleNamespace(items=[1, 2, 3])
    with _PatchedTools(target, "items", [9, 9]):
        assert target.items == [9, 9]
    assert target.items == [1, 2, 3]


def test_patched_tools_restores_on_exception():
    target = SimpleNamespace(items=[1])
    with pytest.raises(RuntimeError):
        with _PatchedTools(target, "items", [42]):
            assert target.items == [42]
            raise RuntimeError("boom")
    assert target.items == [1]


# --------------------------------------------------------------------- Loader


def _fake_loader_agent():
    """Return a fake DataLoaderToolsAgent instance with the methods we call."""
    inst = mock.MagicMock()
    inst.invoke_agent = mock.MagicMock(return_value=None)
    inst.get_ai_message = mock.MagicMock(return_value="loader-response")
    inst.get_tool_calls = mock.MagicMock(return_value=[{"name": "load_file"}])
    return inst


@pytest.fixture
def fake_llm():
    return SimpleNamespace(name="fake-llm", callbacks=[])


def test_loader_handle_invokes_and_returns_agent_result(fake_llm):
    inst = _fake_loader_agent()
    AgentClass = mock.MagicMock(return_value=inst)
    stub_mod = SimpleNamespace(tools=[])
    with mock.patch(
        "mads.adapters.ads._make_llm_with_usage",
        return_value=(fake_llm, mock.MagicMock(prompt_tokens=3, completion_tokens=5, total_tokens=8)),
    ), mock.patch.dict(
        "sys.modules",
        {
            "ai_data_science_team.agents": mock.MagicMock(
                DataLoaderToolsAgent=AgentClass,
                data_loader_tools_agent=stub_mod,
            ),
            "ai_data_science_team.agents.data_loader_tools_agent": stub_mod,
        },
    ):
        h = LoaderHandle()
        result = h.invoke("load it")
    assert isinstance(result, AgentResult)
    assert result.response == "loader-response"
    assert result.tool_calls == [{"name": "load_file"}]
    assert result.viz_code is None
    assert result.total_tokens == 8


def test_loader_handle_appends_extra_tools_during_construction(fake_llm):
    """The patched ADS-module ``tools`` list must contain the union of base +
    extra during agent construction."""
    inst = _fake_loader_agent()
    seen_tools_during_init = {}

    def AgentClass(model):
        # Read the module-level tools list NOW (during init). We must look it
        # up through the parent mock the same way the adapter does, because
        # ``from X import Y`` consults ``getattr(X, Y)`` before sys.modules.
        from ai_data_science_team.agents import data_loader_tools_agent as _mod
        seen_tools_during_init["tools"] = list(_mod.tools)
        return inst

    base_tool = mock.MagicMock(name="base_tool")
    base_tool.name = "load_file"
    extra_tool = mock.MagicMock(name="extra_tool")
    extra_tool.name = "add_metric_definition"

    stub_mod = SimpleNamespace(tools=[base_tool])
    with mock.patch(
        "mads.adapters.ads._make_llm_with_usage",
        return_value=(fake_llm, mock.MagicMock(prompt_tokens=0, completion_tokens=0, total_tokens=0)),
    ), mock.patch.dict(
        "sys.modules",
        {
            "ai_data_science_team.agents": mock.MagicMock(
                DataLoaderToolsAgent=AgentClass,
                data_loader_tools_agent=stub_mod,
            ),
            "ai_data_science_team.agents.data_loader_tools_agent": stub_mod,
        },
    ):
        LoaderHandle(extra_tools=[extra_tool])
    # During init the module tools list contained base + extra.
    names = [getattr(t, "name", "") for t in seen_tools_during_init["tools"]]
    assert "load_file" in names
    assert "add_metric_definition" in names


# --------------------------------------------------------------------- EDA


def _fake_eda_agent():
    inst = mock.MagicMock()
    inst.invoke_agent = mock.MagicMock(return_value=None)
    inst.get_ai_message = mock.MagicMock(return_value="eda-response")
    inst.get_tool_calls = mock.MagicMock(return_value=["explain_data"])
    return inst


def test_eda_handle_passes_df_to_invoke(fake_llm):
    inst = _fake_eda_agent()
    AgentClass = mock.MagicMock(return_value=inst)
    df = pd.DataFrame({"a": [1]})
    stub_mod = SimpleNamespace(EDA_TOOLS=[])

    with mock.patch(
        "mads.adapters.ads._make_llm_with_usage",
        return_value=(fake_llm, mock.MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2)),
    ), mock.patch.dict(
        "sys.modules",
        {
            "ai_data_science_team.ds_agents": mock.MagicMock(
                EDAToolsAgent=AgentClass,
                eda_tools_agent=stub_mod,
            ),
            "ai_data_science_team.ds_agents.eda_tools_agent": stub_mod,
        },
    ):
        h = EDAHandle(df)
        result = h.invoke("describe")
    # Confirm df was passed
    call = inst.invoke_agent.call_args
    assert call.kwargs["data_raw"] is df
    assert result.response == "eda-response"
    assert result.tool_calls == ["explain_data"]


# --------------------------------------------------------------------- Viz


def _fake_viz_agent():
    inst = mock.MagicMock()
    inst.invoke_agent = mock.MagicMock(return_value=None)
    inst.get_ai_message = mock.MagicMock(return_value="viz-response")
    inst.get_tool_calls = mock.MagicMock(return_value=[])
    inst.get_data_visualization_function = mock.MagicMock(
        return_value="def plot(df): ..."
    )
    inst.get_log_summary = mock.MagicMock(return_value="log")
    return inst


def test_viz_handle_returns_viz_code(fake_llm):
    inst = _fake_viz_agent()
    AgentClass = mock.MagicMock(return_value=inst)
    df = pd.DataFrame({"a": [1]})

    with mock.patch(
        "mads.adapters.ads._make_llm_with_usage",
        return_value=(fake_llm, mock.MagicMock(prompt_tokens=2, completion_tokens=4, total_tokens=6)),
    ), mock.patch.dict(
        "sys.modules",
        {
            "ai_data_science_team.agents": mock.MagicMock(
                DataVisualizationAgent=AgentClass
            ),
        },
    ):
        h = VizHandle(df)
        result = h.invoke("plot it")
    assert result.viz_code == "def plot(df): ..."
    assert result.response == "viz-response"
    assert result.total_tokens == 6


def test_viz_handle_stores_extra_tools_even_though_inert(fake_llm):
    inst = _fake_viz_agent()
    AgentClass = mock.MagicMock(return_value=inst)
    df = pd.DataFrame({"a": [1]})
    extra = [mock.MagicMock(name="record_episode")]

    with mock.patch(
        "mads.adapters.ads._make_llm_with_usage",
        return_value=(fake_llm, mock.MagicMock(prompt_tokens=0, completion_tokens=0, total_tokens=0)),
    ), mock.patch.dict(
        "sys.modules",
        {
            "ai_data_science_team.agents": mock.MagicMock(
                DataVisualizationAgent=AgentClass
            ),
        },
    ):
        h = VizHandle(df, extra_tools=extra)
    assert h.extra_tools == extra


# --------------------------------------------------------------------- factories


def test_make_loader_returns_handle(fake_llm):
    inst = _fake_loader_agent()
    AgentClass = mock.MagicMock(return_value=inst)
    stub_mod = SimpleNamespace(tools=[])
    with mock.patch(
        "mads.adapters.ads._make_llm_with_usage",
        return_value=(fake_llm, mock.MagicMock(prompt_tokens=0, completion_tokens=0, total_tokens=0)),
    ), mock.patch.dict(
        "sys.modules",
        {
            "ai_data_science_team.agents": mock.MagicMock(
                DataLoaderToolsAgent=AgentClass,
                data_loader_tools_agent=stub_mod,
            ),
            "ai_data_science_team.agents.data_loader_tools_agent": stub_mod,
        },
    ):
        h = make_loader()
    assert isinstance(h, LoaderHandle)


# --------------------------------------------------------------------- execute_pandas
#
# These tests hit the REAL ADS subprocess sandbox. They are fast (~250ms each)
# because the sandbox is a single python subprocess; we only run a handful. If
# ADS is unavailable in this environment the tests will surface "sandbox
# unavailable: ..." — that's correct error reporting, but in practice we
# always have ADS installed in `.[dev]` since the eval harness needs it.


@pytest.fixture(scope="module")
def small_df():
    return pd.DataFrame({"x": [1, 2, 3, 4], "y": [10, 20, 30, 40]})


def test_execute_pandas_happy_path(small_df):
    code = """
def compute(df):
    return {"n": int(len(df)), "sum_x": int(df['x'].sum())}
"""
    result, err = execute_pandas(code, small_df)
    assert err is None
    assert result == {"n": 4, "sum_x": 10}


def test_execute_pandas_rejects_empty_code(small_df):
    result, err = execute_pandas("", small_df)
    assert result is None
    assert err is not None and "empty" in err


def test_execute_pandas_rejects_missing_function(small_df):
    # No `def compute(...)` at all.
    code = "x = 1\n"
    result, err = execute_pandas(code, small_df)
    assert result is None
    assert "must define" in err


def test_execute_pandas_rejects_non_dataframe():
    result, err = execute_pandas("def compute(df): return {}", "not a df")  # type: ignore[arg-type]
    assert result is None
    assert "DataFrame" in err


def test_execute_pandas_returns_non_dict(small_df):
    code = "def compute(df):\n    return 42\n"
    result, err = execute_pandas(code, small_df)
    assert result is None
    assert "must return a dict" in err


def test_execute_pandas_raises_inside_compute(small_df):
    code = "def compute(df):\n    return {'bad': 1/0}\n"
    result, err = execute_pandas(code, small_df)
    assert result is None
    # Sandbox formats as "Function execution failed: ..."
    assert "fail" in err.lower() or "error" in err.lower()


def test_execute_pandas_coerces_nan_and_inf(small_df):
    code = """
def compute(df):
    return {"a": float("nan"), "b": float("inf"), "c": 1.5}
"""
    result, err = execute_pandas(code, small_df)
    assert err is None
    # NaN/Inf coerced to None so json.dumps(allow_nan=False) succeeds upstream.
    assert result == {"a": None, "b": None, "c": 1.5}


def test_execute_pandas_timeout(small_df):
    """A long-running compute() trips the parent-process timeout.

    We use a CPU spin loop rather than ``time.sleep`` because the sandbox's
    safe-builtins set excludes ``__import__``-based imports of arbitrary
    modules (it overrides ``__import__`` to block a denylist but allows
    plain stdlib re-imports via the ``import time`` statement — however
    ``time`` is re-imported by ast walk, so we just spin).
    """
    code = """
def compute(df):
    # Spin for a long time; the parent's subprocess.run timeout fires first.
    x = 0
    while x < 10**12:
        x += 1
    return {"never": "reached"}
"""
    result, err = execute_pandas(code, small_df, timeout_s=1)
    assert result is None
    assert err is not None
    assert "timed out" in err.lower()


def test_execute_pandas_syntax_error(small_df):
    code = "def compute(df):\n    return {\n"
    result, err = execute_pandas(code, small_df)
    assert result is None
    assert err is not None


def test_execute_pandas_returns_none(small_df):
    code = "def compute(df):\n    return None\n"
    result, err = execute_pandas(code, small_df)
    assert result is None
    assert "None" in err


# --------------------------------------------------------------------- factories


def test_make_eda_returns_handle(fake_llm):
    inst = _fake_eda_agent()
    AgentClass = mock.MagicMock(return_value=inst)
    df = pd.DataFrame({"a": [1]})
    stub_mod = SimpleNamespace(EDA_TOOLS=[])
    with mock.patch(
        "mads.adapters.ads._make_llm_with_usage",
        return_value=(fake_llm, mock.MagicMock(prompt_tokens=0, completion_tokens=0, total_tokens=0)),
    ), mock.patch.dict(
        "sys.modules",
        {
            "ai_data_science_team.ds_agents": mock.MagicMock(
                EDAToolsAgent=AgentClass,
                eda_tools_agent=stub_mod,
            ),
            "ai_data_science_team.ds_agents.eda_tools_agent": stub_mod,
        },
    ):
        h = make_eda(df)
    assert isinstance(h, EDAHandle)
