"""Tests for :mod:`mads.agents.wrapper`.

The wrapper layer is mostly orchestration; these tests verify:

* The right number of memory tools are injected for each agent (loader: 15,
  eda: 22, viz: 22) — counts include the 4 preference tools wired in via
  ``MemoryManager.all_tools`` PLUS the 1 ``run_pandas_code`` execution tool
  emitted when ``include_dataset=True``.
* The nudge is appended when ``nudge_memory=True`` and not when False.
* ``invoke`` returns a :class:`TurnArtifacts` and classifies tool calls
  into reads/writes based on the manager's known tool-name set.
* ``wrap_eda(None, ...)`` and ``wrap_viz(None, ...)`` raise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

import pandas as pd
import pytest

from mads.adapters.ads import AgentResult
from mads.agents.types import TurnArtifacts
from mads.agents.wrapper import (
    _DEFAULT_NUDGE,
    WrappedEDA,
    WrappedLoader,
    WrappedViz,
    wrap_eda,
    wrap_loader,
    wrap_viz,
)
from mads.manager import MemoryManager
from mads.memory import HashEmbedder


# --------------------------------------------------------------------- fixtures


@pytest.fixture
def manager(tmp_path) -> MemoryManager:
    db = str(tmp_path / "w.sqlite")
    m = MemoryManager(
        user_id="u", project_id="p", db_path=db, embedder=HashEmbedder()
    )
    yield m
    m.close()


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2], "b": [3, 4]})


# --------------------------------------------------------------------- fake handles


class _FakeHandle:
    """Captures invoke()'s prompt and returns a canned AgentResult."""

    def __init__(self, response: str = "ok", tool_calls=None):
        self._response = response
        self._tool_calls = tool_calls or []
        self.last_prompt: str | None = None
        self.extra_tools: List = []
        self.invoke_count = 0

    def invoke(self, prompt: str) -> AgentResult:
        self.last_prompt = prompt
        self.invoke_count += 1
        return AgentResult(
            response=self._response,
            tool_calls=list(self._tool_calls),
            prompt_tokens=1,
            completion_tokens=2,
            total_tokens=3,
        )


# --------------------------------------------------------------------- loader


def test_wrap_loader_injects_15_tools(manager, monkeypatch):
    captured = {}

    def fake_make_loader(*, extra_tools=None, model=None):
        captured["extra_tools"] = list(extra_tools or [])
        return _FakeHandle()

    monkeypatch.setattr(
        "mads.agents.wrapper.make_loader", fake_make_loader
    )
    w = wrap_loader(manager)
    assert isinstance(w, WrappedLoader)
    # 5 metric + 4 preference + 6 episode (no dataset, no df yet) = 15.
    assert len(captured["extra_tools"]) == 15
    names = {t.name for t in captured["extra_tools"]}
    assert "add_metric_definition" in names
    assert "record_analysis_episode" in names
    assert "remember_preference" in names
    assert "recall_preference" in names
    # No dataset tools.
    assert "register_dataset" not in names


def test_wrap_loader_nudge_on_by_default(manager, monkeypatch):
    fake = _FakeHandle()
    monkeypatch.setattr(
        "mads.agents.wrapper.make_loader",
        lambda *, extra_tools=None, model=None: fake,
    )
    w = wrap_loader(manager)
    w.invoke("load data")
    assert fake.last_prompt.startswith("load data")
    assert _DEFAULT_NUDGE in fake.last_prompt


def test_wrap_loader_nudge_off(manager, monkeypatch):
    fake = _FakeHandle()
    monkeypatch.setattr(
        "mads.agents.wrapper.make_loader",
        lambda *, extra_tools=None, model=None: fake,
    )
    w = wrap_loader(manager, nudge_memory=False)
    w.invoke("load data")
    assert fake.last_prompt == "load data"


def test_wrap_loader_returns_turn_artifacts(manager, monkeypatch):
    fake = _FakeHandle(
        tool_calls=[{"name": "add_metric_definition", "args": {"metric_name": "x"}}],
    )
    monkeypatch.setattr(
        "mads.agents.wrapper.make_loader",
        lambda *, extra_tools=None, model=None: fake,
    )
    w = wrap_loader(manager)
    ta = w.invoke("load")
    assert isinstance(ta, TurnArtifacts)
    assert ta.agent_name == "loader"
    assert ta.question == "load"
    assert len(ta.memory_writes) == 1
    assert ta.memory_writes[0]["name"] == "add_metric_definition"
    assert ta.memory_reads == []


# --------------------------------------------------------------------- eda


def test_wrap_eda_injects_22_tools(manager, df, monkeypatch):
    captured = {}

    def fake_make_eda(d, *, extra_tools=None, model=None):
        captured["df"] = d
        captured["extra_tools"] = list(extra_tools or [])
        return _FakeHandle()

    monkeypatch.setattr(
        "mads.agents.wrapper.make_eda", fake_make_eda
    )
    w = wrap_eda(manager, df, dataset_name="ds", table_name="t")
    assert isinstance(w, WrappedEDA)
    # 5 metric + 4 preference + 6 episode + 6 dataset + 1 pandas-exec = 22.
    assert len(captured["extra_tools"]) == 22
    assert captured["df"] is df
    names = {t.name for t in captured["extra_tools"]}
    assert "remember_preference" in names
    assert "run_pandas_code" in names


def test_wrap_eda_classifies_reads_and_writes(manager, df, monkeypatch):
    fake = _FakeHandle(
        tool_calls=[
            {"name": "list_metric_definitions", "args": {}},
            {"name": "record_analysis_episode", "args": {"question": "q", "conclusion": "c"}},
            {"name": "explain_data", "args": {}},  # non-memory tool
        ],
    )
    monkeypatch.setattr(
        "mads.agents.wrapper.make_eda",
        lambda d, *, extra_tools=None, model=None: fake,
    )
    w = wrap_eda(manager, df, dataset_name="ds", table_name="t")
    ta = w.invoke("describe")
    # Reads include list_metric_definitions; writes include record_analysis_episode.
    read_names = {r["name"] for r in ta.memory_reads}
    write_names = {r["name"] for r in ta.memory_writes}
    assert "list_metric_definitions" in read_names
    assert "record_analysis_episode" in write_names
    # explain_data is not classified at all (not a memory tool).
    assert "explain_data" not in read_names | write_names


def test_wrap_eda_rejects_none_df(manager):
    with pytest.raises(ValueError):
        wrap_eda(manager, None, dataset_name="d", table_name="t")  # type: ignore[arg-type]


def test_wrap_eda_binds_dataframe(manager, df, monkeypatch):
    monkeypatch.setattr(
        "mads.agents.wrapper.make_eda",
        lambda d, *, extra_tools=None, model=None: _FakeHandle(),
    )
    wrap_eda(manager, df, dataset_name="ds", table_name="t")
    # Manager should have the df bound now.
    tools = manager.dataset_tools()
    assert len(tools) == 6


# --------------------------------------------------------------------- viz


def test_wrap_viz_injects_22_tools(manager, df, monkeypatch):
    captured = {}

    def fake_make_viz(d, *, extra_tools=None, model=None):
        captured["df"] = d
        captured["extra_tools"] = list(extra_tools or [])
        return _FakeHandle(response="viz-out")

    monkeypatch.setattr(
        "mads.agents.wrapper.make_viz", fake_make_viz
    )
    w = wrap_viz(manager, df, dataset_name="ds", table_name="t")
    assert isinstance(w, WrappedViz)
    # 5 metric + 4 preference + 6 episode + 6 dataset + 1 pandas-exec = 22.
    assert len(captured["extra_tools"]) == 22


def test_wrap_viz_rejects_none_df(manager):
    with pytest.raises(ValueError):
        wrap_viz(manager, None, dataset_name="d", table_name="t")  # type: ignore[arg-type]


def test_wrap_viz_returns_turn_artifacts(manager, df, monkeypatch):
    monkeypatch.setattr(
        "mads.agents.wrapper.make_viz",
        lambda d, *, extra_tools=None, model=None: _FakeHandle(
            response="plot",
        ),
    )
    w = wrap_viz(manager, df, dataset_name="ds", table_name="t")
    ta = w.invoke("plot it")
    assert ta.agent_name == "viz"
    assert ta.agent_result.response == "plot"


def test_classify_handles_string_tool_calls(manager, df, monkeypatch):
    fake = _FakeHandle(tool_calls=["add_metric_definition", "explain_data"])
    monkeypatch.setattr(
        "mads.agents.wrapper.make_eda",
        lambda d, *, extra_tools=None, model=None: fake,
    )
    w = wrap_eda(manager, df, dataset_name="ds", table_name="t")
    ta = w.invoke("q")
    write_names = {r["name"] for r in ta.memory_writes}
    assert "add_metric_definition" in write_names


# ----------------------------------------------------------- preference wiring
# Action C: preference tools must be wired through ``MemoryManager.all_tools``
# AND classified as memory_reads/memory_writes by the wrapper. These tests
# lock that contract down.


def test_all_tools_includes_preference(manager):
    """``MemoryManager.all_tools`` must include the 4 preference tools, in
    both ``include_dataset=False`` (15 total) and ``include_dataset=True``
    (22 total) variants. The 22-total variant additionally includes the
    1 ``run_pandas_code`` execution tool."""
    no_df = manager.all_tools(include_dataset=False)
    names_no_df = {t.name for t in no_df}
    assert len(no_df) == 15
    for n in (
        "remember_preference",
        "recall_preference",
        "list_preferences",
        "forget_preference",
    ):
        assert n in names_no_df, f"missing {n} in include_dataset=False"

    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    manager.bind_dataframe(df, dataset_name="d", table_name="t")
    with_df = manager.all_tools(include_dataset=True)
    names_with_df = {t.name for t in with_df}
    assert len(with_df) == 22
    for n in (
        "remember_preference",
        "recall_preference",
        "list_preferences",
        "forget_preference",
        "run_pandas_code",
    ):
        assert n in names_with_df


def test_preference_tools_args_schema_no_tenancy(manager):
    """Preference tools' LLM-visible signatures must not include user_id /
    project_id. Tenancy is closure-bound by the factory."""
    tools = manager.preference.tools()
    assert len(tools) == 4
    for t in tools:
        schema = t.args_schema
        # LangChain stores the pydantic model for tool args on ``args_schema``;
        # ``model_fields`` lists every input field the LLM is supposed to fill.
        fields = set(getattr(schema, "model_fields", {}).keys())
        assert "user_id" not in fields, (
            f"{t.name} leaks user_id into args_schema: {fields}"
        )
        assert "project_id" not in fields, (
            f"{t.name} leaks project_id into args_schema: {fields}"
        )


def test_wrapper_classifies_preference_tool_calls(manager, df, monkeypatch):
    """Synthesised ``tool_calls`` with preference reads + writes must be
    classified by the wrapper's ``_classify_tool_calls`` helper (through
    ``WrappedEDA.invoke``)."""
    fake = _FakeHandle(
        tool_calls=[
            {"name": "remember_preference", "args": {"key": "x", "value": "y"}},
            {"name": "recall_preference", "args": {"key": "x"}},
            {"name": "forget_preference", "args": {"key": "x"}},
            {"name": "list_preferences", "args": {}},
        ],
    )
    monkeypatch.setattr(
        "mads.agents.wrapper.make_eda",
        lambda d, *, extra_tools=None, model=None: fake,
    )
    w = wrap_eda(manager, df, dataset_name="ds", table_name="t")
    ta = w.invoke("q")
    write_names = {r["name"] for r in ta.memory_writes}
    read_names = {r["name"] for r in ta.memory_reads}
    # remember_preference + forget_preference are writes.
    assert "remember_preference" in write_names
    assert "forget_preference" in write_names
    # recall_preference + list_preferences are reads.
    assert "recall_preference" in read_names
    assert "list_preferences" in read_names


def test_memory_aware_eda_passes_preference_tools(manager, df, monkeypatch):
    """The proactive :class:`MemoryAwareEDA` wrapper must also pick up the
    4 preference tools via ``manager.all_tools(include_dataset=True)``."""
    from mads.wrappers.memory_aware import MemoryAwareEDA

    captured = {}

    def fake_make_eda(d, *, extra_tools=None, model=None):
        captured["extra_tools"] = list(extra_tools or [])
        return _FakeHandle()

    monkeypatch.setattr(
        "mads.wrappers.memory_aware.make_eda", fake_make_eda
    )
    aware = MemoryAwareEDA(manager, nudge_memory=False)
    aware.run(
        "describe",
        df=df,
        user_id=manager.user_id,
        project_id=manager.project_id,
        dataset_name="ds",
        table_name="t",
    )
    names = {t.name for t in captured["extra_tools"]}
    assert len(captured["extra_tools"]) == 22
    for n in (
        "remember_preference",
        "recall_preference",
        "list_preferences",
        "forget_preference",
        "run_pandas_code",
    ):
        assert n in names
