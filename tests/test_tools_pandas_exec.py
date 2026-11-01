"""Tests for :mod:`mads.tools.pandas_exec`.

We test the factory's LLM-visible contract directly:

* tool name is exactly ``run_pandas_code``
* args schema exposes only ``code: str`` (no df, no fingerprint, no tenancy)
* "no df bound" path returns the expected error string
* error mapping from ``execute_pandas`` is preserved verbatim
* one happy-path round-trip through the real sandbox confirms the JSON wrap
"""

from __future__ import annotations

from unittest import mock

import pandas as pd
import pytest

from mads.tools.pandas_exec import (
    PandasExecFactory,
    make_pandas_exec_factory,
)


@pytest.fixture
def small_df():
    return pd.DataFrame({"x": [1, 2, 3, 4], "Churn": ["Yes", "No", "Yes", "No"]})


# --------------------------------------------------------------------- shape


def test_factory_emits_single_tool(small_df):
    f = make_pandas_exec_factory(lambda: small_df)
    tools = f.tools()
    assert len(tools) == 1
    assert tools[0].name == "run_pandas_code"


def test_tool_args_schema_only_exposes_code(small_df):
    """Critical axis-model contract: the LLM-visible args expose ONLY
    ``code``. Neither the dataframe nor any tenancy field is reachable
    from the tool surface."""
    f = make_pandas_exec_factory(lambda: small_df)
    tool = f.tools()[0]
    fields = set(tool.args_schema.model_fields.keys())
    assert fields == {"code"}, f"unexpected fields: {fields}"


def test_make_pandas_exec_factory_returns_class():
    f = make_pandas_exec_factory(lambda: None)
    assert isinstance(f, PandasExecFactory)


# --------------------------------------------------------------------- error paths


def test_no_df_bound_returns_error_string():
    f = make_pandas_exec_factory(lambda: None)
    tool = f.tools()[0]
    out = tool.invoke({"code": "def compute(df): return {'n': 1}"})
    assert out.startswith("ERROR:")
    assert "no dataframe" in out.lower()


def test_execute_pandas_error_passed_through(small_df):
    """Adapter-level error strings are re-wrapped with the ``ERROR: `` prefix
    so the LLM gets a uniform shape regardless of failure mode."""
    f = make_pandas_exec_factory(lambda: small_df)
    tool = f.tools()[0]
    with mock.patch(
        "mads.tools.pandas_exec.execute_pandas",
        return_value=(None, "fake adapter error"),
    ):
        out = tool.invoke({"code": "def compute(df): return {}"})
    assert out == "ERROR: fake adapter error"


def test_result_serializes_as_json(small_df):
    """Successful execution returns a JSON string of the result dict."""
    f = make_pandas_exec_factory(lambda: small_df)
    tool = f.tools()[0]
    with mock.patch(
        "mads.tools.pandas_exec.execute_pandas",
        return_value=({"n": 4, "churn_pct": 50.0}, None),
    ):
        out = tool.invoke({"code": "def compute(df): return {}"})
    # JSON shape (not Python-repr); both keys present.
    assert '"n": 4' in out
    assert '"churn_pct": 50.0' in out


# --------------------------------------------------------------------- live get_df


def test_get_df_called_fresh_each_invocation():
    """The factory closes over ``get_df``, NOT over a snapshot of the df.
    A mid-conversation bind_dataframe swap is therefore picked up."""
    df_a = pd.DataFrame({"a": [1]})
    df_b = pd.DataFrame({"b": [2]})
    holder = {"df": df_a}

    f = make_pandas_exec_factory(lambda: holder["df"])
    tool = f.tools()[0]

    # First call sees df_a
    with mock.patch(
        "mads.tools.pandas_exec.execute_pandas",
        return_value=({"col": "a"}, None),
    ) as ep:
        tool.invoke({"code": "def compute(df): return {}"})
        assert ep.call_args.args[1] is df_a

    # Mutate the holder; next call must see df_b.
    holder["df"] = df_b
    with mock.patch(
        "mads.tools.pandas_exec.execute_pandas",
        return_value=({"col": "b"}, None),
    ) as ep:
        tool.invoke({"code": "def compute(df): return {}"})
        assert ep.call_args.args[1] is df_b


# --------------------------------------------------------------------- real round-trip


def test_real_sandbox_round_trip(small_df):
    """One real round-trip through the ADS subprocess sandbox confirms the
    JSON wrap end-to-end (not just the error/mocking paths)."""
    f = make_pandas_exec_factory(lambda: small_df)
    tool = f.tools()[0]
    code = """
def compute(df):
    return {"n": int(len(df)), "yes": int((df["Churn"] == "Yes").sum())}
"""
    out = tool.invoke({"code": code})
    # Result is a JSON string we can decode.
    import json

    parsed = json.loads(out)
    assert parsed == {"n": 4, "yes": 2}
