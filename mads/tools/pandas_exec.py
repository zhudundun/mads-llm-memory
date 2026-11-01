"""Tool factory for ``run_pandas_code`` — executes LLM-generated pandas code
against the currently-bound dataframe via the ADS subprocess sandbox.

This is the ONE LLM-visible execution surface. The LLM writes a function
``def compute(df) -> dict``; the factory closes over a live ``get_df``
callable so the most-recently-bound df is always picked up, even if
``MemoryManager.bind_dataframe`` is called mid-conversation.

The factory pattern mirrors the memory category factories
(``mads/memory/metric_tools.py``, etc.): a small class with a ``.tools()``
method that returns ``@tool``-decorated callables. Tool args expose only
``code: str`` — neither the df nor any tenancy is reachable through the
tool surface, by design.
"""

from __future__ import annotations

import json
from typing import Callable, List, Optional

import pandas as pd
from langchain_core.tools import BaseTool, tool

from mads.adapters.ads import execute_pandas


class PandasExecFactory:
    """Build the ``run_pandas_code`` tool list closing over ``get_df``.

    ``get_df`` is invoked on EVERY tool call (no caching), so a
    ``MemoryManager.bind_dataframe`` swap mid-conversation is picked up
    automatically.
    """

    def __init__(
        self, get_df: Callable[[], Optional[pd.DataFrame]]
    ) -> None:
        self._get_df = get_df

    def tools(self) -> List[BaseTool]:
        get_df = self._get_df

        @tool
        def run_pandas_code(code: str) -> str:
            """Execute pandas code against the currently loaded dataframe.

            Your code MUST define ``def compute(df) -> dict``. The function
            receives the bound dataframe and must return a dict whose values
            are JSON-serializable scalars (numbers, strings, bools, lists of
            those). Use this when memory gives you a metric definition or
            filter rule and you need the actual number — DO NOT estimate
            from training data; DO NOT skip the computation.

            Example::

                run_pandas_code('''
                def compute(df):
                    qual = df.query("tenure > 12 and MonthlyCharges > 50")
                    pct = float((qual["Churn"] == "Yes").mean() * 100)
                    return {"qualified_count": int(len(qual)), "churn_pct": pct}
                ''')

            Returns a JSON string of the result dict, or ``"ERROR: ..."`` on
            failure. The sandbox blocks os/sys/subprocess/network access and
            enforces a 30s timeout.
            """
            df = get_df()
            if df is None:
                return "ERROR: no dataframe is bound to this session."
            result, err = execute_pandas(code, df)
            if err is not None:
                return f"ERROR: {err}"
            try:
                return json.dumps(result, allow_nan=False, default=str)
            except Exception as exc:  # noqa: BLE001
                return f"ERROR: result not JSON-serializable: {exc}"

        return [run_pandas_code]


def make_pandas_exec_factory(
    get_df: Callable[[], Optional[pd.DataFrame]],
) -> PandasExecFactory:
    """Construct a :class:`PandasExecFactory` bound to ``get_df``.

    ``get_df`` should be a zero-arg callable returning the current
    DataFrame (or ``None`` if no df is bound).
    """
    return PandasExecFactory(get_df)


__all__ = ["PandasExecFactory", "make_pandas_exec_factory"]
