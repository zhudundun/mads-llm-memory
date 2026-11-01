#!/usr/bin/env python
"""Baseline test for ai_data_science_team.agents.DataLoaderToolsAgent.

Modes:
  LIVE   -> when OPENAI_API_KEY is set: full invocation; print df.head/shape/dtypes,
            AI message, and tool-call summary.
  OFFLINE -> otherwise: verify import + instantiation + graph node inspection.

Idempotent: caches CSV in baseline/data/churn_data.csv; safe to re-run.
"""
from __future__ import annotations

import os
import sys
import traceback
import urllib.request
from pathlib import Path

BASELINE_DIR = Path("/Users/wang/LLM_memory/baseline")
DATA_DIR = BASELINE_DIR / "data"
RESULTS_DIR = BASELINE_DIR / "results"
CSV_PATH = DATA_DIR / "churn_data.csv"
CSV_URL = (
    "https://raw.githubusercontent.com/business-science/"
    "ai-data-science-team/master/data/churn_data.csv"
)

REPRO_CMD = (
    "OPENAI_API_KEY=sk-... "
    "/Users/wang/LLM_memory/.venv/bin/python "
    "/Users/wang/LLM_memory/baseline/test_loader.py "
    "> /Users/wang/LLM_memory/baseline/results/loader.out "
    "2> /Users/wang/LLM_memory/baseline/results/loader.err"
)


def banner(title: str) -> None:
    line = "=" * 72
    print(f"\n{line}\n{title}\n{line}", flush=True)


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def ensure_csv() -> None:
    if CSV_PATH.exists() and CSV_PATH.stat().st_size > 0:
        print(f"[csv] cached at {CSV_PATH} ({CSV_PATH.stat().st_size} bytes)")
        return
    print(f"[csv] downloading from {CSV_URL}")
    urllib.request.urlretrieve(CSV_URL, CSV_PATH)
    print(f"[csv] saved to {CSV_PATH} ({CSV_PATH.stat().st_size} bytes)")


def describe_graph(agent) -> None:
    """Best-effort inspection of the compiled langgraph nodes."""
    try:
        graph = getattr(agent, "get_graph", None)
        if callable(graph):
            g = agent.get_graph()
            nodes = getattr(g, "nodes", None)
            if nodes is not None:
                try:
                    node_names = list(nodes.keys()) if hasattr(nodes, "keys") else list(nodes)
                except Exception:
                    node_names = [str(n) for n in nodes]
                print(f"[graph] nodes ({len(node_names)}): {node_names}")
                edges = getattr(g, "edges", None)
                if edges is not None:
                    try:
                        edge_list = list(edges)
                    except Exception:
                        edge_list = []
                    print(f"[graph] edges ({len(edge_list)}): {edge_list[:20]}")
                return
        # Fallback: poke at private attrs
        for attr in ("nodes", "_graph", "builder"):
            val = getattr(agent, attr, None)
            if val is not None:
                print(f"[graph] attr {attr}: {type(val).__name__}")
    except Exception as e:
        print(f"[graph] inspection failed: {type(e).__name__}: {e}")


def run_offline() -> int:
    banner("OFFLINE MODE - OPENAI_API_KEY not set")
    try:
        from langchain_openai import ChatOpenAI
        from ai_data_science_team.agents import DataLoaderToolsAgent
        import ai_data_science_team

        print(f"[import] ai_data_science_team OK at {ai_data_science_team.__file__}")
        print(f"[import] DataLoaderToolsAgent class: {DataLoaderToolsAgent}")
        print(f"[import] MRO: {[c.__name__ for c in DataLoaderToolsAgent.__mro__]}")

        # Instantiate with a dummy ChatOpenAI - constructor doesn't call the API
        llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key="sk-dummy-offline")
        print(f"[llm] instantiated ChatOpenAI: model={llm.model_name}")

        agent = DataLoaderToolsAgent(model=llm)
        print(f"[agent] instantiated: {type(agent).__name__}")

        # Public ADS-specific helpers we expect
        for m in ("invoke_agent", "ainvoke_agent", "get_artifacts",
                  "get_ai_message", "get_tool_calls", "get_response",
                  "get_internal_messages", "update_params", "show"):
            present = callable(getattr(agent, m, None))
            print(f"[agent] method {m}: {'OK' if present else 'MISSING'}")

        describe_graph(agent)

        banner("REPRO COMMAND for a live test")
        print(REPRO_CMD)
        print("\nOFFLINE SANITY: PASS")
        return 0
    except Exception as e:
        print(f"[offline] FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1


def run_live() -> int:
    banner("LIVE MODE - OPENAI_API_KEY detected")
    try:
        from langchain_openai import ChatOpenAI
        from ai_data_science_team.agents import DataLoaderToolsAgent

        ensure_csv()

        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        print(f"[llm] ChatOpenAI(model=gpt-4o, temperature=0)")

        agent = DataLoaderToolsAgent(model=llm)
        print(f"[agent] instantiated: {type(agent).__name__}")

        instr = (
            f"Load the CSV at {CSV_PATH} and return a small preview."
        )
        print(f"[invoke] user_instructions: {instr}")
        agent.invoke_agent(user_instructions=instr)
        print("[invoke] returned")

        banner("Artifacts (as_dataframe=True)")
        try:
            df = agent.get_artifacts(as_dataframe=True)
            print(f"type: {type(df).__name__}")
            if df is None:
                print("artifacts is None")
            else:
                try:
                    print(f"shape: {df.shape}")
                    print("dtypes:")
                    print(df.dtypes)
                    print("head():")
                    print(df.head())
                except AttributeError:
                    print(f"value: {df!r}")
        except Exception as e:
            print(f"[artifacts] error: {type(e).__name__}: {e}")
            traceback.print_exc()

        banner("AI message")
        try:
            msg = agent.get_ai_message(markdown=False)
            print(msg)
        except Exception as e:
            print(f"[ai_message] error: {type(e).__name__}: {e}")
            traceback.print_exc()

        banner("Tool calls (compact)")
        try:
            tools = agent.get_tool_calls()
            if tools is None:
                print("None")
            else:
                try:
                    n = len(tools)
                except Exception:
                    n = "?"
                print(f"count: {n}")
                if hasattr(tools, "__iter__"):
                    for i, t in enumerate(tools):
                        s = repr(t)
                        if len(s) > 500:
                            s = s[:500] + "...<truncated>"
                        print(f"  [{i}] {s}")
        except Exception as e:
            print(f"[tool_calls] error: {type(e).__name__}: {e}")
            traceback.print_exc()

        print("\nLIVE TEST: PASS")
        return 0
    except Exception as e:
        print(f"[live] FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("\nLIVE TEST: FAIL")
        return 2


def main() -> int:
    ensure_dirs()
    print(f"[env] python: {sys.executable}")
    print(f"[env] cwd: {os.getcwd()}")
    print(f"[env] OPENAI_API_KEY set: {bool(os.environ.get('OPENAI_API_KEY'))}")
    if os.environ.get("OPENAI_API_KEY"):
        return run_live()
    return run_offline()


if __name__ == "__main__":
    sys.exit(main())
