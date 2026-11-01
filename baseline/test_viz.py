#!/usr/bin/env python
"""Baseline test for ai_data_science_team.agents.DataVisualizationAgent.

Idempotent / self-contained:
  - Caches churn_data.csv at /Users/wang/LLM_memory/baseline/data/churn_data.csv.
  - If OPENAI_API_KEY is set -> live invocation, saves figure HTML and generated code.
  - Otherwise -> offline mode: verify import, instantiate (lazily), confirm dataframe loads,
    print shape + categorical value counts, print canonical live-run command.

Run:
    /Users/wang/LLM_memory/.venv/bin/python /Users/wang/LLM_memory/baseline/test_viz.py
"""

from __future__ import annotations

import os
import sys
import traceback
import urllib.request
from pathlib import Path

# -------- paths (absolute, per spec) -------------------------------------------------
DATA_PATH    = Path("/Users/wang/LLM_memory/baseline/data/churn_data.csv")
DATA_URL     = (
    "https://raw.githubusercontent.com/business-science/"
    "ai-data-science-team/master/data/churn_data.csv"
)
RESULTS_DIR  = Path("/Users/wang/LLM_memory/baseline/results")
PLOT_HTML    = RESULTS_DIR / "viz_plot.html"
GEN_CODE_PY  = RESULTS_DIR / "viz_generated.py"

USER_INSTRUCTIONS = "Plot churn rate by contract type as a bar chart."

# repro command (canonical)
REPRO_CMD = (
    "/Users/wang/LLM_memory/.venv/bin/python "
    "/Users/wang/LLM_memory/baseline/test_viz.py "
    "> /Users/wang/LLM_memory/baseline/results/viz.out "
    "2> /Users/wang/LLM_memory/baseline/results/viz.err"
)


def banner(title: str) -> None:
    print(f"\n===== {title} =====", flush=True)


def ensure_dirs() -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def ensure_data() -> None:
    """Download churn_data.csv once if missing."""
    if DATA_PATH.exists() and DATA_PATH.stat().st_size > 0:
        print(f"[data] cache hit -> {DATA_PATH} ({DATA_PATH.stat().st_size} bytes)")
        return
    print(f"[data] downloading -> {DATA_URL}")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    print(f"[data] saved -> {DATA_PATH} ({DATA_PATH.stat().st_size} bytes)")


def print_env() -> None:
    banner("ENVIRONMENT")
    print(f"python      : {sys.executable}")
    print(f"version     : {sys.version.split()[0]}")
    try:
        from importlib.metadata import version
        print(f"ads version : {version('ai-data-science-team')}")
    except Exception as e:  # pragma: no cover
        print(f"ads version : <unknown: {e}>")
    print(f"OPENAI_API_KEY set: {'yes' if os.environ.get('OPENAI_API_KEY') else 'no'}")


def offline_mode() -> int:
    banner("MODE: OFFLINE (no OPENAI_API_KEY)")
    # 1. import check
    try:
        from ai_data_science_team.agents import DataVisualizationAgent  # noqa: F401
        print("[import] DataVisualizationAgent OK")
    except Exception as e:
        print(f"[import] FAILED: {e!r}")
        traceback.print_exc()
        return 2

    # 2. dataframe load
    import pandas as pd
    df = pd.read_csv(DATA_PATH)
    print(f"[data] df.shape = {df.shape}")
    print(f"[data] columns  = {list(df.columns)}")

    # 3. categorical value-counts (pick a likely categorical column)
    target_col = None
    for cand in ("Contract", "contract", "ContractType"):
        if cand in df.columns:
            target_col = cand
            break
    if target_col is None:
        # fall back: first object-typed column
        objs = [c for c in df.columns if df[c].dtype == "object"]
        target_col = objs[0] if objs else None

    if target_col is not None:
        banner(f"value_counts: {target_col}")
        print(df[target_col].value_counts(dropna=False).to_string())
    else:
        print("[data] no obvious categorical column found")

    if "Churn" in df.columns:
        banner("value_counts: Churn")
        print(df["Churn"].value_counts(dropna=False).to_string())

    # 4. lazy instantiation sanity check — we cannot construct the agent without an LLM,
    #    but we can verify the class is callable and inspect its signature.
    banner("DataVisualizationAgent signature")
    import inspect
    from ai_data_science_team.agents import DataVisualizationAgent
    print(str(inspect.signature(DataVisualizationAgent.__init__)))
    print(str(inspect.signature(DataVisualizationAgent.invoke_agent)))

    # 5. canonical live-run command
    banner("CANONICAL LIVE-RUN COMMAND")
    print(
        "OPENAI_API_KEY=sk-... " + REPRO_CMD
    )

    banner("RESULT: OFFLINE — needs OPENAI_API_KEY")
    return 0


def live_mode() -> int:
    banner("MODE: LIVE")
    import pandas as pd
    from langchain_openai import ChatOpenAI
    from ai_data_science_team.agents import DataVisualizationAgent

    df = pd.read_csv(DATA_PATH)
    print(f"[data] df.shape = {df.shape}")
    print(f"[data] columns  = {list(df.columns)}")

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    agent = DataVisualizationAgent(
        model=llm,
        n_samples=30,
        log=False,
        bypass_recommended_steps=False,
        bypass_explain_code=False,
    )

    banner("invoke_agent")
    try:
        agent.invoke_agent(
            data_raw=df,                       # FIRST positional, per spec
            user_instructions=USER_INSTRUCTIONS,
            max_retries=3,
        )
    except Exception as e:
        print(f"[invoke] FAILED: {e!r}")
        traceback.print_exc()
        banner("RESULT: FAIL")
        return 3

    # --- collect outputs ------------------------------------------------------------
    fig      = None
    code_str = None
    steps_md = None
    log_md   = None
    try:
        fig = agent.get_plotly_graph()
        print(f"[fig] type = {type(fig).__module__}.{type(fig).__name__}")
    except Exception as e:
        print(f"[fig] get_plotly_graph FAILED: {e!r}")
        traceback.print_exc()

    try:
        code_str = agent.get_data_visualization_function()
        if code_str:
            banner("generated code (first 30 lines)")
            for i, ln in enumerate(code_str.splitlines()[:30], 1):
                print(f"{i:>3}: {ln}")
        else:
            print("[code] get_data_visualization_function returned empty")
    except Exception as e:
        print(f"[code] get_data_visualization_function FAILED: {e!r}")
        traceback.print_exc()

    try:
        steps_md = agent.get_recommended_visualization_steps(markdown=True)
        banner("recommended_visualization_steps (markdown)")
        print(steps_md)
    except Exception as e:
        print(f"[steps] FAILED: {e!r}")
        traceback.print_exc()

    try:
        log_md = agent.get_log_summary(markdown=True)
        banner("log_summary (markdown)")
        print(log_md)
    except Exception as e:
        print(f"[log] FAILED: {e!r}")
        traceback.print_exc()

    # --- persist artifacts ----------------------------------------------------------
    saved_html = False
    if fig is not None:
        try:
            # Plotly Figure has .write_html. If fig is a dict, coerce it.
            try:
                import plotly.graph_objs as go
                if isinstance(fig, dict):
                    fig = go.Figure(fig)
            except Exception:
                pass
            fig.write_html(str(PLOT_HTML))
            saved_html = True
            print(f"[save] figure -> {PLOT_HTML}")
        except Exception as e:
            print(f"[save] write_html FAILED: {e!r}")
            traceback.print_exc()
    else:
        print("[save] no figure to save")

    if code_str:
        try:
            GEN_CODE_PY.write_text(code_str)
            print(f"[save] code -> {GEN_CODE_PY}")
        except Exception as e:
            print(f"[save] write code FAILED: {e!r}")

    banner("RESULT: PASS" if (fig is not None and saved_html and code_str) else "RESULT: PARTIAL")
    return 0 if fig is not None else 4


def main() -> int:
    ensure_dirs()
    ensure_data()
    print_env()
    if os.environ.get("OPENAI_API_KEY"):
        return live_mode()
    return offline_mode()


if __name__ == "__main__":
    sys.exit(main())
