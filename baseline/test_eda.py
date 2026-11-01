#!/usr/bin/env python
"""Baseline test for ai_data_science_team.ds_agents.EDAToolsAgent.

Idempotent + self-contained:
- Ensures the churn CSV is cached at baseline/data/churn_data.csv.
- If OPENAI_API_KEY is set, runs a full live EDA invocation and prints
  AI message + tool call summary + artifact description.
- If OPENAI_API_KEY is NOT set, runs an offline sanity check: import path,
  lazy ChatOpenAI construction, dataframe load + shape/dtypes preview,
  and prints the canonical live command for the user to run.

Designed to be run with the project venv:
    /Users/wang/LLM_memory/.venv/bin/python /Users/wang/LLM_memory/baseline/test_eda.py
"""

from __future__ import annotations

import os
import sys
import traceback
import urllib.request
from pathlib import Path

DATA_DIR = Path("/Users/wang/LLM_memory/baseline/data")
DATA_PATH = DATA_DIR / "churn_data.csv"
DATA_URL = (
    "https://raw.githubusercontent.com/business-science/"
    "ai-data-science-team/master/data/churn_data.csv"
)


def banner(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n{title}\n{bar}", flush=True)


def ensure_data() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists() and DATA_PATH.stat().st_size > 0:
        print(f"[data] Cache hit: {DATA_PATH} ({DATA_PATH.stat().st_size} bytes)")
        return DATA_PATH
    print(f"[data] Downloading {DATA_URL} -> {DATA_PATH}")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    print(f"[data] Saved {DATA_PATH} ({DATA_PATH.stat().st_size} bytes)")
    return DATA_PATH


def describe_artifacts(artifacts) -> None:
    """Print a compact description of get_artifacts() output."""
    print(f"[artifacts] top-level type: {type(artifacts).__name__}")
    if artifacts is None:
        print("[artifacts] value: None")
        return
    if isinstance(artifacts, dict):
        print(f"[artifacts] keys ({len(artifacts)}): {list(artifacts.keys())}")
        for k, v in artifacts.items():
            t = type(v).__name__
            extra = ""
            try:
                import pandas as _pd

                if isinstance(v, _pd.DataFrame):
                    extra = f" shape={v.shape} cols={list(v.columns)[:8]}"
                elif isinstance(v, _pd.Series):
                    extra = f" len={len(v)}"
            except Exception:
                pass
            if isinstance(v, (list, tuple)):
                extra = f" len={len(v)}"
            if isinstance(v, str):
                preview = v.replace("\n", " ")[:120]
                extra = f" str_preview={preview!r}"
            print(f"  - {k!r}: {t}{extra}")
    elif isinstance(artifacts, list):
        print(f"[artifacts] list len={len(artifacts)}")
        for i, item in enumerate(artifacts[:5]):
            print(f"  [{i}] {type(item).__name__}: {str(item)[:160]!r}")
    else:
        print(f"[artifacts] repr preview: {str(artifacts)[:400]!r}")


def summarize_tool_calls(tools) -> None:
    print(f"[tools] type: {type(tools).__name__}")
    if not tools:
        print("[tools] (none)")
        return
    from collections import Counter

    names = []
    if isinstance(tools, list):
        for entry in tools:
            if isinstance(entry, dict):
                names.append(entry.get("name") or entry.get("tool") or "<unknown>")
            else:
                names.append(getattr(entry, "name", None) or str(entry)[:60])
    else:
        names.append(str(tools)[:60])
    counts = Counter(names)
    print(f"[tools] total_calls={len(names)} unique={len(counts)}")
    for name, n in counts.most_common():
        print(f"  - {name}: {n}")


def offline_mode() -> int:
    banner("MODE: OFFLINE (OPENAI_API_KEY not set)")
    try:
        from ai_data_science_team.ds_agents import EDAToolsAgent

        print(f"[import] EDAToolsAgent -> {EDAToolsAgent.__module__}.{EDAToolsAgent.__name__}")
    except Exception:
        print("[import] FAILED to import EDAToolsAgent from ds_agents")
        traceback.print_exc()
        return 2

    try:
        from langchain_openai import ChatOpenAI

        # ChatOpenAI eagerly validates credentials, so pass a placeholder for
        # the offline construction check. No network is hit until invoke time.
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key="sk-offline-placeholder",
        )
        print(f"[llm] Instantiated lazily: {type(llm).__name__} model=gpt-4o")
    except Exception:
        print("[llm] FAILED to construct ChatOpenAI")
        traceback.print_exc()
        return 3

    try:
        agent = EDAToolsAgent(model=llm)
        print(f"[agent] Instantiated: {type(agent).__name__}")
        # Surface key public methods to confirm API shape.
        api_methods = [
            "invoke_agent",
            "ainvoke_agent",
            "get_artifacts",
            "get_ai_message",
            "get_tool_calls",
            "get_response",
        ]
        for m in api_methods:
            print(f"  - has {m}: {hasattr(agent, m)}")
    except Exception:
        print("[agent] FAILED to instantiate EDAToolsAgent")
        traceback.print_exc()
        return 4

    try:
        import pandas as pd

        path = ensure_data()
        df = pd.read_csv(path)
        print(f"[data] df.shape = {df.shape}")
        print(f"[data] columns ({len(df.columns)}): {list(df.columns)}")
        print("[data] df.dtypes:")
        for col, dt in df.dtypes.items():
            print(f"  - {col}: {dt}")
        print("[data] head(3):")
        print(df.head(3).to_string(index=False))
    except Exception:
        print("[data] FAILED to load dataframe")
        traceback.print_exc()
        return 5

    banner("CANONICAL LIVE COMMAND")
    print(
        "Set your key, then re-run this script:\n"
        "    export OPENAI_API_KEY=sk-...\n"
        "    /Users/wang/LLM_memory/.venv/bin/python "
        "/Users/wang/LLM_memory/baseline/test_eda.py"
    )
    banner("OFFLINE SANITY CHECK PASSED")
    return 0


def live_mode() -> int:
    banner("MODE: LIVE (OPENAI_API_KEY detected)")
    import pandas as pd

    try:
        from ai_data_science_team.ds_agents import EDAToolsAgent
        from langchain_openai import ChatOpenAI
    except Exception:
        print("[import] FAILED")
        traceback.print_exc()
        return 2

    path = ensure_data()
    try:
        df = pd.read_csv(path)
        print(f"[data] df.shape = {df.shape}")
        print(f"[data] dtypes:\n{df.dtypes}")
    except Exception:
        print("[data] FAILED to read csv")
        traceback.print_exc()
        return 5

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    print(f"[llm] {type(llm).__name__} model=gpt-4o temperature=0")

    agent = EDAToolsAgent(model=llm)
    print(f"[agent] Instantiated: {type(agent).__name__}")

    user_instructions = (
        "Run a quick EDA: shape, dtypes, missing-value summary, and "
        "notable correlations."
    )

    banner("INVOKE EDAToolsAgent")
    try:
        agent.invoke_agent(user_instructions=user_instructions, data_raw=df)
        print("[invoke] invoke_agent returned without exception.")
    except Exception:
        print("[invoke] invoke_agent raised — capturing for postmortem.")
        traceback.print_exc()
        # Still attempt to dump whatever state is present.
        try:
            banner("POSTMORTEM: get_ai_message()")
            msg = agent.get_ai_message(markdown=False)
            print(msg)
        except Exception:
            traceback.print_exc()
        try:
            banner("POSTMORTEM: get_tool_calls()")
            summarize_tool_calls(agent.get_tool_calls())
        except Exception:
            traceback.print_exc()
        try:
            banner("POSTMORTEM: get_artifacts()")
            describe_artifacts(agent.get_artifacts(as_dataframe=False))
        except Exception:
            traceback.print_exc()
        return 6

    try:
        banner("AI MESSAGE")
        msg = agent.get_ai_message(markdown=False)
        print(msg if msg else "<empty AI message>")
    except Exception:
        print("[get_ai_message] FAILED")
        traceback.print_exc()

    try:
        banner("TOOL CALLS SUMMARY")
        tools = agent.get_tool_calls()
        summarize_tool_calls(tools)
    except Exception:
        print("[get_tool_calls] FAILED")
        traceback.print_exc()

    try:
        banner("ARTIFACTS")
        artifacts = agent.get_artifacts(as_dataframe=False)
        describe_artifacts(artifacts)
    except Exception:
        print("[get_artifacts] FAILED")
        traceback.print_exc()

    banner("LIVE INVOCATION COMPLETE")
    return 0


def main() -> int:
    banner("EDAToolsAgent baseline test")
    print(f"python: {sys.executable}")
    print(f"cwd: {os.getcwd()}")
    try:
        import ai_data_science_team as _ads

        print(f"ai_data_science_team: {getattr(_ads, '__version__', 'n/a')} @ {_ads.__file__}")
    except Exception:
        print("[env] ai_data_science_team not importable")
        traceback.print_exc()
        return 1

    if os.environ.get("OPENAI_API_KEY"):
        return live_mode()
    return offline_mode()


if __name__ == "__main__":
    sys.exit(main())
