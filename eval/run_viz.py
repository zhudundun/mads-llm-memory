"""VIZ-RUNNER driver: run the 5 viz scenarios x 2 sessions through the harness.

Touches only viz outputs:
    /Users/wang/LLM_memory/eval/results/viz/
        - per-session .json/.md  (written by harness)
        - figures/<id>__session_<a|b>.html
        - code/<id>__session_<a|b>.py
        - runs.jsonl
        - SUMMARY.md
        - run.log (caller-tee'd)

Design note: we do NOT modify the harness. To capture the Plotly Figure and
generated code artifacts, we bypass run_scenario_pair() and drive each session
manually using the same primitives (agent_factory + invoke_for_agent). The
per-session JSON/MD persistence is replicated locally so the on-disk layout
matches what run_scenario_pair() would have produced.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

# Make eval importable as a package
REPO_ROOT = Path("/Users/wang/LLM_memory")
sys.path.insert(0, str(REPO_ROOT))

from eval.harness import (  # noqa: E402
    SessionRecord,
    _compute_cost_usd,
    _invoke_once,
    _normalize_tool_calls,
    _persist,
)
from eval.agent_factories import make_viz_factory, invoke_for_agent  # noqa: E402


SCENARIOS_PATH = REPO_ROOT / "eval" / "scenarios.json"
DATA_PATH = REPO_ROOT / "baseline" / "data" / "churn_data.csv"
RESULT_DIR = REPO_ROOT / "eval" / "results" / "viz"
FIG_DIR = RESULT_DIR / "figures"
CODE_DIR = RESULT_DIR / "code"
RUNS_JSONL = RESULT_DIR / "runs.jsonl"
SUMMARY_MD = RESULT_DIR / "SUMMARY.md"
TIMEOUT_S = 180


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _save_side_artifacts(agent: Any, scenario_id: str, session_label: str) -> Dict[str, bool]:
    """After a session runs, pull the plotly figure + code and persist them.

    Returns flags for the SUMMARY row.
    """
    fig_saved = False
    code_saved = False

    # Figure -> HTML
    try:
        fig = agent.get_plotly_graph()
    except Exception:
        fig = None
    if fig is not None:
        fig_path = FIG_DIR / f"{scenario_id}__session_{session_label}.html"
        try:
            # Plotly Figure path
            fig.write_html(str(fig_path))
            fig_saved = True
        except Exception:
            # Some ADS versions return a dict/JSON-serializable spec
            try:
                fig_path.write_text(json.dumps(fig, default=str))
                fig_saved = True
            except Exception:
                pass

    # Code -> .py
    try:
        code = agent.get_data_visualization_function(markdown=False)
    except Exception:
        code = None
    if code:
        code_path = CODE_DIR / f"{scenario_id}__session_{session_label}.py"
        code_path.write_text(code)
        code_saved = True

    return {"fig_saved": fig_saved, "code_saved": code_saved}


def run_one_session(
    scenario: Dict[str, Any],
    session_label: str,
    agent_factory,
) -> tuple[SessionRecord, Dict[str, bool]]:
    """Single-session runner that mirrors harness.run_scenario_pair's per-session
    logic but also pulls viz side-artifacts (figure HTML + code .py).
    """
    sess = scenario[f"session_{session_label}"]
    rec = SessionRecord(
        scenario_id=scenario["id"],
        session=session_label,
        agent_name=scenario["agent"],
        prompt=sess["prompt"],
        intent=sess["intent"],
    )
    side = {"fig_saved": False, "code_saved": False}

    try:
        agent = agent_factory()
    except BaseException as e:
        rec.error = f"factory_error: {type(e).__name__}: {e}"
        rec.error_traceback = traceback.format_exc()
        _persist(rec, RESULT_DIR)
        return rec, side

    def _invoke_fn(prompt: str, ag: Any) -> Dict[str, Any]:
        return invoke_for_agent("viz", ag, prompt)

    payload = _invoke_once(
        agent=agent,
        agent_name="viz",
        prompt=sess["prompt"],
        invoke_fn=_invoke_fn,
        timeout_s=TIMEOUT_S,
    )

    rec.response = payload.get("response")
    rec.latency_ms = payload.get("latency_ms")
    rec.prompt_tokens = payload.get("prompt_tokens", 0)
    rec.completion_tokens = payload.get("completion_tokens", 0)
    rec.total_tokens = payload.get("total_tokens", 0)
    rec.cost_usd = round(
        _compute_cost_usd(rec.prompt_tokens, rec.completion_tokens), 6
    )
    rec.viz_code = payload.get("viz_code")
    rec.error = payload.get("error")
    rec.error_traceback = payload.get("error_traceback")

    records, names, count = _normalize_tool_calls(payload.get("tool_calls"))
    rec.tool_calls = records
    rec.tool_call_names = names
    rec.tool_call_count = count

    # Pull side artifacts from the agent (even on error — some agents leave
    # partial state).
    try:
        side = _save_side_artifacts(agent, scenario["id"], session_label)
    except Exception:
        pass

    _persist(rec, RESULT_DIR)
    return rec, side


def run_pair(scenario: Dict[str, Any], agent_factory) -> Dict[str, Any]:
    rec_a, side_a = run_one_session(scenario, "a", agent_factory)
    rec_b, side_b = run_one_session(scenario, "b", agent_factory)
    ok = rec_a.error is None and rec_b.error is None
    combined_err = rec_a.error or rec_b.error
    return {
        "scenario_id": scenario["id"],
        "agent": scenario["agent"],
        "a": rec_a.to_dict(),
        "b": rec_b.to_dict(),
        "side_a": side_a,
        "side_b": side_b,
        "ok": ok,
        "error": combined_err,
        "timestamp": _now_iso(),
    }


def _b_followed_preference(scenario_id: str, code_path: Path, fig_path: Path) -> str:
    """Heuristic check on session_b artifacts. Returns 'yes' | 'no' | 'partial' | 'n/a'."""
    if not code_path.exists():
        return "n/a"
    code = code_path.read_text().lower()
    fig_text = fig_path.read_text().lower() if fig_path.exists() else ""

    if scenario_id == "viz_01_horizontal_viridis":
        has_viridis = "viridis" in code or "viridis" in fig_text
        # heuristic for horizontal: orientation='h' OR a known horizontal pattern
        has_horizontal = (
            "orientation='h'" in code
            or 'orientation="h"' in code
            or "orientation = 'h'" in code
            or 'orientation = "h"' in code
        )
        if has_viridis and has_horizontal:
            return "yes"
        if has_viridis or has_horizontal:
            return "partial"
        return "no"

    if scenario_id == "viz_02_log_y_axis":
        # observable: any mention of log scale OR explicit dynamic-range rule
        if "log" in code and ("yaxis" in code or "type='log'" in code or 'type="log"' in code):
            return "yes"
        if "log" in code:
            return "partial"
        return "no"

    if scenario_id == "viz_03_exec_minimal":
        has_title = "title" in code and "title=''" not in code
        has_no_legend = "showlegend=false" in code or "showlegend = false" in code
        has_subtitle = "subtitle" in code or "annotation" in code or "<br>" in code
        if has_title and has_no_legend and has_subtitle:
            return "yes"
        hits = sum([has_title, has_no_legend, has_subtitle])
        if hits >= 1:
            return "partial"
        return "no"

    if scenario_id == "viz_04_contract_color_map":
        # Looking for explicit color mapping Month-to-month=red etc.
        has_map = "color_discrete_map" in code or "color_discrete_sequence" in code
        # Check that all 3 of red/blue/green are referenced
        colors_referenced = sum(c in code for c in ["red", "blue", "green"])
        if has_map and colors_referenced >= 3:
            return "yes"
        if has_map or colors_referenced >= 2:
            return "partial"
        return "no"

    if scenario_id == "viz_05_tenure_bucket_grain":
        # Looking for binning with edges 12/24/48
        has_buckets = ("pd.cut" in code or "cut(" in code) and (
            "12" in code and "24" in code and "48" in code
        )
        if has_buckets:
            return "yes"
        if "tenure_bucket" in code or "bucket" in code:
            return "partial"
        return "no"

    return "n/a"


def _one_line_observation(scenario_id: str, side_b: Dict[str, bool], b_followed: str) -> str:
    if side_b.get("code_saved") is False:
        return "no code captured for session B"
    obs_map = {
        "viz_01_horizontal_viridis": "looked for orientation='h' + viridis in B",
        "viz_02_log_y_axis": "looked for log scale handling in B",
        "viz_03_exec_minimal": "looked for title + no-legend + subtitle in B",
        "viz_04_contract_color_map": "looked for color_discrete_map + red/blue/green in B",
        "viz_05_tenure_bucket_grain": "looked for tenure bucketing 12/24/48 in B",
    }
    base = obs_map.get(scenario_id, "")
    return f"{base}; verdict={b_followed}"


def main() -> int:
    # 1) Create output dirs
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CODE_DIR.mkdir(parents=True, exist_ok=True)

    # Truncate runs.jsonl at start so a re-run is clean
    if RUNS_JSONL.exists():
        RUNS_JSONL.unlink()

    # 2) Load scenarios + filter to viz
    payload = json.loads(SCENARIOS_PATH.read_text())
    viz_scenarios = [s for s in payload["scenarios"] if s["agent"] == "viz"]
    print(f"[viz-runner] Loaded {len(viz_scenarios)} viz scenarios")

    # 3) Load df ONCE
    df = pd.read_csv(DATA_PATH)
    print(f"[viz-runner] Loaded dataframe: shape={df.shape}")

    # 4) Build factory
    factory = make_viz_factory(df)

    # 5) Run each scenario
    pairs: list[Dict[str, Any]] = []
    t_start = time.monotonic()
    for idx, scenario in enumerate(viz_scenarios, start=1):
        sid = scenario["id"]
        print(f"\n[viz-runner] ({idx}/{len(viz_scenarios)}) {sid} — running pair...")
        try:
            pair = run_pair(scenario, factory)
        except BaseException as e:
            print(f"[viz-runner]   FATAL during pair: {type(e).__name__}: {e}")
            pair = {
                "scenario_id": sid,
                "agent": "viz",
                "a": {"error": f"runner_error: {type(e).__name__}: {e}"},
                "b": {"error": None},
                "side_a": {"fig_saved": False, "code_saved": False},
                "side_b": {"fig_saved": False, "code_saved": False},
                "ok": False,
                "error": f"runner_error: {type(e).__name__}: {e}",
                "timestamp": _now_iso(),
            }

        # Append to runs.jsonl immediately
        with RUNS_JSONL.open("a") as fh:
            fh.write(json.dumps(pair, default=str) + "\n")

        pairs.append(pair)

        a = pair.get("a", {}) or {}
        b = pair.get("b", {}) or {}
        side_a = pair.get("side_a", {}) or {}
        side_b = pair.get("side_b", {}) or {}
        tokens = (a.get("total_tokens", 0) or 0) + (b.get("total_tokens", 0) or 0)
        cost = (a.get("cost_usd", 0.0) or 0.0) + (b.get("cost_usd", 0.0) or 0.0)
        fig_flags = f"{'Y' if side_a.get('fig_saved') else 'N'}/{'Y' if side_b.get('fig_saved') else 'N'}"
        code_flags = f"{'Y' if side_a.get('code_saved') else 'N'}/{'Y' if side_b.get('code_saved') else 'N'}"
        notes = pair.get("error") or "ok"
        print(
            f"[viz-runner]   {sid} | ok={pair['ok']} | tokens(a+b)={tokens} | "
            f"cost(a+b)=${cost:.4f} | fig a/b={fig_flags} | code a/b={code_flags} | notes={notes}"
        )

    elapsed = time.monotonic() - t_start
    print(f"\n[viz-runner] Completed {len(pairs)} pairs in {elapsed:.1f}s")

    # 6) Totals
    total_tokens = sum(
        (p.get("a", {}).get("total_tokens", 0) or 0) + (p.get("b", {}).get("total_tokens", 0) or 0)
        for p in pairs
    )
    total_cost = sum(
        (p.get("a", {}).get("cost_usd", 0.0) or 0.0) + (p.get("b", {}).get("cost_usd", 0.0) or 0.0)
        for p in pairs
    )
    passed = sum(1 for p in pairs if p.get("ok"))
    failed = len(pairs) - passed
    print(f"[viz-runner] Total tokens: {total_tokens}")
    print(f"[viz-runner] Total cost USD: ${total_cost:.4f}")
    print(f"[viz-runner] PASS: {passed}/{len(pairs)}  FAIL: {failed}")

    # 7) SUMMARY.md
    status = (
        "PASS" if failed == 0 else ("PARTIAL" if passed > 0 else "FAIL")
    )
    lines: list[str] = []
    lines.append("# VIZ-RUNNER SUMMARY")
    lines.append("")
    lines.append(f"- **Status**: {status} ({passed}/{len(pairs)} pairs ok, {failed} failed)")
    lines.append(f"- **Model**: gpt-4o (real OpenAI calls)")
    lines.append(f"- **Timestamp (UTC)**: {_now_iso()}")
    lines.append(f"- **Total wallclock**: {elapsed:.1f}s")
    lines.append("")
    lines.append("## Results table")
    lines.append("")
    lines.append(
        "| id | tokens (a+b) | cost USD (a+b) | latency A/B (ms) | fig A/B | "
        "B followed pref? | observation |"
    )
    lines.append("|---|---:|---:|---|:---:|:---:|---|")

    memory_gap_rows: list[tuple[str, str]] = []
    for p in pairs:
        sid = p["scenario_id"]
        a = p.get("a", {}) or {}
        b = p.get("b", {}) or {}
        side_a = p.get("side_a", {}) or {}
        side_b = p.get("side_b", {}) or {}
        tokens = (a.get("total_tokens", 0) or 0) + (b.get("total_tokens", 0) or 0)
        cost = (a.get("cost_usd", 0.0) or 0.0) + (b.get("cost_usd", 0.0) or 0.0)
        lat = f"{a.get('latency_ms', '-')}/{b.get('latency_ms', '-')}"
        fig_flags = f"{'Y' if side_a.get('fig_saved') else 'N'}/{'Y' if side_b.get('fig_saved') else 'N'}"

        code_b_path = CODE_DIR / f"{sid}__session_b.py"
        fig_b_path = FIG_DIR / f"{sid}__session_b.html"
        b_followed = _b_followed_preference(sid, code_b_path, fig_b_path)
        obs = _one_line_observation(sid, side_b, b_followed)

        lines.append(
            f"| {sid} | {tokens} | ${cost:.4f} | {lat} | {fig_flags} | "
            f"{b_followed} | {obs} |"
        )

        # Memory-gap evidence collection: a "no" verdict IS the gap
        if b_followed in ("no", "partial"):
            memory_gap_rows.append((sid, b_followed))

    lines.append("")
    lines.append("## Totals")
    lines.append("")
    lines.append(f"- **Total tokens** (a+b across all pairs): {total_tokens}")
    lines.append(f"- **Total cost USD**: ${total_cost:.4f}")
    lines.append("")
    lines.append("## Memory-gap evidence")
    lines.append("")
    lines.append(
        "Baseline (no memory) hypothesis: session B has no access to session A's stated "
        "preference, so the observable named in each scenario should be ABSENT in B. "
        "A 'no' verdict above is positive evidence of the gap; 'yes' is a weak signal "
        "(e.g. defaults coincidentally match)."
    )
    lines.append("")
    if memory_gap_rows:
        lines.append("Scenarios where session B clearly missed the preference (gap evident):")
        lines.append("")
        for sid, verdict in memory_gap_rows:
            scen = next((s for s in viz_scenarios if s["id"] == sid), None)
            hyp = scen.get("memory_hypothesis", "") if scen else ""
            obsv = scen.get("observable", "") if scen else ""
            lines.append(f"- **{sid}** (verdict={verdict})")
            lines.append(f"    - observable expected if memory worked: {obsv}")
            lines.append(f"    - hypothesis: {hyp}")
    else:
        lines.append("(No clear gaps detected — every B happened to match its A preference. "
                     "Likely defaults aligned with the preference; not strong evidence either way.)")

    lines.append("")
    lines.append("## Repro command")
    lines.append("")
    lines.append("```bash")
    lines.append(
        "cd /Users/wang/LLM_memory && set -a && source .env && set +a && "
        ".venv/bin/python eval/run_viz.py 2>&1 | tee eval/results/viz/run.log"
    )
    lines.append("```")
    lines.append("")

    SUMMARY_MD.write_text("\n".join(lines))
    print(f"[viz-runner] Wrote SUMMARY.md -> {SUMMARY_MD}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
