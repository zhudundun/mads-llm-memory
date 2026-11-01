"""Resume: re-run only viz_03 / viz_04 / viz_05 (the 3 Viz pairs that didn't
complete in the previous run because Anthropic credits were exhausted).

Reuses the helpers from eval.run_viz so per-session JSON/MD layout, figure/code
side-artifacts, and the b_followed_preference heuristic stay identical to the
canonical driver. Final state matches what a clean full run would produce:

    eval/results/viz/runs.jsonl    — 5 pair entries (viz_01..viz_05)
    eval/results/viz/SUMMARY.md    — regenerated, correctly labels Sonnet 4.6
    eval/results/viz/*.json/*.md   — per-session; viz_03/04/05 overwritten
    eval/results/viz/figures/*.html, code/*.py — viz_03/04/05 fresh, viz_01/02 untouched
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path("/Users/wang/LLM_memory")
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from eval.agent_factories import make_viz_factory  # noqa: E402
from eval.run_viz import (  # noqa: E402
    CODE_DIR,
    DATA_PATH,
    FIG_DIR,
    RESULT_DIR,
    RUNS_JSONL,
    SCENARIOS_PATH,
    SUMMARY_MD,
    _b_followed_preference,
    _one_line_observation,
    run_pair,
)


RESUME_IDS = {
    "viz_03_exec_minimal",
    "viz_04_contract_color_map",
    "viz_05_tenure_bucket_grain",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CODE_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Preserve any previously-passing pair entries we don't intend to re-run.
    existing: dict[str, dict] = {}
    if RUNS_JSONL.exists():
        for line in RUNS_JSONL.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            sid = r.get("scenario_id")
            if sid and sid not in RESUME_IDS and r.get("ok"):
                existing[sid] = r
    print(f"[viz-resume] preserving {len(existing)} prior PASS pairs: {sorted(existing)}")

    # 2) Filter scenarios to the 3 we're resuming.
    payload = json.loads(SCENARIOS_PATH.read_text())
    viz_scenarios = [s for s in payload["scenarios"] if s["agent"] == "viz"]
    todo = [s for s in viz_scenarios if s["id"] in RESUME_IDS]
    assert len(todo) == 3, f"expected 3 to resume, got {len(todo)}"

    # 3) Load df + factory once.
    df = pd.read_csv(DATA_PATH)
    factory = make_viz_factory(df)
    print(f"[viz-resume] loaded df {df.shape}; running {len(todo)} pairs")

    # 4) Run the 3 pairs (sequentially — each session is short).
    new_pairs: list[dict] = []
    t_start = time.monotonic()
    for idx, scenario in enumerate(todo, start=1):
        sid = scenario["id"]
        print(f"\n[viz-resume] ({idx}/{len(todo)}) {sid} — running pair...")
        pair = run_pair(scenario, factory)
        new_pairs.append(pair)
        a, b = pair.get("a", {}) or {}, pair.get("b", {}) or {}
        tokens = (a.get("total_tokens", 0) or 0) + (b.get("total_tokens", 0) or 0)
        cost = (a.get("cost_usd", 0.0) or 0.0) + (b.get("cost_usd", 0.0) or 0.0)
        notes = pair.get("error") or "ok"
        print(
            f"[viz-resume]   {sid} | ok={pair['ok']} | tokens(a+b)={tokens} | "
            f"cost(a+b)=${cost:.4f} | notes={notes}"
        )
    elapsed = time.monotonic() - t_start
    print(f"\n[viz-resume] resume wallclock: {elapsed:.1f}s")

    # 5) Combine and rebuild runs.jsonl in canonical scenario order.
    by_id = dict(existing)
    for p in new_pairs:
        by_id[p["scenario_id"]] = p
    pairs = [by_id[s["id"]] for s in viz_scenarios if s["id"] in by_id]
    with RUNS_JSONL.open("w") as fh:
        for p in pairs:
            fh.write(json.dumps(p, default=str) + "\n")
    print(f"[viz-resume] wrote runs.jsonl with {len(pairs)} pairs")

    # 6) Rebuild SUMMARY.md.
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
    status = "PASS" if failed == 0 else ("PARTIAL" if passed > 0 else "FAIL")

    L: list[str] = []
    L.append("# VIZ-RUNNER SUMMARY")
    L.append("")
    L.append(f"- **Status**: {status} ({passed}/{len(pairs)} pairs ok, {failed} failed)")
    L.append("- **Model**: claude-sonnet-4-6 (via langchain-anthropic)")
    L.append(f"- **Timestamp (UTC)**: {_now_iso()}")
    L.append(f"- **Resume wallclock** (last 3 pairs only): {elapsed:.1f}s")
    L.append(
        "- **Patched**: yes — local fix to "
        "ai_data_science_team/agents/data_visualization_agent.py (lines 773/775/777, "
        "doubled-backslash regex bug)."
    )
    L.append("")
    L.append("## Results table")
    L.append("")
    L.append(
        "| id | tokens (a+b) | cost USD (a+b) | latency A/B (ms) | fig A/B | "
        "B followed pref? | observation |"
    )
    L.append("|---|---:|---:|---|:---:|:---:|---|")

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
        fig_flags = (
            f"{'Y' if side_a.get('fig_saved') else 'N'}/"
            f"{'Y' if side_b.get('fig_saved') else 'N'}"
        )
        code_b_path = CODE_DIR / f"{sid}__session_b.py"
        fig_b_path = FIG_DIR / f"{sid}__session_b.html"
        b_followed = _b_followed_preference(sid, code_b_path, fig_b_path)
        obs = _one_line_observation(sid, side_b, b_followed)
        L.append(
            f"| {sid} | {tokens} | ${cost:.4f} | {lat} | {fig_flags} | "
            f"{b_followed} | {obs} |"
        )
        if b_followed in ("no", "partial"):
            memory_gap_rows.append((sid, b_followed))

    L.append("")
    L.append("## Totals")
    L.append("")
    L.append(f"- **Total tokens** (a+b across all pairs): {total_tokens}")
    L.append(f"- **Total cost USD**: ${total_cost:.4f}")
    L.append("")
    L.append("## Memory-gap evidence")
    L.append("")
    L.append(
        "Baseline (no memory) hypothesis: session B has no access to session A's stated "
        "preference, so the observable named in each scenario should be ABSENT in B. "
        "A 'no'/'partial' verdict is positive evidence of the gap; 'yes' is a weak signal."
    )
    L.append("")
    if memory_gap_rows:
        L.append("Scenarios where session B missed the preference:")
        L.append("")
        for sid, verdict in memory_gap_rows:
            scen = next((s for s in viz_scenarios if s["id"] == sid), None)
            hyp = scen.get("memory_hypothesis", "") if scen else ""
            obsv = scen.get("observable", "") if scen else ""
            L.append(f"- **{sid}** (verdict={verdict})")
            L.append(f"    - observable expected if memory worked: {obsv}")
            L.append(f"    - hypothesis: {hyp}")
    else:
        L.append("(No clear gaps detected.)")

    L.append("")
    L.append("## Repro commands")
    L.append("")
    L.append("```bash")
    L.append("# Full from-scratch re-run of all 5 viz pairs:")
    L.append(
        "cd /Users/wang/LLM_memory && set -a && source .env && set +a && "
        ".venv/bin/python eval/run_viz.py 2>&1 | tee eval/results/viz/run.log"
    )
    L.append("")
    L.append("# Resume only viz_03/04/05 (what produced this file):")
    L.append(
        "cd /Users/wang/LLM_memory && set -a && source .env && set +a && "
        ".venv/bin/python eval/run_viz_resume.py 2>&1 | tee -a eval/results/viz/run.log"
    )
    L.append("```")
    L.append("")

    SUMMARY_MD.write_text("\n".join(L))
    print(f"[viz-resume] wrote {SUMMARY_MD}")
    print(
        f"[viz-resume] FINAL: {passed}/{len(pairs)} PASS  "
        f"tokens={total_tokens}  cost=${total_cost:.4f}"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
