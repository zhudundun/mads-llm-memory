"""EDA-agent paired-session driver.

Runs the 5 `agent == "eda"` scenarios from eval/scenarios.json through the
harness against gpt-4o. Per-session JSON+MD goes to eval/results/eda/, one
JSONL line per scenario pair, and a one-page SUMMARY.md at the end.

Usage:
    cd /Users/wang/LLM_memory && set -a && source .env && set +a && \
        .venv/bin/python eval/run_eda.py 2>&1 | tee eval/results/eda/run.log
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Ensure project root is on sys.path so the `eval` package imports cleanly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.agent_factories import invoke_dispatcher, make_eda_factory  # noqa: E402
from eval.harness import run_scenario_pair  # noqa: E402


SCENARIOS_PATH = PROJECT_ROOT / "eval" / "scenarios.json"
DATA_PATH = PROJECT_ROOT / "baseline" / "data" / "churn_data.csv"
RESULT_DIR = PROJECT_ROOT / "eval" / "results" / "eda"
RUNS_JSONL = RESULT_DIR / "runs.jsonl"
SUMMARY_MD = RESULT_DIR / "SUMMARY.md"
TIMEOUT_S = 180
AGENT_NAME = "eda"


# -- B-references-A heuristic --------------------------------------------------
# Per-scenario detector: returns ("yes"|"no", short_reason).
# We look at session B's ai_message + tool_call_names for the observable
# signature described in the scenario. If neither references A's info, we
# return "no" (the expected baseline state).
def _detect_b_references_a(
    scenario_id: str, b: Dict[str, Any]
) -> tuple[str, str]:
    resp = (b.get("response") or "").lower()
    tool_names_text = " ".join(b.get("tool_call_names") or []).lower()
    haystack = resp + " \n " + tool_names_text

    if scenario_id == "eda_01_contract_typos":
        # Observable: B normalizes typos before groupby. Look for normalization signals.
        hit = any(
            k in haystack
            for k in [
                "normalize",
                "case-fold",
                "case fold",
                "casefold",
                "strip",
                ".str.lower",
                "str.lower(",
                "typo",
            ]
        )
        return ("yes", "mentions normalize/strip/casefold/typo") if hit else (
            "no",
            "raw groupby, no typo-normalization signal",
        )

    if scenario_id == "eda_02_churn_definition":
        # Observable: percentage with exactly one decimal, e.g. '26.5%'.
        import re

        one_dec_pct = re.search(r"\b\d{1,3}\.\d%", resp)
        return ("yes", f"reports one-decimal pct ({one_dec_pct.group(0)})") if one_dec_pct else (
            "no",
            "no one-decimal pct format detected",
        )

    if scenario_id == "eda_03_grouping_preference":
        # Observable: both 'contract' AND tenure-bucket grouping with edges 12/24/48.
        has_contract = "contract" in haystack
        has_buckets = (
            ("0-12" in haystack and ("13-24" in haystack or "25-48" in haystack))
            or ("tenure_bucket" in haystack)
            or ("0,12" in haystack and "24" in haystack and "48" in haystack)
        )
        return ("yes", "contract+tenure-bucket grouping present") if (
            has_contract and has_buckets
        ) else (
            "no",
            "no Contract x tenure-bucket cross-cut with edges 12/24/48",
        )

    if scenario_id == "eda_04_exec_summary_first":
        # Observable: 3-bullet exec summary first, then separate technical-details section.
        has_exec = (
            "executive summary" in resp
            or "exec summary" in resp
            or ("summary" in resp[:200] and resp[:400].count("\n- ") >= 3)
        )
        has_tech = (
            "technical details" in resp
            or "technical-details" in resp
            or "dtypes" in resp
        )
        return ("yes", "exec-first + technical-details headers present") if (
            has_exec and has_tech
        ) else (
            "no",
            "no exec-summary-first / technical-details template",
        )

    if scenario_id == "eda_05_followup_finding":
        # Observable: mention fiber/DSL/InternetService as prior finding.
        hit = any(
            k in resp
            for k in [
                "fiber",
                "fibre",
                "dsl",
                "internetservice",
                "internet service",
            ]
        )
        return ("yes", "references fiber/DSL/InternetService") if hit else (
            "no",
            "generic next-steps, no fiber/DSL reference",
        )

    # Default fallthrough — unknown scenario.
    return ("no", "no detector configured")


# -- Driver --------------------------------------------------------------------
def main() -> int:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[eda-runner] loading scenarios from {SCENARIOS_PATH}")
    spec = json.loads(SCENARIOS_PATH.read_text())
    scenarios: List[Dict[str, Any]] = [
        s for s in spec["scenarios"] if s.get("agent") == AGENT_NAME
    ]
    print(f"[eda-runner] {len(scenarios)} eda scenarios queued")

    print(f"[eda-runner] loading dataframe from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"[eda-runner] df shape: {df.shape}")

    factory = make_eda_factory(df)
    invoke_fn = invoke_dispatcher(AGENT_NAME, df=df)

    # Reset runs.jsonl for a clean run.
    if RUNS_JSONL.exists():
        RUNS_JSONL.unlink()

    pair_rows: List[Dict[str, Any]] = []
    t_start = time.monotonic()

    for i, scenario in enumerate(scenarios, start=1):
        sid = scenario["id"]
        print(
            f"\n[eda-runner] ({i}/{len(scenarios)}) running pair: {sid}",
            flush=True,
        )
        try:
            pair = run_scenario_pair(
                scenario,
                factory,
                RESULT_DIR,
                timeout_s=TIMEOUT_S,
                invoke_fn=invoke_fn,
            )
        except BaseException as e:  # noqa: BLE001 — capture+continue
            print(f"[eda-runner] PAIR ERROR ({sid}): {type(e).__name__}: {e}")
            pair = {
                "scenario_id": sid,
                "agent": AGENT_NAME,
                "a": None,
                "b": None,
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
            }

        # Append to runs.jsonl immediately.
        with RUNS_JSONL.open("a") as f:
            f.write(json.dumps(pair, default=str) + "\n")

        a = pair.get("a") or {}
        b = pair.get("b") or {}
        tokens_a = int(a.get("total_tokens") or 0)
        tokens_b = int(b.get("total_tokens") or 0)
        cost_a = float(a.get("cost_usd") or 0.0)
        cost_b = float(b.get("cost_usd") or 0.0)
        n_calls_a = int(a.get("tool_call_count") or 0)
        n_calls_b = int(b.get("tool_call_count") or 0)
        notes = pair.get("error") or "ok"

        print(
            f"{sid} | ok={pair.get('ok')} | "
            f"tokens={tokens_a + tokens_b} ({tokens_a}+{tokens_b}) | "
            f"cost=${cost_a + cost_b:.4f} (${cost_a:.4f}+${cost_b:.4f}) | "
            f"tool_calls_a={n_calls_a} | tool_calls_b={n_calls_b} | "
            f"notes={notes}",
            flush=True,
        )

        b_ref, b_ref_reason = _detect_b_references_a(sid, b)
        pair_rows.append(
            {
                "id": sid,
                "ok": bool(pair.get("ok")),
                "error": pair.get("error"),
                "tokens_a": tokens_a,
                "tokens_b": tokens_b,
                "cost_a": cost_a,
                "cost_b": cost_b,
                "latency_a": int(a.get("latency_ms") or 0),
                "latency_b": int(b.get("latency_ms") or 0),
                "n_calls_a": n_calls_a,
                "n_calls_b": n_calls_b,
                "b_references_a": b_ref,
                "b_ref_reason": b_ref_reason,
                "memory_hypothesis": scenario.get("memory_hypothesis", ""),
                "observable": scenario.get("observable", ""),
            }
        )

    elapsed_s = time.monotonic() - t_start

    # -- Totals ----------------------------------------------------------------
    total_tokens = sum(r["tokens_a"] + r["tokens_b"] for r in pair_rows)
    total_cost = sum(r["cost_a"] + r["cost_b"] for r in pair_rows)
    n_ok = sum(1 for r in pair_rows if r["ok"])
    n_total = len(pair_rows)

    if n_ok == n_total:
        status = "PASS"
    elif n_ok == 0:
        status = "FAIL"
    else:
        status = "PARTIAL"

    print(
        f"\n[eda-runner] DONE: {n_ok}/{n_total} ok in {elapsed_s:.1f}s | "
        f"total_tokens={total_tokens} | total_cost=${total_cost:.4f}"
    )

    _write_summary(pair_rows, status, n_ok, n_total, total_tokens, total_cost)
    print(f"[eda-runner] wrote {SUMMARY_MD}")
    return 0 if status == "PASS" else 1


def _write_summary(
    rows: List[Dict[str, Any]],
    status: str,
    n_ok: int,
    n_total: int,
    total_tokens: int,
    total_cost: float,
) -> None:
    lines: List[str] = []
    lines.append(f"# EDA agent — paired-session baseline run")
    lines.append("")
    lines.append(f"**Status:** {status} ({n_ok}/{n_total} scenario pairs ok)")
    lines.append("")
    lines.append(
        "Each pair runs session A (states a preference/fact) then session B "
        "(asks an indirectly related question) on TWO fresh `EDAToolsAgent` "
        "instances. Baseline expectation: B has no memory of A, so B's "
        "behavior should be independent of A — i.e. the 'B references A?' "
        "column should be **no** across the board. Any **yes** indicates "
        "the agent happened to honor the preference by chance or default."
    )
    lines.append("")

    # Table
    lines.append("## Per-scenario results")
    lines.append("")
    lines.append(
        "| id | tokens (a+b) | cost (a+b) | latency A/B (ms) | "
        "tool_calls A/B | did B reference A? | observation |"
    )
    lines.append("|---|---:|---:|---|---|---|---|")
    for r in rows:
        tokens = f"{r['tokens_a'] + r['tokens_b']} ({r['tokens_a']}+{r['tokens_b']})"
        cost = f"${r['cost_a'] + r['cost_b']:.4f} (${r['cost_a']:.4f}+${r['cost_b']:.4f})"
        latency = f"{r['latency_a']}/{r['latency_b']}"
        tcalls = f"{r['n_calls_a']}/{r['n_calls_b']}"
        b_ref = r["b_references_a"]
        obs = r["b_ref_reason"]
        if not r["ok"]:
            obs = f"ERROR: {r['error']} | {obs}"
        lines.append(
            f"| `{r['id']}` | {tokens} | {cost} | {latency} | {tcalls} | "
            f"{b_ref} | {obs} |"
        )
    lines.append("")

    # Totals
    lines.append("## Totals")
    lines.append("")
    lines.append(f"- **Total tokens:** {total_tokens:,}")
    lines.append(f"- **Total cost (USD):** ${total_cost:.4f}")
    lines.append(f"- **Pairs ok:** {n_ok}/{n_total}")
    lines.append("")

    # Memory-gap evidence notes
    lines.append("## Memory-gap evidence")
    lines.append("")
    lines.append(
        "The **observable** for each scenario describes what session B *would* "
        "do if the agent had memory of session A. With no memory wired in, B "
        "should miss every observable below. Cleanest signals (highest-contrast "
        "between baseline and 'memory on') are the scenarios where B's "
        "structure is concretely auditable from text:"
    )
    lines.append("")
    no_count = sum(1 for r in rows if r["b_references_a"] == "no" and r["ok"])
    yes_count = sum(1 for r in rows if r["b_references_a"] == "yes" and r["ok"])
    lines.append(
        f"- {no_count}/{n_ok} successful pairs show **no** evidence B used A's "
        f"info — the expected baseline state."
    )
    if yes_count:
        lines.append(
            f"- {yes_count}/{n_ok} pair(s) coincidentally honored A's preference. "
            "Treat these as weak signals — see per-row notes for the specific text match."
        )
    lines.append("")
    lines.append("Per-scenario observability hypothesis (reference):")
    for r in rows:
        lines.append(f"- `{r['id']}` — _{r['memory_hypothesis']}_")
    lines.append("")

    # Repro
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append(
        "cd /Users/wang/LLM_memory && set -a && source .env && set +a && "
        ".venv/bin/python eval/run_eda.py 2>&1 | tee eval/results/eda/run.log"
    )
    lines.append("```")
    lines.append("")

    SUMMARY_MD.write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
