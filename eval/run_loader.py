"""Driver: run all loader scenario pairs through the harness.

Usage (from repo root, with .env sourced and venv active):
    .venv/bin/python eval/run_loader.py

Writes:
    eval/results/loader/<scenario_id>__session_<a|b>.{json,md}   (harness)
    eval/results/loader/runs.jsonl                                (one line per pair)
    eval/results/loader/SUMMARY.md                                (one-page summary)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Make sibling modules importable regardless of CWD.
REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from agent_factories import invoke_dispatcher, make_loader_factory  # noqa: E402
from harness import run_scenario_pair  # noqa: E402

RESULT_DIR = REPO_ROOT / "eval" / "results" / "loader"
SCENARIOS_PATH = REPO_ROOT / "eval" / "scenarios.json"
RUNS_JSONL = RESULT_DIR / "runs.jsonl"
SUMMARY_MD = RESULT_DIR / "SUMMARY.md"
TIMEOUT_S = 120


# -----------------------------------------------------------------------------
# Heuristic: did session B's response reference session A's content?
# -----------------------------------------------------------------------------
def _b_referenced_a(scenario: dict, b_rec: dict) -> tuple[bool, str]:
    """Return (referenced, reason).

    We look at B's `response` (ai_message) and tool_call names for the
    observable signal described in the scenario. If neither the A-side fact
    nor the observable shows up, we return (False, "<expected baseline>").
    """
    sid = scenario["id"]
    resp = (b_rec.get("response") or "").lower()
    tool_names = " ".join(b_rec.get("tool_call_names") or []).lower()
    haystack = f"{resp}\n{tool_names}"

    if sid == "loader_01_snake_case":
        # Observable: snake_case columns (customer_id, monthly_charges) present
        # in B's preview rather than original casing.
        snake = ("customer_id" in haystack) or ("monthly_charges" in haystack)
        camel = ("customerid" in haystack and "customer_id" not in haystack) or (
            "monthlycharges" in haystack
        )
        if snake and not camel:
            return True, "preview shows snake_case columns"
        return False, "preview retains original casing (customerID/MonthlyCharges)"

    if sid == "loader_02_no_phone_service_null":
        # Observable: MultipleLines missing count ~682 (not 0); B treating
        # 'No phone service' as NaN.
        mentions_no_phone = "no phone service" in haystack
        # Look for a >0 missing count near "multiplelines" — crude but
        # sufficient for a baseline signal.
        m = re.search(r"multiplelines[^0-9]{0,60}(\d{1,5})", haystack)
        elevated = bool(m and int(m.group(1)) > 0)
        if mentions_no_phone or elevated:
            return True, "treated 'No phone service' as null"
        return False, "MultipleLines reported as 0 missing (no null-coercion rule)"

    if sid == "loader_03_drop_pii":
        # Observable: column list does NOT contain customerID.
        has_customer_id = "customerid" in haystack
        if not has_customer_id:
            return True, "customerID absent from column list"
        return False, "customerID still present in column list"

    if sid == "loader_04_tenure_as_months":
        # Observable: preview contains a 'tenure_years' column.
        if "tenure_years" in haystack:
            return True, "derived tenure_years column present"
        return False, "no tenure_years column derived"

    if sid == "loader_05_total_charges_numeric":
        # Observable: TotalCharges dtype reported as float64 (not object/string).
        m = re.search(r"totalcharges[^a-z0-9]{0,40}(float\d*|object|string)", haystack)
        if m:
            dtype = m.group(1)
            if dtype.startswith("float"):
                return True, f"TotalCharges dtype={dtype}"
            return False, f"TotalCharges dtype={dtype} (default, no coercion)"
        return False, "TotalCharges dtype not clearly float (default behavior)"

    return False, "no scenario-specific check matched"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    scenarios_doc = json.loads(SCENARIOS_PATH.read_text())
    loader_scenarios = [
        s for s in scenarios_doc["scenarios"] if s.get("agent") == "loader"
    ]
    print(f"loader scenarios: {len(loader_scenarios)}")

    factory = make_loader_factory()
    invoke_fn = invoke_dispatcher("loader")

    # Reset runs.jsonl for this batch — we append per-scenario below.
    RUNS_JSONL.write_text("")

    rows: list[dict] = []
    total_tokens = 0
    total_cost = 0.0
    pass_count = 0
    fail_ids: list[tuple[str, str]] = []  # (id, error)

    for scenario in loader_scenarios:
        sid = scenario["id"]
        try:
            pair = run_scenario_pair(
                scenario,
                factory,
                RESULT_DIR,
                timeout_s=TIMEOUT_S,
                invoke_fn=invoke_fn,
            )
        except BaseException as e:  # harness itself blew up — record and move on
            err = f"{type(e).__name__}: {e}"
            print(f"{sid} | HARNESS ERROR | {err}")
            fail_ids.append((sid, err))
            with RUNS_JSONL.open("a") as f:
                f.write(json.dumps({"scenario_id": sid, "ok": False, "error": err}) + "\n")
            continue

        a = pair["a"]
        b = pair["b"]
        ok = pair["ok"]
        toks = (a.get("total_tokens") or 0) + (b.get("total_tokens") or 0)
        cost = (a.get("cost_usd") or 0.0) + (b.get("cost_usd") or 0.0)
        total_tokens += toks
        total_cost += cost
        if ok:
            pass_count += 1
        else:
            fail_ids.append((sid, pair.get("error") or "unknown"))

        ref, ref_reason = _b_referenced_a(scenario, b)

        notes = ref_reason
        print(
            f"{sid} | ok={ok} | tokens={a.get('total_tokens', 0)}+{b.get('total_tokens', 0)}"
            f" | cost=${a.get('cost_usd', 0.0):.4f}+${b.get('cost_usd', 0.0):.4f}"
            f" | B_refs_A={ref} | {notes}"
        )

        # Persist this row to runs.jsonl IMMEDIATELY so a crash later doesn't
        # lose data.
        row = {
            "scenario_id": sid,
            "agent": pair["agent"],
            "ok": ok,
            "error": pair.get("error"),
            "a_total_tokens": a.get("total_tokens", 0),
            "b_total_tokens": b.get("total_tokens", 0),
            "a_cost_usd": a.get("cost_usd", 0.0),
            "b_cost_usd": b.get("cost_usd", 0.0),
            "a_latency_ms": a.get("latency_ms"),
            "b_latency_ms": b.get("latency_ms"),
            "a_tool_call_names": a.get("tool_call_names", []),
            "b_tool_call_names": b.get("tool_call_names", []),
            "b_referenced_a": ref,
            "b_referenced_a_reason": ref_reason,
        }
        rows.append(row)
        with RUNS_JSONL.open("a") as f:
            f.write(json.dumps(row) + "\n")

    # ---- Totals ----
    print("---")
    print(f"TOTAL tokens: {total_tokens}")
    print(f"TOTAL cost  : ${total_cost:.4f}")
    print(f"PASS: {pass_count}/{len(loader_scenarios)}  FAIL: {len(fail_ids)}")
    if fail_ids:
        for sid, err in fail_ids:
            print(f"  FAIL {sid}: {err}")

    # ---- SUMMARY.md ----
    if len(fail_ids) == 0 and pass_count == len(loader_scenarios):
        status = "PASS (all 10 invocations completed without harness error)"
    elif pass_count > 0:
        listed = ", ".join(s for s, _ in fail_ids)
        status = f"PARTIAL ({len(fail_ids)}/{len(loader_scenarios)} failed: {listed})"
    else:
        status = "FAIL"

    md = []
    md.append("# Loader baseline results — SUMMARY")
    md.append("")
    md.append(f"**Status:** {status}")
    md.append("")
    md.append("## Per-scenario table")
    md.append("")
    md.append(
        "| id | tokens (a+b) | cost (a+b) USD | A latency_ms | B latency_ms | B referenced A? | observation |"
    )
    md.append("|---|---:|---:|---:|---:|:--:|---|")
    for row in rows:
        toks = row["a_total_tokens"] + row["b_total_tokens"]
        cost = row["a_cost_usd"] + row["b_cost_usd"]
        md.append(
            f"| {row['scenario_id']} "
            f"| {row['a_total_tokens']}+{row['b_total_tokens']}={toks} "
            f"| ${row['a_cost_usd']:.4f}+${row['b_cost_usd']:.4f}=${cost:.4f} "
            f"| {row['a_latency_ms']} "
            f"| {row['b_latency_ms']} "
            f"| {'yes' if row['b_referenced_a'] else 'no'} "
            f"| {row['b_referenced_a_reason']} |"
        )
    md.append("")
    md.append("## Totals")
    md.append("")
    md.append(f"- Total tokens: **{total_tokens}**")
    md.append(f"- Total cost: **${total_cost:.4f}**")
    md.append(f"- Pass/fail: **{pass_count}/{len(loader_scenarios)}** passed")
    md.append("")
    md.append("## Memory-gap notes (baseline shows where memory would help)")
    md.append("")
    md.append(
        "Each scenario's session A states a preference / fact; session B is run on a"
        " fresh agent with no memory bridge, so by design B cannot honor A. The rows"
        " marked **B referenced A? = no** are the EXPECTED baseline state — these are"
        " the ones that should flip to **yes** once a memory layer is wired in."
    )
    md.append("")
    gap_rows = [r for r in rows if not r["b_referenced_a"]]
    if gap_rows:
        md.append("Scenarios that surfaced a clear memory-would-have-helped gap:")
        for r in gap_rows:
            md.append(f"- `{r['scenario_id']}` — {r['b_referenced_a_reason']}")
    coincident = [r for r in rows if r["b_referenced_a"]]
    if coincident:
        md.append("")
        md.append(
            "Scenarios where B coincidentally honored A (weakens comparison signal —"
            " investigate before relying on these for the post-memory delta):"
        )
        for r in coincident:
            md.append(f"- `{r['scenario_id']}` — {r['b_referenced_a_reason']}")
    md.append("")
    md.append("## Repro")
    md.append("")
    md.append("```bash")
    md.append(
        "cd /Users/wang/LLM_memory && set -a && source .env && set +a && "
        ".venv/bin/python eval/run_loader.py 2>&1 | tee eval/results/loader/run.log"
    )
    md.append("```")
    md.append("")

    SUMMARY_MD.write_text("\n".join(md))
    print(f"wrote {SUMMARY_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
