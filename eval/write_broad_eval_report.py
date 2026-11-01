"""Generate eval/results/BROAD_EVAL_RESULTS.md from the runs.jsonl files.

Reads:
  eval/results/eda__with_memory/runs.jsonl   (legacy = baseline)
  eval/results/eda__memory_aware/runs.jsonl  (memory-aware variant; has gt_passed/judge fields)

Writes:
  eval/results/BROAD_EVAL_RESULTS.md

Run after the broad eval finishes; pure Python, no LLM calls.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "eval" / "results"
LEGACY_JSONL = RESULTS_ROOT / "eda__with_memory" / "runs.jsonl"
MEMAWARE_JSONL = RESULTS_ROOT / "eda__memory_aware" / "runs.jsonl"
OUT_PATH = RESULTS_ROOT / "BROAD_EVAL_RESULTS.md"

BROAD_IDS = (
    "cons_01_churn_rate_replay",
    "cons_02_grouping_replay",
    "learn_01_metric_correction",
    "learn_02_dtype_correction",
    "eff_01_re_spec_avoidance",
    "eff_02_no_clarifying_q",
    "cross_01_no_bleed",
    "cross_02_dataset_recognition",
    "multi_01_chained_metric",
    "multi_02_filter_plus_groupby",
    "adv_01_ambiguous_term",
    "neg_01_unrelated_query",
)

CATEGORIES = {
    "cons_01_churn_rate_replay": "Consistency",
    "cons_02_grouping_replay": "Consistency",
    "learn_01_metric_correction": "Learning-from-feedback",
    "learn_02_dtype_correction": "Learning-from-feedback",
    "eff_01_re_spec_avoidance": "Efficiency",
    "eff_02_no_clarifying_q": "Efficiency",
    "cross_01_no_bleed": "Cross-dataset isolation",
    "cross_02_dataset_recognition": "Cross-dataset isolation",
    "multi_01_chained_metric": "Multi-step",
    "multi_02_filter_plus_groupby": "Multi-step",
    "adv_01_ambiguous_term": "Adversarial disambiguation",
    "neg_01_unrelated_query": "Negative",
}


def _load_jsonl(p: Path) -> dict:
    out = {}
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        sid = row.get("scenario_id")
        if sid in BROAD_IDS:
            # Last-write-wins (we appended cons_01 smoke earlier; full run row overwrites).
            out[sid] = row
    return out


def _load_response(suffix: str, sid: str) -> str:
    p = RESULTS_ROOT / f"eda__{suffix}" / f"{sid}__session_b.json"
    if not p.exists():
        return ""
    try:
        return (json.loads(p.read_text()).get("response") or "")[:1200]
    except Exception:
        return ""


def main() -> int:
    legacy = _load_jsonl(LEGACY_JSONL)
    mem = _load_jsonl(MEMAWARE_JSONL)

    n_total = len(BROAD_IDS)
    n_legacy_pass = sum(1 for sid in BROAD_IDS if (legacy.get(sid) or {}).get("gt_passed"))
    n_mem_pass = sum(1 for sid in BROAD_IDS if (mem.get(sid) or {}).get("gt_passed"))
    n_judge_mem_wins = sum(
        1 for sid in BROAD_IDS if (mem.get(sid) or {}).get("judge_winner") == "memory-aware"
    )
    n_agree = sum(1 for sid in BROAD_IDS if (mem.get(sid) or {}).get("agree") is True)
    n_disagree = sum(1 for sid in BROAD_IDS if (mem.get(sid) or {}).get("agree") is False)

    legacy_cost = sum(
        (legacy.get(sid) or {}).get("a_cost_usd", 0.0) + (legacy.get(sid) or {}).get("b_cost_usd", 0.0)
        for sid in BROAD_IDS
    )
    mem_cost = sum(
        (mem.get(sid) or {}).get("a_cost_usd", 0.0) + (mem.get(sid) or {}).get("b_cost_usd", 0.0)
        for sid in BROAD_IDS
    )
    total_cost = legacy_cost + mem_cost

    by_category = defaultdict(lambda: {"total": 0, "mem_pass": 0, "agree": 0, "disagree": 0})
    for sid in BROAD_IDS:
        c = CATEGORIES[sid]
        by_category[c]["total"] += 1
        if (mem.get(sid) or {}).get("gt_passed"):
            by_category[c]["mem_pass"] += 1
        if (mem.get(sid) or {}).get("agree") is True:
            by_category[c]["agree"] += 1
        elif (mem.get(sid) or {}).get("agree") is False:
            by_category[c]["disagree"] += 1

    lines: list[str] = []
    lines.append("# BROAD_EVAL_RESULTS — 12-scenario EDA broad eval")
    lines.append("")
    lines.append(
        "Each new scenario was scored TWO ways: (1) a programmatic "
        "ground-truth-anchored rubric in `eval/run_with_memory.py` (using "
        "constants from `eval/ground_truth.py` computed at import time from "
        "`baseline/data/churn_data.csv`), and (2) a single-shot LLM judge "
        "(`eval/judge.py`, model `claude-sonnet-4-6`). Both signals are "
        "reported side-by-side; agreement = trust, disagreement = flagged."
    )
    lines.append("")
    lines.append("## 1. Headline")
    lines.append("")
    lines.append(
        f"- Scenarios run: **{n_total}** "
        f"(legacy + memory-aware variants both ran for each)"
    )
    lines.append(
        f"- Rubric pass: legacy **{n_legacy_pass}/{n_total}**, "
        f"memory-aware **{n_mem_pass}/{n_total}**"
    )
    lines.append(
        f"- Judge winner == memory-aware: **{n_judge_mem_wins}/{n_total}**"
    )
    lines.append(
        f"- Rubric/judge agreement: **{n_agree} agree**, "
        f"**{n_disagree} disagree** "
        f"(remaining {n_total - n_agree - n_disagree} unscored)"
    )
    lines.append(
        f"- Total cost (scenario runs only, both variants): "
        f"**${total_cost:.4f}** "
        f"(legacy ${legacy_cost:.4f} + memory-aware ${mem_cost:.4f}); "
        f"judge LLM cost is on top, ~1 call per scenario."
    )
    lines.append("")

    lines.append("## 2. Per-scenario two-signal table")
    lines.append("")
    lines.append(
        "| id | category | gt(legacy) | gt(mem) | rubric reason (mem) "
        "| judge winner | base / mem score | judge reason | agree |"
    )
    lines.append("|---|---|:--:|:--:|---|---|:--:|---|:--:|")
    for sid in BROAD_IDS:
        lr = legacy.get(sid) or {}
        mr = mem.get(sid) or {}
        agree = mr.get("agree")
        agree_str = "DISAGREE" if agree is False else ("yes" if agree is True else "-")
        rubric_reason_short = str(mr.get("gt_reason", "-"))[:90].replace("|", "\\|")
        judge_reason_short = str(mr.get("judge_reason", "-"))[:90].replace("|", "\\|")
        lines.append(
            f"| `{sid}` | {CATEGORIES[sid]} "
            f"| {'y' if lr.get('gt_passed') else 'n'} "
            f"| {'y' if mr.get('gt_passed') else 'n'} "
            f"| {rubric_reason_short} "
            f"| {mr.get('judge_winner', '-')} "
            f"| {mr.get('judge_baseline_score', '-')} / {mr.get('judge_mem_score', '-')} "
            f"| {judge_reason_short} "
            f"| {agree_str} |"
        )
    lines.append("")

    lines.append("## 3. Disagreements (rubric vs judge)")
    lines.append("")
    disagree_rows = [sid for sid in BROAD_IDS if (mem.get(sid) or {}).get("agree") is False]
    if not disagree_rows:
        lines.append("_None._ All 12 scenarios have rubric+judge agreement.")
    else:
        lines.append(f"_{len(disagree_rows)} scenario(s) flagged:_")
        lines.append("")
        for sid in disagree_rows:
            mr = mem.get(sid) or {}
            lines.append(f"### `{sid}`")
            lines.append("")
            lines.append(f"- rubric ({'PASS' if mr.get('gt_passed') else 'FAIL'}): {mr.get('gt_reason', '-')}")
            lines.append(f"- judge winner: **{mr.get('judge_winner')}** "
                         f"(baseline_score={mr.get('judge_baseline_score')}, "
                         f"mem_score={mr.get('judge_mem_score')})")
            lines.append(f"- judge reason: {mr.get('judge_reason', '-')}")
            lines.append("")
            mem_resp = _load_response("memory_aware", sid)
            base_resp = _load_response("with_memory", sid)
            if base_resp:
                lines.append("<details><summary>baseline (legacy) response (first 1200 chars)</summary>\n\n```\n" + base_resp + "\n```\n\n</details>")
            if mem_resp:
                lines.append("<details><summary>memory-aware response (first 1200 chars)</summary>\n\n```\n" + mem_resp + "\n```\n\n</details>")
            lines.append("")

    lines.append("## 4. Per-criterion breakdown")
    lines.append("")
    lines.append("| criterion | scenarios | mem rubric pass | rubric/judge agree | disagree |")
    lines.append("|---|:--:|:--:|:--:|:--:|")
    for cat, c in by_category.items():
        lines.append(
            f"| {cat} | {c['total']} | {c['mem_pass']}/{c['total']} "
            f"| {c['agree']}/{c['total']} | {c['disagree']} |"
        )
    lines.append("")

    lines.append("## 5. Cross-dataset findings")
    lines.append("")
    lines.append(
        "Synthetic sales dataset is generated deterministically from "
        "`eval/synthetic_dataset.py` (`SYNTHETIC_SEED=42`): 120 rows × 7 "
        "columns (`order_id`, `product_category` ∈ {A,B,C,D}, "
        "`customer_segment` ∈ {small, medium, large}, `amount` ∈ [10, 500], "
        "`order_date` (last 90 days from 2026-06-01), `region` ∈ {N,S,E,W}, "
        "`discount_pct` ∈ [0, 25]). Materialized to a tmp CSV path; the "
        "runner substitutes `{SYNTHETIC_CSV}` in session B's prompt at load "
        "time. Computed aggregates: mean amount ≈ 248.85; product category "
        "counts A=28 / B=29 / C=29 / D=34."
    )
    lines.append("")
    for sid in ("cross_01_no_bleed", "cross_02_dataset_recognition"):
        mr = mem.get(sid) or {}
        lr = legacy.get(sid) or {}
        lines.append(
            f"- `{sid}`: legacy rubric={'PASS' if lr.get('gt_passed') else 'FAIL'} "
            f"({lr.get('gt_reason', '-')}); memory-aware "
            f"rubric={'PASS' if mr.get('gt_passed') else 'FAIL'} "
            f"({mr.get('gt_reason', '-')}); judge winner="
            f"{mr.get('judge_winner', '-')}"
        )
    lines.append("")

    lines.append("## 6. Negative scenario")
    lines.append("")
    mr = mem.get("neg_01_unrelated_query") or {}
    lr = legacy.get("neg_01_unrelated_query") or {}
    lines.append(
        f"- `neg_01_unrelated_query`: legacy "
        f"memories_injected={lr.get('memories_injected_count', '-')}, "
        f"memories_cited={lr.get('memories_cited_count', '-')}, "
        f"rubric={'PASS' if lr.get('gt_passed') else 'FAIL'}"
    )
    lines.append(
        f"- memory-aware "
        f"memories_injected={mr.get('memories_injected_count', '-')}, "
        f"memories_cited={mr.get('memories_cited_count', '-')}, "
        f"rubric={'PASS' if mr.get('gt_passed') else 'FAIL'} — "
        f"{mr.get('gt_reason', '-')}"
    )
    lines.append("")

    lines.append("## 7. Coverage caveat (verbatim)")
    lines.append("")
    lines.append(
        "> **Now measured**: Consistency, Learning-from-feedback, Efficiency, "
        "Cross-dataset isolation, Multi-step analytics, Disambiguation, "
        "Negative recall behavior."
    )
    lines.append(
        "> **Still not measured**: Long-horizon learning across many "
        "sessions, multi-turn dialogue within a single session, error "
        "recovery after an agent's own mistake, cost/latency at scale."
    )
    lines.append("")

    lines.append("## 8. Surprises and honest residuals")
    lines.append("")
    surprises = []
    # Negative-scenario sanity
    if mr.get("memories_injected_count", 0) > 1:
        surprises.append(
            f"- `neg_01`: memory-aware injected "
            f"{mr['memories_injected_count']} memories for an unrelated "
            "creative query (>1 indicates over-retrieval — flagged)."
        )
    # Disagreement audit
    if n_disagree:
        surprises.append(
            f"- {n_disagree} rubric/judge disagreements (see §3) — "
            "review whether the rubric is too strict or the judge too lenient."
        )
    # Legacy already passing differentiating scenarios
    diff_ids = (
        "learn_01_metric_correction",
        "learn_02_dtype_correction",
        "eff_01_re_spec_avoidance",
        "eff_02_no_clarifying_q",
        "multi_01_chained_metric",
        "multi_02_filter_plus_groupby",
        "adv_01_ambiguous_term",
        "cross_01_no_bleed",
    )
    leg_diff_pass = [sid for sid in diff_ids if (legacy.get(sid) or {}).get("gt_passed")]
    if leg_diff_pass:
        surprises.append(
            f"- Legacy passed {len(leg_diff_pass)} of the {len(diff_ids)} "
            f"discriminating scenarios anyway: {leg_diff_pass}. "
            "This is the reactive-wrapper baseline (it still has memory tools "
            "available even if it doesn't proactively retrieve). True ADS-baseline "
            "would likely fail more of these."
        )
    if not surprises:
        surprises.append("- _No surprising observations on this run._")
    for s in surprises:
        lines.append(s)
    lines.append("")

    lines.append("## 9. Cost breakdown")
    lines.append("")
    lines.append("| id | legacy cost ($) | memory-aware cost ($) | total ($) |")
    lines.append("|---|---:|---:|---:|")
    for sid in BROAD_IDS:
        lc = (legacy.get(sid) or {}).get("a_cost_usd", 0.0) + (legacy.get(sid) or {}).get("b_cost_usd", 0.0)
        mc = (mem.get(sid) or {}).get("a_cost_usd", 0.0) + (mem.get(sid) or {}).get("b_cost_usd", 0.0)
        lines.append(f"| `{sid}` | {lc:.4f} | {mc:.4f} | {lc + mc:.4f} |")
    lines.append(f"| **TOTAL** | **{legacy_cost:.4f}** | **{mem_cost:.4f}** | **{total_cost:.4f}** |")
    lines.append("")
    lines.append(
        f"Mean per-scenario (both variants): "
        f"**{(total_cost / n_total) * 100:.2f}¢**. "
        f"Judge LLM cost (~12 calls, claude-sonnet-4-6) is on top — typically "
        "~1¢ per pair, since each call has only the two B-responses as "
        "context."
    )
    lines.append("")

    OUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
