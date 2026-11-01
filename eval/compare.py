"""Compare baseline vs with-memory results.

Reads ``eval/results/<agent>/runs.jsonl`` (baseline) and
``eval/results/<agent>__with_memory/runs.jsonl`` (or any suffix via
``--with-memory-suffix``) and emits a diff table to
``eval/results/COMPARE.md``.

Per V1_DESIGN_BRIEF §8:

* Output ``COMPARE.md`` with one row per scenario.
* Label each row ``gap-closed`` (baseline=no, v1=yes), ``unchanged``,
  ``regressed`` (baseline=yes, v1=no), or ``still-helping`` (both yes).
* Exit 0 if at least one row is gap-closed; exit 1 if none are.

The baseline EDA runs were produced by ``eval/run_eda.py``, which writes its
verdict only to SUMMARY.md (not into runs.jsonl). When comparing against
the EDA memory-aware variant, pass ``--rescore-with-eda-rubric`` to
re-apply the tight per-scenario observables from
:mod:`eval.run_with_memory` to BOTH sides; without it the baseline rows are
treated as ``b_referenced_a=False`` (the absence of the field).

Public API (importable for tests)::

    load_runs(path)              -> dict[str, dict]
    classify(baseline, v1)       -> "gap-closed" | "unchanged" | "regressed" | "still-helping"
    compare(agent, baseline_dir, with_memory_dir) -> {"rows": [...], "n_closed": int}
    render_markdown(comparisons) -> str
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "eval" / "results"
COMPARE_MD = RESULTS_ROOT / "COMPARE.md"

AGENTS = ("loader", "eda", "viz")


# -----------------------------------------------------------------------------
# JSONL helpers
# -----------------------------------------------------------------------------
def load_runs(path: Path | str) -> dict:
    """Read a runs.jsonl file. Returns ``{scenario_id: row}``.

    Returns an empty dict if the file does not exist or is empty. Skips
    malformed lines silently — they were already logged by the runner.
    """
    p = Path(path)
    if not p.exists():
        return {}
    out: dict = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        sid = row.get("scenario_id")
        if sid:
            out[sid] = row
    return out


# -----------------------------------------------------------------------------
# Classification
# -----------------------------------------------------------------------------
def classify(baseline_row: Optional[dict], v1_row: Optional[dict]) -> str:
    """Compare baseline.b_referenced_a vs v1.b_referenced_a.

    Returns one of ``gap-closed``, ``unchanged``, ``regressed``, ``still-helping``,
    ``baseline-missing``, ``v1-missing``.
    """
    if baseline_row is None and v1_row is None:
        return "both-missing"
    if baseline_row is None:
        return "baseline-missing"
    if v1_row is None:
        return "v1-missing"
    b = bool(baseline_row.get("b_referenced_a"))
    v = bool(v1_row.get("b_referenced_a"))
    if not b and v:
        return "gap-closed"
    if b and v:
        return "still-helping"
    if b and not v:
        return "regressed"
    return "unchanged"


# -----------------------------------------------------------------------------
# Comparison
# -----------------------------------------------------------------------------
def _rescore_with_eda_rubric(row: dict, *, results_dir: Path | None = None) -> dict:
    """Apply the tight per-scenario EDA observable from
    :mod:`eval.run_with_memory` to a runs.jsonl row (in-memory). Returns a
    new dict with ``b_referenced_a`` / ``b_referenced_a_reason`` derived
    from session B's response. Non-EDA rows are returned unchanged.

    Handles BOTH runs.jsonl shapes:

    * baseline (``run_eda.py``): row has nested ``a`` / ``b`` session records
      with ``response`` inside.
    * memory-aware (``run_with_memory.py``): row is flat (no inline
      response). Session B's response is read from
      ``<results_dir>/<scenario_id>__session_b.json``.
    """
    sid = row.get("scenario_id", "")
    if not sid.startswith("eda_"):
        return row
    # Late-import to avoid a hard dependency for the loader-only path.
    import sys as _sys
    from pathlib import Path as _Path
    _eval_dir = _Path(__file__).resolve().parent
    if str(_eval_dir) not in _sys.path:
        _sys.path.insert(0, str(_eval_dir))
    import run_with_memory as rwm

    # Find session B's response. Prefer the inline ``b`` block (baseline
    # shape); fall back to reading the on-disk per-session JSON
    # (run_with_memory shape).
    b_inline = row.get("b")
    b_resp = ""
    b_dict: dict = {}
    if isinstance(b_inline, dict):
        b_resp = b_inline.get("response") or ""
        b_dict = b_inline
    elif results_dir is not None:
        session_b_path = results_dir / f"{sid}__session_b.json"
        if session_b_path.exists():
            try:
                b_dict = json.loads(session_b_path.read_text())
                b_resp = b_dict.get("response") or ""
            except Exception:  # noqa: BLE001
                b_dict = {}
                b_resp = ""

    ok, reason = rwm._eda_observable(sid, b_resp, b_dict)
    out = dict(row)
    out["b_referenced_a"] = ok
    out["b_referenced_a_reason"] = reason
    return out


def compare(
    agent: str,
    *,
    baseline_dir: Path,
    with_memory_dir: Path,
    rescore_with_eda_rubric: bool = False,
) -> dict:
    """Produce a per-scenario comparison for one agent.

    Returns ``{"agent": str, "rows": [...], "n_closed": int, "n_total": int}``
    where each row is ``{scenario_id, label, baseline_ref, v1_ref,
    baseline_reason, v1_reason}``.

    When ``rescore_with_eda_rubric`` is True, both baseline and v1 rows are
    re-scored with the tight EDA observables before comparison (no-op for
    non-EDA rows). Use this when the baseline runs.jsonl was produced by
    ``run_eda.py`` (which stores its verdict only in SUMMARY.md, not in
    runs.jsonl) and you want an apples-to-apples comparison.
    """
    baseline = load_runs(baseline_dir / "runs.jsonl")
    v1 = load_runs(with_memory_dir / "runs.jsonl")
    if rescore_with_eda_rubric:
        baseline = {
            sid: _rescore_with_eda_rubric(r, results_dir=baseline_dir)
            for sid, r in baseline.items()
        }
        v1 = {
            sid: _rescore_with_eda_rubric(r, results_dir=with_memory_dir)
            for sid, r in v1.items()
        }
    all_ids = sorted(set(baseline) | set(v1))
    rows = []
    n_closed = 0
    for sid in all_ids:
        b_row = baseline.get(sid)
        v_row = v1.get(sid)
        label = classify(b_row, v_row)
        if label == "gap-closed":
            n_closed += 1
        rows.append(
            {
                "scenario_id": sid,
                "label": label,
                "baseline_ref": bool((b_row or {}).get("b_referenced_a")),
                "v1_ref": bool((v_row or {}).get("b_referenced_a")),
                "baseline_reason": (b_row or {}).get("b_referenced_a_reason", "-"),
                "v1_reason": (v_row or {}).get("b_referenced_a_reason", "-"),
            }
        )
    return {
        "agent": agent,
        "rows": rows,
        "n_closed": n_closed,
        "n_total": len(rows),
    }


# -----------------------------------------------------------------------------
# Markdown rendering
# -----------------------------------------------------------------------------
def render_markdown(comparisons: list[dict], *, include_reasons: bool = False) -> str:
    """Render a stable markdown table for one or more agent comparisons.

    With ``include_reasons=True`` the table grows two columns showing the
    per-row reasons from both sides (handy when the rubric is per-scenario
    programmatic).
    """
    lines = ["# Baseline vs with-memory comparison", ""]
    total_closed = sum(c["n_closed"] for c in comparisons)
    total_rows = sum(c["n_total"] for c in comparisons)
    lines.append(f"**Gap-closed:** {total_closed}/{total_rows}")
    lines.append("")

    for comp in comparisons:
        lines.append(f"## {comp['agent']} ({comp['n_closed']}/{comp['n_total']} closed)")
        lines.append("")
        if include_reasons:
            lines.append(
                "| scenario_id | baseline | v1 | label | baseline_reason | v1_reason |"
            )
            lines.append("|---|:--:|:--:|---|---|---|")
            for r in comp["rows"]:
                lines.append(
                    f"| {r['scenario_id']} "
                    f"| {'yes' if r['baseline_ref'] else 'no'} "
                    f"| {'yes' if r['v1_ref'] else 'no'} "
                    f"| {r['label']} "
                    f"| {str(r['baseline_reason']).replace('|', '\\|')} "
                    f"| {str(r['v1_reason']).replace('|', '\\|')} |"
                )
        else:
            lines.append(
                "| scenario_id | baseline B_refs_A | v1 B_refs_A | label |"
            )
            lines.append("|---|:--:|:--:|---|")
            for r in comp["rows"]:
                lines.append(
                    f"| {r['scenario_id']} "
                    f"| {'yes' if r['baseline_ref'] else 'no'} "
                    f"| {'yes' if r['v1_ref'] else 'no'} "
                    f"| {r['label']} |"
                )
        lines.append("")
    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------------
# Two-signal (rubric + judge) rendering
# -----------------------------------------------------------------------------
def render_two_signal_markdown(rows: list[dict]) -> str:
    """Render a flat table with rubric label + judge winner + agree column.

    Rows are sorted so disagreements (``agree == False``) float to the top,
    then unscored rows, then agreements. ``rows`` is the list of per-scenario
    result dicts from ``runs.jsonl`` of a memory-aware run that included
    ``--judge``.
    """
    def _sort_key(r: dict):
        agree = r.get("agree")
        if agree is False:
            rank = 0  # disagreement first
        elif agree is None:
            rank = 1  # unscored
        else:
            rank = 2  # agreement
        return (rank, r.get("scenario_id", ""))

    rows_sorted = sorted(rows, key=_sort_key)
    lines = ["# Two-signal comparison (rubric + judge)", ""]
    disagree_count = sum(1 for r in rows if r.get("agree") is False)
    agree_count = sum(1 for r in rows if r.get("agree") is True)
    no_judge = sum(1 for r in rows if r.get("agree") is None)
    lines.append(
        f"**Agree: {agree_count}** · "
        f"**Disagree: {disagree_count}** · "
        f"**Unscored: {no_judge}** · "
        f"**Total: {len(rows)}**"
    )
    lines.append("")
    lines.append(
        "| scenario_id | gt_passed | rubric_reason | judge_winner | base_score | mem_score | judge_reason | agree |"
    )
    lines.append("|---|:--:|---|---|:--:|:--:|---|:--:|")
    for r in rows_sorted:
        agree_marker = (
            "DISAGREE" if r.get("agree") is False
            else ("yes" if r.get("agree") is True else "-")
        )
        lines.append(
            f"| {r.get('scenario_id', '?')} "
            f"| {'yes' if r.get('gt_passed') else 'no'} "
            f"| {str(r.get('gt_reason', '-'))[:80].replace('|', '\\|')} "
            f"| {r.get('judge_winner', '-')} "
            f"| {r.get('judge_baseline_score', '-')} "
            f"| {r.get('judge_mem_score', '-')} "
            f"| {str(r.get('judge_reason', '-'))[:80].replace('|', '\\|')} "
            f"| {agree_marker} |"
        )
    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare baseline runs.jsonl vs with_memory runs.jsonl."
    )
    parser.add_argument(
        "--agent",
        default="all",
        help="One of loader / eda / viz / all (default: all).",
    )
    parser.add_argument(
        "--baseline-root",
        default=str(RESULTS_ROOT),
        help="Directory containing the per-agent baseline subdirs.",
    )
    parser.add_argument(
        "--with-memory-suffix",
        default="__with_memory",
        help="Suffix used for the memory-augmented results subdir.",
    )
    parser.add_argument(
        "--out",
        default=str(COMPARE_MD),
        help="Output markdown path (default: eval/results/COMPARE.md).",
    )
    parser.add_argument(
        "--rescore-with-eda-rubric",
        action="store_true",
        help=(
            "Re-apply the tight EDA observables from eval/run_with_memory.py "
            "to both baseline and v1 rows before comparing. Required when "
            "comparing eval/results/eda/ (produced by run_eda.py, no "
            "b_referenced_a field in its JSONL) against an EDA "
            "memory-augmented run."
        ),
    )
    parser.add_argument(
        "--include-reasons",
        action="store_true",
        help="Include per-row baseline/v1 reasons in COMPARE.md.",
    )
    parser.add_argument(
        "--two-signal",
        action="store_true",
        help=(
            "Render a flat (rubric + judge + agree) table from "
            "<with-memory-dir>/runs.jsonl. Use this for the broad EDA eval; "
            "disagreements are sorted to the top."
        ),
    )
    args = parser.parse_args(argv)

    agents = AGENTS if args.agent == "all" else (args.agent,)
    if args.agent not in ("all", *AGENTS):
        print(f"unknown --agent: {args.agent!r}")
        return 2

    base_root = Path(args.baseline_root)

    if args.two_signal:
        # Read the with-memory runs.jsonl directly and render the
        # (rubric + judge + agree) flat table. No baseline-side join.
        all_rows = []
        for agent in agents:
            wm = base_root / f"{agent}{args.with_memory_suffix}" / "runs.jsonl"
            if not wm.exists():
                continue
            for line in wm.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                all_rows.append(row)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(render_two_signal_markdown(all_rows))
        print(f"wrote {args.out} (two-signal mode, {len(all_rows)} rows)")
        return 0

    comparisons = []
    for agent in agents:
        comp = compare(
            agent,
            baseline_dir=base_root / agent,
            with_memory_dir=base_root / f"{agent}{args.with_memory_suffix}",
            rescore_with_eda_rubric=args.rescore_with_eda_rubric,
        )
        comparisons.append(comp)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        render_markdown(comparisons, include_reasons=args.include_reasons)
    )
    print(f"wrote {args.out}")

    total_closed = sum(c["n_closed"] for c in comparisons)
    print(f"gap-closed: {total_closed}")
    return 0 if total_closed > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
