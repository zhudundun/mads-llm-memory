"""End-to-end CLI demo for the ``data_quality_issue`` memory module.

Exercises every tool produced by :func:`make_data_quality_factory` against
an in-memory SQLite store, tells the four-issue story called out in the
implementation brief (refund_status temporal lag, session_duration iOS
reliability, campaign_id semantic null, users row-filter requirement),
demonstrates in-place updates vs. versioning vs. resolution, and verifies
project-level isolation.

Run from the repo root::

    cd /Users/wang/LLM_memory && \\
        .venv/bin/python eval/demo_data_quality_memory.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make sure we can import the in-tree package no matter where the script is run from.
REPO_ROOT = Path("/Users/wang/LLM_memory")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mads.memory.data_quality import DataQualityStore  # noqa: E402
from mads.memory.data_quality_tools import make_data_quality_factory  # noqa: E402


def header(n: int, title: str) -> None:
    print()
    line = f">>> Step {n}: {title}"
    print(line)
    print("-" * len(line))


def main() -> int:
    # In-memory SQLite -> zero filesystem touch, idempotent.
    store = DataQualityStore(db_path=":memory:")

    # -------------------------------------------------------------------------
    header(1, "Setup factory for alice on project=growth")
    factory = make_data_quality_factory(store, "alice", "growth")
    by_name = {t.name: t for t in factory.tools()}
    print(
        "Tool names ("
        + str(len(by_name))
        + "): "
        + ", ".join(sorted(by_name.keys()))
    )
    add = by_name["add_data_quality_issue"]
    get_col = by_name["get_data_quality_issues_for_column"]
    list_tbl = by_name["list_data_quality_issues_for_table"]
    search = by_name["search_data_quality_issues"]
    history = by_name["get_data_quality_issue_history"]
    resolve = by_name["resolve_data_quality_issue"]

    # -------------------------------------------------------------------------
    header(2, "Issue 1: refund_status lags 24h (payments.refund_status, temporal_lag)")
    print(
        add.invoke(
            {
                "table_name": "payments",
                "column_name": "refund_status",
                "issue_type": "temporal_lag",
                "issue_description": (
                    "refund_status is populated by an upstream batch job that "
                    "lags the transactional source by ~24 hours."
                ),
                "recommended_handling": (
                    "Exclude rows where event_ts > now() - INTERVAL '24 hours' "
                    "when computing same-day refund rates."
                ),
                "severity": "warning",
                "confidence": 0.9,
            }
        )
    )

    # -------------------------------------------------------------------------
    header(
        3,
        "Issue 2: session_duration unreliable on iOS 4.2 "
        "(sessions.session_duration, reliability_conditional)",
    )
    print(
        add.invoke(
            {
                "table_name": "sessions",
                "column_name": "session_duration",
                "issue_type": "reliability_conditional",
                "issue_description": (
                    "session_duration is unreliable on iOS 4.2 due to a known "
                    "telemetry bug — background time is double-counted."
                ),
                "recommended_handling": (
                    "Filter os_version != '4.2' for any analysis touching "
                    "session_duration."
                ),
                "severity": "warning",
            }
        )
    )

    # -------------------------------------------------------------------------
    header(
        4,
        "Issue 3: campaign_id null for organic (events.campaign_id, semantic_null)",
    )
    print(
        add.invoke(
            {
                "table_name": "events",
                "column_name": "campaign_id",
                "issue_type": "semantic_null",
                "issue_description": (
                    "campaign_id is NULL for organic / direct traffic — NULL "
                    "is meaningful (= 'no campaign'), not missing data."
                ),
                "recommended_handling": (
                    "COALESCE(campaign_id, 'organic') in funnels; do NOT drop "
                    "NULLs."
                ),
                "severity": "info",
            }
        )
    )

    # -------------------------------------------------------------------------
    header(
        5,
        "Issue 4: test users must be excluded (users, row_filter_required, TABLE-LEVEL)",
    )
    print(
        add.invoke(
            {
                "table_name": "users",
                "issue_type": "row_filter_required",
                "issue_description": (
                    "Internal test users contaminate all top-level user "
                    "metrics; they must be filtered out from every cohort."
                ),
                "recommended_handling": "WHERE is_test_user = FALSE",
                "severity": "critical",
            }
        )
    )

    # -------------------------------------------------------------------------
    header(6, "list_data_quality_issues_for_table('users') — table-level row-filter")
    print(list_tbl.invoke({"table_name": "users"}))

    # -------------------------------------------------------------------------
    header(
        7,
        "get_data_quality_issues_for_column('sessions','session_duration') — iOS issue",
    )
    print(
        get_col.invoke(
            {"table_name": "sessions", "column_name": "session_duration"}
        )
    )

    # -------------------------------------------------------------------------
    header(8, "search_data_quality_issues('ios') — should hit issue 2")
    print(search.invoke({"query": "ios"}))

    # -------------------------------------------------------------------------
    header(9, "Bump issue 2's severity warning -> critical (updated_in_place)")
    print(
        add.invoke(
            {
                "table_name": "sessions",
                "column_name": "session_duration",
                "issue_type": "reliability_conditional",
                "issue_description": (
                    "session_duration is unreliable on iOS 4.2 due to a known "
                    "telemetry bug — background time is double-counted."
                ),
                "recommended_handling": (
                    "Filter os_version != '4.2' for any analysis touching "
                    "session_duration."
                ),
                "severity": "critical",
            }
        )
    )

    # -------------------------------------------------------------------------
    header(10, "Revise issue 1's recommended_handling (versioned)")
    print(
        add.invoke(
            {
                "table_name": "payments",
                "column_name": "refund_status",
                "issue_type": "temporal_lag",
                "issue_description": (
                    "refund_status is populated by an upstream batch job that "
                    "lags the transactional source by ~24 hours."
                ),
                "recommended_handling": (
                    "Prefer the new refunds_realtime table; if you must use "
                    "payments.refund_status, exclude rows where event_ts > "
                    "now() - INTERVAL '24 hours'."
                ),
                "severity": "warning",
                "confidence": 0.9,
            }
        )
    )

    # -------------------------------------------------------------------------
    header(
        11,
        "get_data_quality_issue_history('payments','temporal_lag','refund_status') — 2 rows",
    )
    print(
        history.invoke(
            {
                "table_name": "payments",
                "column_name": "refund_status",
                "issue_type": "temporal_lag",
            }
        )
    )

    # -------------------------------------------------------------------------
    header(
        12,
        "resolve_data_quality_issue('payments','temporal_lag','refund_status') — upstream fixed",
    )
    print(
        resolve.invoke(
            {
                "table_name": "payments",
                "column_name": "refund_status",
                "issue_type": "temporal_lag",
            }
        )
    )

    # -------------------------------------------------------------------------
    header(13, "list_data_quality_issues_for_table('payments') — now empty")
    print(list_tbl.invoke({"table_name": "payments"}))

    # -------------------------------------------------------------------------
    header(14, "History after resolve — still shows both historic rows")
    print(
        history.invoke(
            {
                "table_name": "payments",
                "column_name": "refund_status",
                "issue_type": "temporal_lag",
            }
        )
    )

    # -------------------------------------------------------------------------
    header(15, "Switch to project=marketing — list_data_quality_issues_for_table('users') empty")
    other_factory = make_data_quality_factory(store, "alice", "marketing")
    other_tools = {t.name: t for t in other_factory.tools()}
    print(
        other_tools["list_data_quality_issues_for_table"].invoke(
            {"table_name": "users"}
        )
    )

    # -------------------------------------------------------------------------
    print()
    print("=== DEMO COMPLETE ===")
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
