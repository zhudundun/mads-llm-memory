"""End-to-end CLI demo for the ``metric_definition`` memory module.

Exercises every tool produced by :func:`make_metric_factory` against an
in-memory SQLite store, tells the four-metric story called out in the
implementation brief (Active user, Conversion, Revenue, Retention),
demonstrates in-place updates vs. versioning, prints history, and
verifies project-level isolation.

Run from the repo root::

    cd /Users/wang/LLM_memory && \\
        .venv/bin/python eval/demo_metric_memory.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make sure we can import the in-tree package no matter where the script is run from.
REPO_ROOT = Path("/Users/wang/LLM_memory")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mads.memory.metric_definition import MetricDefinitionStore  # noqa: E402
from mads.memory.metric_tools import make_metric_factory  # noqa: E402


def header(n: int, title: str) -> None:
    print()
    print(f"Step {n}: {title}")
    print("-" * (len(f"Step {n}: {title}")))


def main() -> int:
    # In-memory SQLite -> zero filesystem touch, idempotent.
    store = MetricDefinitionStore(db_path=":memory:")

    # -------------------------------------------------------------------------
    header(1, "Setup factory for alice on project=growth")
    factory = make_metric_factory(store, "alice", "growth")
    by_name = {t.name: t for t in factory.tools()}
    print(
        "Tool names ("
        + str(len(by_name))
        + "): "
        + ", ".join(sorted(by_name.keys()))
    )
    add = by_name["add_metric_definition"]
    get = by_name["get_metric_definition"]
    list_ = by_name["list_metric_definitions"]
    search = by_name["search_metric_definitions"]
    history = by_name["get_metric_history"]

    # -------------------------------------------------------------------------
    header(2, "Define metric: Active user (business + sql)")
    print(
        add.invoke(
            {
                "metric_name": "Active user",
                "business_definition": (
                    "A distinct user with at least one session in the last 28 days."
                ),
                "sql_definition": (
                    "SELECT COUNT(DISTINCT user_id) FROM sessions "
                    "WHERE ts >= now() - INTERVAL '28 days'"
                ),
                "confidence": 0.7,
                "source": "analytics_team",
            }
        )
    )

    # -------------------------------------------------------------------------
    header(3, "Define metric: Conversion (business + sql + default_filters)")
    print(
        add.invoke(
            {
                "metric_name": "Conversion",
                "business_definition": (
                    "Orders divided by sessions in the same window."
                ),
                "sql_definition": (
                    "SELECT SUM(is_order)::float / COUNT(*) FROM events"
                ),
                "default_filters": "country='US' AND tier IN ('paid','trial')",
                "source": "analytics_team",
            }
        )
    )

    # -------------------------------------------------------------------------
    header(4, "Define metric: Revenue (business + python)")
    print(
        add.invoke(
            {
                "metric_name": "Revenue",
                "business_definition": (
                    "Sum of net USD billed across all completed orders."
                ),
                "python_definition": (
                    "df.query('status == \"completed\"').net_usd.sum()"
                ),
                "source": "finance_team",
            }
        )
    )

    # -------------------------------------------------------------------------
    header(5, "Define metric: Retention (business + sql)")
    print(
        add.invoke(
            {
                "metric_name": "Retention",
                "business_definition": (
                    "Share of week-N actives that are still active in week N+1."
                ),
                "sql_definition": (
                    "WITH wk AS (...) SELECT week_n_plus_1_actives / week_n_actives "
                    "FROM wk"
                ),
                "source": "growth_team",
            }
        )
    )

    # -------------------------------------------------------------------------
    header(6, "list_metric_definitions — all 4 metrics")
    print(list_.invoke({}))

    # -------------------------------------------------------------------------
    header(7, "search_metric_definitions('session') — should hit Active user")
    print(search.invoke({"query": "session"}))

    # -------------------------------------------------------------------------
    header(8, "Bump Active user confidence 0.7 -> 0.95 (updated_in_place)")
    print(
        add.invoke(
            {
                "metric_name": "Active user",
                "business_definition": (
                    "A distinct user with at least one session in the last 28 days."
                ),
                "sql_definition": (
                    "SELECT COUNT(DISTINCT user_id) FROM sessions "
                    "WHERE ts >= now() - INTERVAL '28 days'"
                ),
                "confidence": 0.95,
                "source": "analytics_team",
            }
        )
    )
    print(get.invoke({"metric_name": "Active user"}))

    # -------------------------------------------------------------------------
    header(9, "Revise Conversion business_definition (exclude bots) -> versioned")
    print(
        add.invoke(
            {
                "metric_name": "Conversion",
                "business_definition": (
                    "Orders divided by sessions in the same window, "
                    "excluding bot sessions."
                ),
                "sql_definition": (
                    "SELECT SUM(is_order)::float / COUNT(*) FROM events"
                ),
                "default_filters": "country='US' AND tier IN ('paid','trial')",
                "source": "analytics_team",
            }
        )
    )

    # -------------------------------------------------------------------------
    header(10, "get_metric_history('Conversion') — should show 2 versions")
    print(history.invoke({"metric_name": "Conversion"}))

    # -------------------------------------------------------------------------
    header(11, "Switch to project=marketing — list_metric_definitions is empty")
    other_factory = make_metric_factory(store, "alice", "marketing")
    other_tools = {t.name: t for t in other_factory.tools()}
    print(other_tools["list_metric_definitions"].invoke({}))

    # -------------------------------------------------------------------------
    header(12, "Done")
    store.close()
    print("=== DEMO COMPLETE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
