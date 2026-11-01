"""Self-contained CLI demo for the Dataset/Schema memory module.

Walks through every behavior an LLM would touch via ``make_dataset_tools``:
registration, recall, supersession, history, and user-scoping. Uses the
cached churn dataset so the trace looks like a real session.

Run:
    cd /Users/wang/LLM_memory && \\
        .venv/bin/python demo_dataset_schema.py > demo_dataset_schema.out 2>&1
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

import pandas as pd

# Make sure we can import the in-tree package no matter where the script is run from.
REPO_ROOT = Path("/Users/wang/LLM_memory")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mads.memory import (  # noqa: E402
    DatasetSchemaStore,
    compute_fingerprint,
    make_dataset_tools,
)

DATASET_PATH = REPO_ROOT / "baseline" / "data" / "churn_data.csv"
DATASET_URL = (
    "https://raw.githubusercontent.com/business-science/"
    "ai-data-science-team/master/data/churn_data.csv"
)
DB_PATH = REPO_ROOT / ".demo_dataset_schema.sqlite"


def header(n: int, title: str) -> None:
    print()
    print(f"Step {n}: {title}")
    print("-" * (8 + len(title) + len(str(n))))


def ensure_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(DATASET_URL) as resp:
            DATASET_PATH.write_bytes(resp.read())
    return pd.read_csv(DATASET_PATH)


def main() -> None:
    # Wipe any prior DB so the run is idempotent.
    if DB_PATH.exists():
        DB_PATH.unlink()

    # -----------------------------------------------------------------
    header(1, "Setup")
    df = ensure_dataset()
    print(f"Loaded churn_data.csv: shape={df.shape}")

    store = DatasetSchemaStore(str(DB_PATH))
    alice_tools = make_dataset_tools(
        store,
        df,
        dataset_name="telecom_churn_2026Q1",
        table_name="customers",
        user_id="alice",
        project_id="growth",
    )
    tool_names = [t.name for t in alice_tools]
    print(f"Tool names ({len(tool_names)}): {tool_names}")

    fp = compute_fingerprint(df)
    print(f"Fingerprint (first 16): {fp[:16]}")

    # Convenient handles by name.
    by_name = {t.name: t for t in alice_tools}
    register_dataset = by_name["register_dataset"]
    recall_dataset = by_name["recall_dataset"]
    list_datasets = by_name["list_datasets"]
    register_column = by_name["register_column"]
    recall_column = by_name["recall_column"]
    list_columns = by_name["list_columns"]

    # -----------------------------------------------------------------
    header(2, "Fingerprint stability check")
    fp_base = compute_fingerprint(df)
    fp_col_shuffle = compute_fingerprint(df[list(reversed(list(df.columns)))])
    fp_row_shuffle = compute_fingerprint(
        df.sample(frac=1, random_state=42).reset_index(drop=True)
    )
    print(f"base          : {fp_base[:16]}")
    print(f"col-reversed  : {fp_col_shuffle[:16]}")
    print(f"row-shuffled  : {fp_row_shuffle[:16]}")
    assert fp_base == fp_col_shuffle == fp_row_shuffle, (
        "fingerprint should be invariant to column and row order"
    )
    print("fingerprint stable across column-shuffle and row-shuffle")

    # -----------------------------------------------------------------
    header(3, "Register dataset with grain")
    out = register_dataset.invoke({"grain": "one row per customer"})
    print(out)

    # -----------------------------------------------------------------
    header(4, "Recall")
    print(recall_dataset.invoke({}))

    # -----------------------------------------------------------------
    header(5, "Add primary key + join keys (merge semantics)")
    out = register_dataset.invoke(
        {
            "primary_key": ["customerID"],
            "join_keys": [
                {
                    "table": "orders",
                    "left": ["customerID"],
                    "right": ["customer_id"],
                }
            ],
        }
    )
    print(out)

    # -----------------------------------------------------------------
    header(6, "Recall after merge")
    recall_after_merge = recall_dataset.invoke({})
    print(recall_after_merge)
    assert "one row per customer" in recall_after_merge, (
        "merged record should still carry the original grain"
    )
    assert "customerID" in recall_after_merge, (
        "merged record should now include primary_key"
    )

    # -----------------------------------------------------------------
    header(7, "List all datasets for alice")
    print(list_datasets.invoke({}))

    # -----------------------------------------------------------------
    header(8, "Register 5 columns")
    column_specs = [
        {
            "column_name": "customerID",
            "semantic_type": "id",
            "business_meaning": "opaque per-customer identifier",
            "pii_level": "medium",
            "null_behavior": "never null",
        },
        {
            "column_name": "Churn",
            "semantic_type": "boolean_yn",
            "business_meaning": "customer cancelled service this month",
            "null_behavior": "never null",
        },
        {
            "column_name": "tenure",
            "semantic_type": "duration_months",
            "business_meaning": "months as a customer",
        },
        {
            "column_name": "MonthlyCharges",
            "semantic_type": "currency_usd",
            "business_meaning": "current monthly bill",
            "example_values": [29.85, 56.95, 53.85],
        },
        {
            "column_name": "TotalCharges",
            "semantic_type": "currency_usd_string",
            "business_meaning": "lifetime charges; parsed as string due to blanks",
            "null_behavior": "blank string for new customers",
        },
    ]
    for spec in column_specs:
        print(register_column.invoke(spec))

    # -----------------------------------------------------------------
    header(9, "Recall a specific column (MonthlyCharges)")
    print(recall_column.invoke({"column_name": "MonthlyCharges"}))

    # -----------------------------------------------------------------
    header(10, "List all columns")
    print(list_columns.invoke({}))

    # -----------------------------------------------------------------
    header(11, "Supersede a column (versioning)")
    out = register_column.invoke(
        {"column_name": "MonthlyCharges", "pii_level": "low"}
    )
    print(out)
    recall_after_supersede = recall_column.invoke({"column_name": "MonthlyCharges"})
    print(recall_after_supersede)
    assert "current monthly bill" in recall_after_supersede, (
        "merged column should still carry the original business_meaning"
    )
    assert "pii_level: low" in recall_after_supersede, (
        "merged column should now report pii_level=low"
    )

    # -----------------------------------------------------------------
    header(12, "Show column history (MonthlyCharges)")
    history = store.list_column_history(
        fingerprint=fp, table_name="customers", column_name="MonthlyCharges"
    )
    print(f"history rows: {len(history)}")
    for row in history:
        rendered = {
            "semantic_type": row.get("semantic_type"),
            "business_meaning": row.get("business_meaning"),
            "pii_level": row.get("pii_level"),
            "example_values": row.get("example_values"),
            "null_behavior": row.get("null_behavior"),
        }
        print(f"  {row.get('valid_from')} -> {row.get('valid_to')} : {rendered}")
    assert len(history) == 2, f"expected 2 history rows, got {len(history)}"
    assert history[0]["valid_to"] is not None, "older row should be closed"
    assert history[-1]["valid_to"] is None, "newest row should still be active"

    # -----------------------------------------------------------------
    header(13, "Invalid column name (error path)")
    out = register_column.invoke(
        {"column_name": "not_a_real_column", "semantic_type": "id"}
    )
    print(out)
    assert isinstance(out, str) and out.lower().startswith("error"), (
        f"expected an error string, got: {out!r}"
    )

    # -----------------------------------------------------------------
    header(14, "Invalid pii_level (error path)")
    out = register_column.invoke(
        {"column_name": "Churn", "pii_level": "EXTREME"}
    )
    print(out)
    assert isinstance(out, str) and out.lower().startswith("error"), (
        f"expected an error string, got: {out!r}"
    )

    # -----------------------------------------------------------------
    header(15, "User scoping")
    bob_tools = make_dataset_tools(
        store,
        df,
        dataset_name="telecom_churn_2026Q1",
        table_name="customers",
        user_id="bob",
        project_id="growth",
    )
    bob_by_name = {t.name: t for t in bob_tools}

    bob_list = bob_by_name["list_datasets"].invoke({})
    print("bob's list_datasets (before bob registers anything):")
    print(bob_list)
    assert bob_list == "No datasets registered yet.", (
        f"bob should see no datasets at first, got: {bob_list!r}"
    )

    bob_register = bob_by_name["register_dataset"].invoke(
        {"grain": "one row per customer (bob's copy)"}
    )
    print("bob registers his own dataset:")
    print(bob_register)

    alice_list_again = list_datasets.invoke({})
    print("alice's list_datasets after bob registered:")
    print(alice_list_again)
    assert "bob" not in alice_list_again.lower(), (
        "alice's list_datasets must not leak bob's content"
    )
    # alice still has exactly her one dataset.
    assert "telecom_churn_2026Q1" in alice_list_again
    assert alice_list_again.count("- ") == 1, (
        "alice should still see exactly one dataset entry"
    )

    # -----------------------------------------------------------------
    header(16, "Column visibility is global per design")
    bob_cols = bob_by_name["list_columns"].invoke({})
    print("bob's list_columns (same fingerprint -> sees alice's column facts):")
    print(bob_cols)
    assert "MonthlyCharges" in bob_cols, (
        "by design column_memory has no user_id, so bob sees the same columns"
    )
    print(
        "Confirmed: column_memory has no user_id column, so column facts "
        "are shared across users for the same dataset fingerprint."
    )

    # -----------------------------------------------------------------
    header(17, "Done")
    store.close()
    print("=== DEMO COMPLETE ===")


if __name__ == "__main__":
    main()
