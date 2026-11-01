"""End-to-end CLI demo for the user_preference memory module.

Exercises all four LangChain tools produced by `make_preference_tools`,
demonstrates versioning via `PreferenceStore.list_history`, and proves
per-user scoping. No LLM is used; tools are invoked directly.

Run from the repo root:

    cd /Users/wang/LLM_memory && \
        .venv/bin/python demo_preferences.py > demo_preferences.out 2>&1
"""

from __future__ import annotations

import os
from pathlib import Path

from mads.memory.preference_tools import make_preference_tools
from mads.memory.user_preference import PreferenceStore

DB_PATH = "/Users/wang/LLM_memory/.demo_preferences.sqlite"


def header(n: int, title: str) -> None:
    print()
    print(f"Step {n}: {title}")
    print("-" * (len(f"Step {n}: {title}")))


def main() -> None:
    # Ensure a fresh DB for every run.
    try:
        os.remove(DB_PATH)
        print(f"Removed pre-existing DB at {DB_PATH}.")
    except FileNotFoundError:
        print(f"No pre-existing DB at {DB_PATH}; starting clean.")

    # --- Step 1: Setup -----------------------------------------------------
    header(1, "Setup PreferenceStore and build alice's tools")
    store = PreferenceStore(DB_PATH)
    alice_tools = make_preference_tools(store, "alice")
    print(f"DB file exists: {Path(DB_PATH).exists()}")
    print(f"Tool count: {len(alice_tools)}")
    print("Tool names: " + ", ".join(t.name for t in alice_tools))

    remember, recall, list_prefs, forget = alice_tools

    # --- Step 2: Remember 5 preferences ------------------------------------
    header(2, "Remember 5 preferences for alice")
    print(
        remember.invoke(
            {
                "key": "report.format",
                "value": "executive summary first, then technical details",
            }
        )
    )
    print(
        remember.invoke(
            {
                "key": "chart.library",
                "value": "plotly",
                "source": "user_explicit",
                "confidence": 1.0,
            }
        )
    )
    print(remember.invoke({"key": "cohort.grain", "value": "signup_week"}))
    print(remember.invoke({"key": "sql.show_assumptions", "value": "true"}))
    print(
        remember.invoke(
            {"key": "model.metrics", "value": "precision, recall, F1, calibration"}
        )
    )

    # --- Step 3: List all preferences --------------------------------------
    header(3, "List all preferences for alice")
    print(list_prefs.invoke({}))

    # --- Step 4: Recall a specific preference ------------------------------
    header(4, "Recall chart.library")
    print(recall.invoke({"key": "chart.library"}))

    # --- Step 5: Recall a missing preference -------------------------------
    header(5, "Recall a key that was never set")
    print(recall.invoke({"key": "nonexistent.key"}))

    # --- Step 6: Supersede an existing preference --------------------------
    header(6, "Supersede chart.library with a new value")
    print(
        remember.invoke(
            {
                "key": "chart.library",
                "value": "plotly + matplotlib for static export",
            }
        )
    )

    # --- Step 7: Confirm latest value is returned --------------------------
    header(7, "Recall chart.library again — must show the NEW value")
    print(recall.invoke({"key": "chart.library"}))

    # --- Step 8: Full history of chart.library -----------------------------
    header(8, "Full history of chart.library (via store.list_history)")
    history = store.list_history("alice", "chart.library")
    print(f"History row count: {len(history)}")
    for row in history:
        print(
            f"{row['valid_from']} -> {row['valid_to']} : {row['preference_value']}"
        )

    # --- Step 9: Forget one preference -------------------------------------
    header(9, "Forget sql.show_assumptions")
    print(forget.invoke({"key": "sql.show_assumptions"}))

    # --- Step 10: Confirm forgotten ----------------------------------------
    header(10, "List preferences — sql.show_assumptions must be absent")
    listing = list_prefs.invoke({})
    print(listing)
    assert "sql.show_assumptions" not in listing, (
        "sql.show_assumptions still present after forget!"
    )
    print("(verified: sql.show_assumptions is absent)")

    # --- Step 11: Forget the same key again --------------------------------
    header(11, "Forget sql.show_assumptions again — expect 'no active' msg")
    print(forget.invoke({"key": "sql.show_assumptions"}))

    # --- Step 12: User scoping --------------------------------------------
    header(12, "Build bob's tools — list_preferences must be empty for bob")
    bob_tools = make_preference_tools(store, "bob")
    _, _, bob_list, _ = bob_tools
    print(bob_list.invoke({}))

    # --- Step 13: Done -----------------------------------------------------
    header(13, "Demo complete")
    store.close()
    print("=== DEMO COMPLETE ===")


if __name__ == "__main__":
    main()
