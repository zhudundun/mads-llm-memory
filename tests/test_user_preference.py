"""Pytest suite for the user_preference memory module.

All tests use an in-memory SQLite store (`:memory:`) for zero filesystem
touch. The store-level tests exercise `PreferenceStore` directly; the
tool-level tests exercise the LangChain `@tool` callables returned by
`make_preference_tools`.
"""

from __future__ import annotations

import re
from datetime import datetime

import pytest

from mads.memory.preference_tools import make_preference_tools
from mads.memory.user_preference import PreferenceStore


# --------------------------------------------------------------------- fixture


@pytest.fixture
def store():
    """Fresh in-memory PreferenceStore per test."""
    s = PreferenceStore(db_path=":memory:")
    try:
        yield s
    finally:
        s.close()


# ----------------------------------------------------------- helper assertions


_UUID_HEX_RE = re.compile(r"^[0-9a-f]{32}$")


def _assert_iso8601(value: str) -> None:
    """Assert `value` is a non-empty ISO-8601 string parseable by fromisoformat."""
    assert isinstance(value, str) and value, f"expected non-empty str, got {value!r}"
    # datetime.fromisoformat handles "2026-06-01T18:42:07+00:00" form.
    datetime.fromisoformat(value)


def _tools_by_name(tools):
    return {t.name: t for t in tools}


# ============================================================ STORE-LEVEL TESTS


def test_set_returns_uuid_and_inserts_row(store):
    """1. `set` returns a UUID-like id and inserts a row visible to `get`."""
    new_id = store.set("u1", "chart.library", "plotly")
    assert isinstance(new_id, str)
    assert _UUID_HEX_RE.match(new_id), f"expected 32-char uuid hex, got {new_id!r}"

    row = store.get("u1", "chart.library")
    assert row is not None
    assert row["id"] == new_id
    assert row["user_id"] == "u1"
    assert row["preference_key"] == "chart.library"
    assert row["preference_value"] == "plotly"
    assert row["valid_to"] is None


def test_get_unknown_key_returns_none(store):
    """2. `get` on an unknown key returns `None`."""
    assert store.get("u1", "nope") is None
    # Even after writing a different key, missing key still None
    store.set("u1", "other.key", "v")
    assert store.get("u1", "nope") is None


def test_list_active_empty_then_ordered(store):
    """3. `list_active` returns [] for a new user; populated rows ordered by key ASC."""
    assert store.list_active("u1") == []

    # Insert in non-alphabetical order to verify the ORDER BY.
    store.set("u1", "z_last", "v_z")
    store.set("u1", "m_middle", "v_m")
    store.set("u1", "a_first", "v_a")

    rows = store.list_active("u1")
    assert [r["preference_key"] for r in rows] == ["a_first", "m_middle", "z_last"]


def test_versioning_supersedes_and_preserves_history(store):
    """4. Set "A" then "B": get->"B", list_active len 1, history len 2 ordered asc,
    older row valid_to non-null, newer row valid_to null."""
    id_a = store.set("u1", "theme", "A")
    id_b = store.set("u1", "theme", "B")
    assert id_a != id_b

    assert store.get("u1", "theme")["preference_value"] == "B"

    active = store.list_active("u1")
    assert len(active) == 1
    assert active[0]["preference_value"] == "B"

    history = store.list_history("u1", "theme")
    assert len(history) == 2
    # Ordered by valid_from ASC -> older row (A) is first.
    assert history[0]["preference_value"] == "A"
    assert history[1]["preference_value"] == "B"
    # valid_from monotonic non-decreasing.
    assert history[0]["valid_from"] <= history[1]["valid_from"]

    # Older row is closed; newer row is open.
    assert history[0]["valid_to"] is not None
    _assert_iso8601(history[0]["valid_to"])
    assert history[1]["valid_to"] is None


def test_forget_true_then_false(store):
    """5. `forget` returns True for an active row (and closes it), False if no active row."""
    store.set("u1", "k", "v")
    assert store.forget("u1", "k") is True

    # No longer in list_active.
    assert store.get("u1", "k") is None
    assert store.list_active("u1") == []

    # Still in list_history, with valid_to set.
    history = store.list_history("u1", "k")
    assert len(history) == 1
    assert history[0]["valid_to"] is not None
    _assert_iso8601(history[0]["valid_to"])

    # Second forget returns False (no active row to close).
    assert store.forget("u1", "k") is False

    # Forget of a never-set key also False.
    assert store.forget("u1", "never_set") is False


def test_confidence_validation(store):
    """6. confidence out of [0,1] raises ValueError; 0.5 and None are accepted."""
    with pytest.raises(ValueError):
        store.set("u1", "k", "v", confidence=1.5)
    with pytest.raises(ValueError):
        store.set("u1", "k", "v", confidence=-0.1)

    # 0.5 succeeds
    store.set("u1", "k_mid", "v", confidence=0.5)
    row_mid = store.get("u1", "k_mid")
    assert row_mid is not None and row_mid["confidence"] == 0.5

    # None succeeds (and is stored as None)
    store.set("u1", "k_none", "v", confidence=None)
    row_none = store.get("u1", "k_none")
    assert row_none is not None and row_none["confidence"] is None

    # Boundary values 0.0 and 1.0 should also succeed.
    store.set("u1", "k_zero", "v", confidence=0.0)
    store.set("u1", "k_one", "v", confidence=1.0)
    assert store.get("u1", "k_zero")["confidence"] == 0.0
    assert store.get("u1", "k_one")["confidence"] == 1.0


def test_user_scoping(store):
    """7. Prefs set under `u1` are invisible to `u2` via get/list_active/list_history."""
    store.set("u1", "shared_key", "u1_value")
    store.set("u2", "shared_key", "u2_value")
    store.set("u1", "u1_only", "x")

    # get
    assert store.get("u1", "shared_key")["preference_value"] == "u1_value"
    assert store.get("u2", "shared_key")["preference_value"] == "u2_value"
    assert store.get("u2", "u1_only") is None

    # list_active
    u1_keys = [r["preference_key"] for r in store.list_active("u1")]
    u2_keys = [r["preference_key"] for r in store.list_active("u2")]
    assert "u1_only" in u1_keys
    assert "u1_only" not in u2_keys
    assert sorted(u2_keys) == ["shared_key"]

    # list_history: u2 only sees its own row
    u1_history = store.list_history("u1", "shared_key")
    u2_history = store.list_history("u2", "shared_key")
    assert len(u1_history) == 1 and u1_history[0]["preference_value"] == "u1_value"
    assert len(u2_history) == 1 and u2_history[0]["preference_value"] == "u2_value"

    # u2 has no history for u1_only
    assert store.list_history("u2", "u1_only") == []


def test_timestamps_iso8601_and_valid_to_none(store):
    """8. valid_from, created_at, updated_at are non-empty ISO-8601; valid_to is None."""
    store.set("u1", "k", "v")
    row = store.get("u1", "k")
    assert row is not None
    _assert_iso8601(row["valid_from"])
    _assert_iso8601(row["created_at"])
    _assert_iso8601(row["updated_at"])
    assert row["valid_to"] is None


# ============================================================= TOOL-LEVEL TESTS


def test_factory_returns_four_tools_with_expected_names(store):
    """9. Factory returns exactly 4 tools; sorted names match the contract."""
    tools = make_preference_tools(store, "u1")
    assert len(tools) == 4
    names = sorted(t.name for t in tools)
    assert names == [
        "forget_preference",
        "list_preferences",
        "recall_preference",
        "remember_preference",
    ]


def test_remember_preference_tool_returns_descriptive_string(store):
    """10. remember_preference.invoke returns a string mentioning key and value."""
    tools = _tools_by_name(make_preference_tools(store, "u1"))
    out = tools["remember_preference"].invoke(
        {"key": "chart.library", "value": "plotly"}
    )
    assert isinstance(out, str)
    assert "chart.library" in out
    assert "plotly" in out

    # And it actually persisted into the store.
    assert store.get("u1", "chart.library")["preference_value"] == "plotly"


def test_recall_preference_missing_key_message(store):
    """11. recall_preference for an unknown key returns a 'not set' message."""
    tools = _tools_by_name(make_preference_tools(store, "u1"))
    out = tools["recall_preference"].invoke({"key": "missing.key"})
    assert isinstance(out, str)
    # Match the contract's "no preference set" semantics without locking to
    # exact punctuation -- substrings cover the current message.
    lowered = out.lower()
    assert "missing.key" in out
    assert "no preference" in lowered or "not set" in lowered


def test_list_preferences_empty_then_populated(store):
    """12. list_preferences: empty -> 'No preferences' message; populated -> both keys."""
    tools = _tools_by_name(make_preference_tools(store, "u1"))

    empty_out = tools["list_preferences"].invoke({})
    assert isinstance(empty_out, str)
    assert "No preferences" in empty_out

    tools["remember_preference"].invoke({"key": "chart.library", "value": "plotly"})
    tools["remember_preference"].invoke({"key": "color.scheme", "value": "dark"})

    populated = tools["list_preferences"].invoke({})
    assert "chart.library" in populated
    assert "plotly" in populated
    assert "color.scheme" in populated
    assert "dark" in populated


def test_forget_preference_tool_confirms_then_reports_no_active(store):
    """13. forget_preference after remember returns confirmation; second call -> 'no active'."""
    tools = _tools_by_name(make_preference_tools(store, "u1"))
    tools["remember_preference"].invoke({"key": "chart.library", "value": "plotly"})

    first = tools["forget_preference"].invoke({"key": "chart.library"})
    assert isinstance(first, str)
    assert "chart.library" in first
    # Confirmation -> "Forgot ..." in current implementation; allow any
    # affirmative phrasing.
    assert "forgot" in first.lower() or "forget" in first.lower()

    second = tools["forget_preference"].invoke({"key": "chart.library"})
    assert isinstance(second, str)
    assert "no active preference" in second.lower()


# ============================================================ AXIS-MODEL (v1.2)


def test_global_pref_default_has_null_project_id(store):
    """15. Default ``set`` (no project_id) yields a global pref (project_id IS NULL)."""
    store.set("u1", "k", "v")
    row = store.get("u1", "k")
    assert row is not None
    assert row["project_id"] is None


def test_project_scoped_pref_is_stored(store):
    """16. Setting with project_id stores the project on the row."""
    store.set("u1", "k", "v", project_id="proj1")
    row = store.get("u1", "k", project_id="proj1")
    assert row is not None
    assert row["project_id"] == "proj1"
    assert row["preference_value"] == "v"


def test_global_and_project_scoped_coexist(store):
    """17. Both a global pref and a project-scoped pref for same (user, key) can coexist."""
    store.set("u1", "k", "global_val")
    store.set("u1", "k", "project_val", project_id="proj1")

    # Both rows are active.
    glob = store.get("u1", "k")  # no project_id → global only
    assert glob["preference_value"] == "global_val"
    assert glob["project_id"] is None

    scoped = store.get("u1", "k", project_id="proj1")
    assert scoped["preference_value"] == "project_val"
    assert scoped["project_id"] == "proj1"


def test_get_returns_only_global_when_project_id_none(store):
    """18. ``get(project_id=None)`` returns ONLY the global row, even if a
    project-scoped variant exists."""
    store.set("u1", "k", "global_val")
    store.set("u1", "k", "project_val", project_id="proj1")
    row = store.get("u1", "k")
    assert row["preference_value"] == "global_val"


def test_get_returns_project_when_both_exist(store):
    """19. ``get(project_id=X)`` returns the project-scoped row when both
    global and project-specific exist (project takes precedence)."""
    store.set("u1", "k", "global_val")
    store.set("u1", "k", "project_val", project_id="proj1")
    row = store.get("u1", "k", project_id="proj1")
    assert row["preference_value"] == "project_val"


def test_get_falls_back_to_global_when_no_project_scoped(store):
    """20. ``get(project_id=X)`` falls back to the global row when no
    project-scoped one exists."""
    store.set("u1", "k", "global_val")
    row = store.get("u1", "k", project_id="proj1")
    assert row["preference_value"] == "global_val"
    assert row["project_id"] is None


def test_list_active_global_only_default(store):
    """21. ``list_active(project_id=None)`` returns ONLY global prefs."""
    store.set("u1", "g1", "v")
    store.set("u1", "g2", "v")
    store.set("u1", "p1_only", "v", project_id="proj1")
    rows = store.list_active("u1")
    keys = [r["preference_key"] for r in rows]
    assert keys == ["g1", "g2"]


def test_list_active_with_project_returns_both(store):
    """22. ``list_active(project_id=X)`` returns BOTH global and project-scoped
    prefs for that project."""
    store.set("u1", "g1", "v_glob")
    store.set("u1", "p1_only", "v_proj", project_id="proj1")
    store.set("u1", "other_proj_only", "v", project_id="proj2")
    rows = store.list_active("u1", project_id="proj1")
    keys = sorted(r["preference_key"] for r in rows)
    # global "g1" AND project-1 "p1_only" but NOT "other_proj_only"
    assert keys == ["g1", "p1_only"]


def test_list_active_project_specific_overrides_global_in_listing(store):
    """23 (v1.3). When both global and project-scoped exist for same key,
    ``list_active`` returns ONLY the project-scoped row (most-specific
    wins; effective view collapses to one row per key)."""
    store.set("u1", "k", "global_val")
    store.set("u1", "k", "project_val", project_id="proj1")
    rows = store.list_active("u1", project_id="proj1")
    values = [(r["preference_key"], r["preference_value"]) for r in rows]
    # Only the project-scoped row surfaces (effective view).
    assert ("k", "project_val") in values
    assert ("k", "global_val") not in values
    # One row per key.
    assert len([k for k, _ in values if k == "k"]) == 1


def test_forget_global_does_not_touch_project_scoped(store):
    """24. ``forget(project_id=None)`` only closes the global row, not project-scoped."""
    store.set("u1", "k", "global_val")
    store.set("u1", "k", "project_val", project_id="proj1")
    assert store.forget("u1", "k") is True

    # Global is gone.
    assert store.get("u1", "k") is None
    # Project-scoped still alive.
    row = store.get("u1", "k", project_id="proj1")
    assert row is not None
    assert row["preference_value"] == "project_val"


def test_forget_project_scoped_does_not_touch_global(store):
    """25. ``forget(project_id=X)`` only closes the project-scoped row."""
    store.set("u1", "k", "global_val")
    store.set("u1", "k", "project_val", project_id="proj1")
    assert store.forget("u1", "k", project_id="proj1") is True

    # Global still alive.
    assert store.get("u1", "k")["preference_value"] == "global_val"
    # Project-scoped is gone (get with project_id falls back to global).
    row = store.get("u1", "k", project_id="proj1")
    assert row["preference_value"] == "global_val"


def test_versioning_only_affects_same_scope(store):
    """26. Versioning is per-(user, project, key) tuple — a project-scoped
    update doesn't close the global row, and vice versa."""
    store.set("u1", "k", "global_v1")
    store.set("u1", "k", "project_v1", project_id="proj1")
    # Now version the project-scoped row.
    store.set("u1", "k", "project_v2", project_id="proj1")
    # Global pref untouched.
    glob = store.get("u1", "k")
    assert glob["preference_value"] == "global_v1"
    # Project-scoped is v2.
    scoped = store.get("u1", "k", project_id="proj1")
    assert scoped["preference_value"] == "project_v2"


def test_list_history_respects_project(store):
    """27. ``list_history`` returns only the chain for the requested
    (user, project, key) tuple."""
    store.set("u1", "k", "g_v1")
    store.set("u1", "k", "g_v2")
    store.set("u1", "k", "p_v1", project_id="proj1")

    g_history = store.list_history("u1", "k")  # global
    assert [r["preference_value"] for r in g_history] == ["g_v1", "g_v2"]

    p_history = store.list_history("u1", "k", project_id="proj1")
    assert [r["preference_value"] for r in p_history] == ["p_v1"]


# ============================================================ AXIS-MODEL TOOLS


def test_remember_preference_default_is_global(store):
    """28. Without ``project_scoped=True``, ``remember_preference`` writes a global row."""
    tools = _tools_by_name(make_preference_tools(store, "u1", project_id="proj1"))
    tools["remember_preference"].invoke(
        {"key": "chart.library", "value": "plotly"}
    )
    row = store.get("u1", "chart.library")  # global lookup
    assert row is not None
    assert row["project_id"] is None
    assert row["preference_value"] == "plotly"


def test_remember_preference_project_scoped(store):
    """29. With ``project_scoped=True``, the row carries the current project_id."""
    tools = _tools_by_name(make_preference_tools(store, "u1", project_id="proj1"))
    tools["remember_preference"].invoke(
        {
            "key": "column_rename_customerID",
            "value": "rename to customer_id on load",
            "project_scoped": True,
        }
    )
    # Global lookup misses.
    assert store.get("u1", "column_rename_customerID") is None
    # Project lookup hits.
    scoped = store.get(
        "u1", "column_rename_customerID", project_id="proj1"
    )
    assert scoped is not None
    assert scoped["project_id"] == "proj1"


def test_remember_preference_default_does_not_bleed_across_projects(store):
    """30. A project-scoped pref written under proj1 is NOT visible to a
    factory bound to proj2 (the cross_01 fix)."""
    f_proj1 = _tools_by_name(make_preference_tools(store, "u1", project_id="proj1"))
    f_proj2 = _tools_by_name(make_preference_tools(store, "u1", project_id="proj2"))

    f_proj1["remember_preference"].invoke(
        {
            "key": "column_rename_customerID",
            "value": "rename to customer_id",
            "project_scoped": True,
        }
    )

    # proj1 list shows it.
    p1_list = f_proj1["list_preferences"].invoke({})
    assert "column_rename_customerID" in p1_list

    # proj2 list does NOT.
    p2_list = f_proj2["list_preferences"].invoke({})
    assert "column_rename_customerID" not in p2_list


def test_list_preferences_includes_both_global_and_project(store):
    """31. ``list_preferences`` (with bound project) shows both global and
    project-scoped prefs for that project."""
    tools = _tools_by_name(make_preference_tools(store, "u1", project_id="proj1"))
    tools["remember_preference"].invoke(
        {"key": "chart.library", "value": "plotly"}  # global
    )
    tools["remember_preference"].invoke(
        {
            "key": "rename_rule",
            "value": "customerID → customer_id",
            "project_scoped": True,
        }
    )
    out = tools["list_preferences"].invoke({})
    assert "chart.library" in out
    assert "plotly" in out
    assert "rename_rule" in out
    assert "customerID" in out


def test_recall_preference_returns_project_scoped_first(store):
    """32. ``recall_preference`` returns the project-scoped value when one
    exists for the bound project, else the global value."""
    tools = _tools_by_name(make_preference_tools(store, "u1", project_id="proj1"))
    tools["remember_preference"].invoke(
        {"key": "k", "value": "global_val"}
    )
    out_global = tools["recall_preference"].invoke({"key": "k"})
    assert "global_val" in out_global

    tools["remember_preference"].invoke(
        {"key": "k", "value": "project_val", "project_scoped": True}
    )
    out_proj = tools["recall_preference"].invoke({"key": "k"})
    assert "project_val" in out_proj


def test_forget_preference_project_scope_flag(store):
    """33. ``forget_preference`` accepts ``project_scoped`` to control which
    row it closes."""
    tools = _tools_by_name(make_preference_tools(store, "u1", project_id="proj1"))
    tools["remember_preference"].invoke({"key": "k", "value": "global_v"})
    tools["remember_preference"].invoke(
        {"key": "k", "value": "project_v", "project_scoped": True}
    )

    # Forget only the project-scoped one.
    out = tools["forget_preference"].invoke(
        {"key": "k", "project_scoped": True}
    )
    assert "Forgot" in out

    # Global pref survives.
    assert store.get("u1", "k")["preference_value"] == "global_v"
    # Project-scoped is closed → recall falls back to global.
    rec = tools["recall_preference"].invoke({"key": "k"})
    assert "global_v" in rec


def test_preference_tools_args_schema_hides_project_id(store):
    """34. The tool args schema MUST NOT contain ``project_id`` even with the
    project-scoping feature: only the LLM-visible ``project_scoped: bool``."""
    tools = make_preference_tools(store, "u1", project_id="proj1")
    for t in tools:
        schema = t.args_schema
        fields = set(getattr(schema, "model_fields", {}).keys())
        assert "user_id" not in fields, (
            f"{t.name} leaks user_id: {fields}"
        )
        assert "project_id" not in fields, (
            f"{t.name} leaks project_id: {fields}"
        )


def test_factory_without_project_id_works_globally(store):
    """35. Backward-compat: ``make_preference_tools(store, user_id)`` still
    works without ``project_id``. project_scoped=True writes a global row
    (effectively a no-op for scope)."""
    tools = _tools_by_name(make_preference_tools(store, "u1"))
    tools["remember_preference"].invoke(
        {"key": "k", "value": "v", "project_scoped": True}
    )
    row = store.get("u1", "k")  # global lookup
    assert row is not None
    assert row["project_id"] is None


# ============================================================ ORIGINAL TOOL TEST


# ============================================================ v1.3 DATASET AXIS (STORE)


def test_dataset_scoped_pref_is_stored(store):
    """37 (v1.3). Setting with dataset_fingerprint stores the fingerprint on the row."""
    store.set("u1", "k", "v", dataset_fingerprint="fp_churn")
    row = store.get("u1", "k", dataset_fingerprint="fp_churn")
    assert row is not None
    assert row["dataset_fingerprint"] == "fp_churn"
    assert row["preference_value"] == "v"
    # The same key with no fingerprint should miss (no global row exists).
    assert store.get("u1", "k") is None


def test_dataset_scoped_pref_does_not_bleed_across_datasets(store):
    """38 (v1.3). A dataset-scoped pref under fp_a is NOT visible under fp_b."""
    store.set("u1", "rule", "v_a", dataset_fingerprint="fp_a")
    store.set("u1", "rule", "v_b", dataset_fingerprint="fp_b")
    a = store.get("u1", "rule", dataset_fingerprint="fp_a")
    b = store.get("u1", "rule", dataset_fingerprint="fp_b")
    assert a["preference_value"] == "v_a"
    assert b["preference_value"] == "v_b"


def test_get_most_specific_wins_dataset_over_project_over_global(store):
    """39 (v1.3). With all three rows present, ``get`` returns the
    project+dataset row first, then dataset, then project, then global."""
    store.set("u1", "k", "global_v")
    store.set("u1", "k", "project_v", project_id="proj1")
    store.set("u1", "k", "dataset_v", dataset_fingerprint="fp1")
    store.set(
        "u1", "k", "project_dataset_v",
        project_id="proj1", dataset_fingerprint="fp1",
    )

    # All four scopes can co-exist for the same (user, key).
    assert store.get("u1", "k")["preference_value"] == "global_v"
    assert (
        store.get("u1", "k", project_id="proj1")["preference_value"]
        == "project_v"
    )
    assert (
        store.get("u1", "k", dataset_fingerprint="fp1")["preference_value"]
        == "dataset_v"
    )
    assert (
        store.get(
            "u1", "k", project_id="proj1", dataset_fingerprint="fp1"
        )["preference_value"]
        == "project_dataset_v"
    )


def test_get_falls_back_to_less_specific_when_more_specific_absent(store):
    """40 (v1.3). dataset+project lookup falls back to dataset, then project,
    then global if more-specific rows don't exist."""
    store.set("u1", "k", "global_v")
    # Only global exists — both axes set in lookup should still find global.
    assert (
        store.get(
            "u1", "k", project_id="proj1", dataset_fingerprint="fp1"
        )["preference_value"]
        == "global_v"
    )

    # Add a project-scoped row → that wins over global.
    store.set("u1", "k", "project_v", project_id="proj1")
    assert (
        store.get(
            "u1", "k", project_id="proj1", dataset_fingerprint="fp1"
        )["preference_value"]
        == "project_v"
    )

    # Add a dataset-scoped row → dataset wins over project for the
    # same-rank tie? No — dataset and project are both rank-1 in absolute
    # terms but dataset specificity (rank=2 in our scheme) > project
    # (rank=1). Confirm:
    store.set("u1", "k", "dataset_v", dataset_fingerprint="fp1")
    assert (
        store.get(
            "u1", "k", project_id="proj1", dataset_fingerprint="fp1"
        )["preference_value"]
        == "dataset_v"
    )


def test_list_active_collapses_to_most_specific_per_key(store):
    """41 (v1.3). ``list_active`` returns one row per key — the most specific."""
    store.set("u1", "k", "global_v")
    store.set("u1", "k", "project_v", project_id="proj1")
    store.set("u1", "k", "dataset_v", dataset_fingerprint="fp1")

    rows = store.list_active(
        "u1", project_id="proj1", dataset_fingerprint="fp1"
    )
    assert len(rows) == 1
    assert rows[0]["preference_value"] == "dataset_v"


def test_list_active_global_only_when_both_scopes_none(store):
    """42 (v1.3). list_active with no project/dataset returns ONLY rows where
    BOTH project_id and dataset_fingerprint are NULL (= true global)."""
    store.set("u1", "g_k", "g_v")  # global
    store.set("u1", "p_k", "p_v", project_id="proj1")  # project only
    store.set("u1", "d_k", "d_v", dataset_fingerprint="fp1")  # dataset only
    rows = store.list_active("u1")
    keys = sorted(r["preference_key"] for r in rows)
    assert keys == ["g_k"]


def test_forget_dataset_scoped_does_not_touch_global_or_project(store):
    """43 (v1.3). forget(dataset_fingerprint=X) only closes the dataset row."""
    store.set("u1", "k", "global_v")
    store.set("u1", "k", "project_v", project_id="proj1")
    store.set("u1", "k", "dataset_v", dataset_fingerprint="fp1")

    assert store.forget("u1", "k", dataset_fingerprint="fp1") is True

    # Dataset row gone — get(dataset_fp=fp1) falls back to global (rank-0)
    # because project_id is None on the lookup.
    row = store.get("u1", "k", dataset_fingerprint="fp1")
    assert row["preference_value"] == "global_v"
    # Project survives.
    assert (
        store.get("u1", "k", project_id="proj1")["preference_value"]
        == "project_v"
    )
    # Global survives.
    assert store.get("u1", "k")["preference_value"] == "global_v"


def test_versioning_only_affects_same_dataset_scope(store):
    """44 (v1.3). Versioning is per-full-identity-tuple — updating the
    dataset-scoped row doesn't close the global or project-scoped row."""
    store.set("u1", "k", "g_v1")
    store.set("u1", "k", "p_v1", project_id="proj1")
    store.set("u1", "k", "d_v1", dataset_fingerprint="fp1")
    # Version the dataset row.
    store.set("u1", "k", "d_v2", dataset_fingerprint="fp1")
    assert store.get("u1", "k")["preference_value"] == "g_v1"
    assert (
        store.get("u1", "k", project_id="proj1")["preference_value"]
        == "p_v1"
    )
    assert (
        store.get("u1", "k", dataset_fingerprint="fp1")["preference_value"]
        == "d_v2"
    )


# ============================================================ v1.3 TOOL BEHAVIOR MATRIX


def _make_tools_with_fp(store, *, project_id, get_fp):
    """Helper: build tools with a callable fingerprint accessor."""
    return _tools_by_name(
        make_preference_tools(
            store, "u1", project_id=project_id, get_dataset_fingerprint=get_fp
        )
    )


def test_matrix_row_1_no_flags_any_df(store):
    """45 (v1.3) Behavior matrix R1: no flags → global. Stored row has both
    project_id and dataset_fingerprint NULL; return says 'global'."""
    tools = _make_tools_with_fp(store, project_id="proj1", get_fp=lambda: "fp1")
    out = tools["remember_preference"].invoke(
        {"key": "k", "value": "v"}
    )
    assert "global" in out.lower()
    row = store.get("u1", "k")
    assert row["project_id"] is None
    assert row["dataset_fingerprint"] is None


def test_matrix_row_2_project_only_any_df(store):
    """46 (v1.3) R2: project_scoped=True, dataset_scoped=False. Stored
    row has project_id=<bound>, dataset_fingerprint=NULL."""
    tools = _make_tools_with_fp(store, project_id="proj1", get_fp=lambda: "fp1")
    out = tools["remember_preference"].invoke(
        {"key": "k", "value": "v", "project_scoped": True}
    )
    assert "project" in out.lower()
    row = store.get("u1", "k", project_id="proj1")
    assert row["project_id"] == "proj1"
    assert row["dataset_fingerprint"] is None


def test_matrix_row_3_dataset_only_df_bound(store):
    """47 (v1.3) R3: dataset_scoped=True, project_scoped=False, df bound.
    Stored row has project_id=NULL, dataset_fingerprint=<bound>."""
    tools = _make_tools_with_fp(store, project_id="proj1", get_fp=lambda: "fp1")
    out = tools["remember_preference"].invoke(
        {"key": "k", "value": "v", "dataset_scoped": True}
    )
    assert "dataset" in out.lower()
    row = store.get("u1", "k", dataset_fingerprint="fp1")
    assert row["project_id"] is None
    assert row["dataset_fingerprint"] == "fp1"


def test_matrix_row_4_project_and_dataset_df_bound(store):
    """48 (v1.3) R4: both flags True, df bound. Stored row has both axes
    populated; return says 'project+dataset'."""
    tools = _make_tools_with_fp(store, project_id="proj1", get_fp=lambda: "fp1")
    out = tools["remember_preference"].invoke(
        {
            "key": "k",
            "value": "v",
            "project_scoped": True,
            "dataset_scoped": True,
        }
    )
    assert "project+dataset" in out.lower()
    row = store.get(
        "u1", "k", project_id="proj1", dataset_fingerprint="fp1"
    )
    assert row["project_id"] == "proj1"
    assert row["dataset_fingerprint"] == "fp1"


def test_matrix_row_5_dataset_only_no_df_bound_downgrades_to_global(store):
    """49 (v1.3) R5: dataset_scoped=True, no df → downgrade to global with warning."""
    tools = _make_tools_with_fp(store, project_id="proj1", get_fp=lambda: None)
    out = tools["remember_preference"].invoke(
        {"key": "k", "value": "v", "dataset_scoped": True}
    )
    # Warning string must explicitly mention the downgrade target.
    low = out.lower()
    assert "downgraded to global" in low
    assert "no dataset was bound" in low
    row = store.get("u1", "k")
    assert row is not None
    assert row["project_id"] is None
    assert row["dataset_fingerprint"] is None


def test_matrix_row_6_project_and_dataset_no_df_downgrades_to_project(store):
    """50 (v1.3) R6: both flags True, no df → downgrade to project with warning."""
    tools = _make_tools_with_fp(store, project_id="proj1", get_fp=lambda: None)
    out = tools["remember_preference"].invoke(
        {
            "key": "k",
            "value": "v",
            "project_scoped": True,
            "dataset_scoped": True,
        }
    )
    low = out.lower()
    assert "downgraded to project" in low
    assert "no dataset was bound" in low
    row = store.get("u1", "k", project_id="proj1")
    assert row is not None
    assert row["project_id"] == "proj1"
    assert row["dataset_fingerprint"] is None


def test_remember_preference_args_schema_hides_fingerprint(store):
    """51 (v1.3) The tool args schema must NOT contain ``dataset_fingerprint``
    or ``user_id`` or ``project_id`` — only the LLM-visible flags."""
    tools = make_preference_tools(
        store, "u1",
        project_id="proj1",
        get_dataset_fingerprint=lambda: "fp1",
    )
    by_name = {t.name: t for t in tools}
    fields = set(
        getattr(by_name["remember_preference"].args_schema, "model_fields", {}).keys()
    )
    # Allowed: key, value, project_scoped, dataset_scoped, source, confidence.
    assert "dataset_fingerprint" not in fields, fields
    assert "user_id" not in fields, fields
    assert "project_id" not in fields, fields
    assert "dataset_scoped" in fields, fields
    assert "project_scoped" in fields, fields


def test_remember_preference_live_fingerprint_accessor(store):
    """52 (v1.3) The factory reads the fingerprint accessor on EVERY call,
    so a mid-conversation bind_dataframe is reflected without rebuilding tools."""
    fp_holder = {"current": None}
    tools = _make_tools_with_fp(
        store, project_id="proj1", get_fp=lambda: fp_holder["current"]
    )

    # First call: no df yet → dataset_scoped downgrades to global.
    out1 = tools["remember_preference"].invoke(
        {"key": "k1", "value": "v1", "dataset_scoped": True}
    )
    assert "downgraded to global" in out1.lower()

    # Now simulate bind_dataframe by changing what the accessor returns.
    fp_holder["current"] = "fp_xyz"

    # Same call: dataset_scoped now succeeds with the new fingerprint.
    out2 = tools["remember_preference"].invoke(
        {"key": "k2", "value": "v2", "dataset_scoped": True}
    )
    assert "downgraded" not in out2.lower()
    assert "dataset" in out2.lower()
    row = store.get("u1", "k2", dataset_fingerprint="fp_xyz")
    assert row is not None
    assert row["dataset_fingerprint"] == "fp_xyz"


def test_recall_preference_returns_most_specific_match(store):
    """53 (v1.3) recall_preference returns the dataset-scoped row when df bound
    and a dataset-scoped row exists."""
    store.set("u1", "k", "global_v")
    store.set("u1", "k", "dataset_v", dataset_fingerprint="fp1")
    tools = _make_tools_with_fp(
        store, project_id="proj1", get_fp=lambda: "fp1"
    )
    out = tools["recall_preference"].invoke({"key": "k"})
    assert "dataset_v" in out
    assert "scope: dataset" in out


def test_forget_preference_dataset_scope_flag(store):
    """54 (v1.3) forget_preference with dataset_scoped=True closes ONLY the
    dataset row."""
    store.set("u1", "k", "g_v")
    store.set("u1", "k", "d_v", dataset_fingerprint="fp1")
    tools = _make_tools_with_fp(
        store, project_id="proj1", get_fp=lambda: "fp1"
    )
    out = tools["forget_preference"].invoke(
        {"key": "k", "dataset_scoped": True}
    )
    assert "forgot" in out.lower()
    # Dataset row gone.
    assert store.get("u1", "k", dataset_fingerprint="fp1")["preference_value"] == "g_v"
    # Global survives.
    assert store.get("u1", "k")["preference_value"] == "g_v"


def test_forget_dataset_scoped_no_df_is_noop(store):
    """55 (v1.3) forget_preference with dataset_scoped=True but no df bound
    returns a 'no active' message — it does NOT fall through to forget the
    global row."""
    store.set("u1", "k", "g_v")
    tools = _make_tools_with_fp(
        store, project_id="proj1", get_fp=lambda: None
    )
    out = tools["forget_preference"].invoke(
        {"key": "k", "dataset_scoped": True}
    )
    assert "no active preference" in out.lower()
    # Global still active.
    assert store.get("u1", "k")["preference_value"] == "g_v"


# ============================================================ ORIGINAL TOOL TEST


def test_tool_factory_enforces_user_scoping_via_closure(store):
    """36. Tools built for u1 do not see prefs the store has under u2."""
    u1_tools = _tools_by_name(make_preference_tools(store, "u1"))
    u2_tools = _tools_by_name(make_preference_tools(store, "u2"))

    # Write u2-only data via the store directly.
    store.set("u2", "u2_secret", "do_not_leak")

    # u1 recall should not see it.
    recall_out = u1_tools["recall_preference"].invoke({"key": "u2_secret"})
    assert "do_not_leak" not in recall_out
    assert "no preference" in recall_out.lower() or "not set" in recall_out.lower()

    # u1 list should not include it.
    list_out = u1_tools["list_preferences"].invoke({})
    assert "u2_secret" not in list_out
    assert "do_not_leak" not in list_out

    # u1 forget should report no active preference for u2's key.
    forget_out = u1_tools["forget_preference"].invoke({"key": "u2_secret"})
    assert "no active preference" in forget_out.lower()

    # And u2's data must still be there afterwards.
    assert store.get("u2", "u2_secret")["preference_value"] == "do_not_leak"

    # And u2's own tools can see it.
    u2_recall = u2_tools["recall_preference"].invoke({"key": "u2_secret"})
    assert "do_not_leak" in u2_recall
