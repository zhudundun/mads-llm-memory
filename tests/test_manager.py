"""Tests for :class:`mads.manager.MemoryManager`.

The manager owns one shared SQLite connection over the three in-scope stores
(metric, dataset, episode) and binds ``(user_id, project_id)`` into every
factory closure. These tests confirm tenancy isolation, late-bound df handling,
tool counts (11 vs 17), connection sharing, and the close() contract.
"""

from __future__ import annotations

import os
import tempfile

import pandas as pd
import pytest

from mads.manager import MemoryManager
from mads.memory import HashEmbedder


# --------------------------------------------------------------------- fixtures


@pytest.fixture
def manager(tmp_path) -> MemoryManager:
    db = str(tmp_path / "manager.sqlite")
    m = MemoryManager(
        user_id="alice",
        project_id="proj1",
        db_path=db,
        embedder=HashEmbedder(),
    )
    yield m
    m.close()


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "monthly_charges": [10.0, 20.0, 30.0],
            "churn": ["No", "Yes", "No"],
        }
    )


# --------------------------------------------------------------------- construction


def test_constructor_records_tenancy(manager):
    assert manager.user_id == "alice"
    assert manager.project_id == "proj1"


def test_rejects_empty_user_id(tmp_path):
    db = str(tmp_path / "x.sqlite")
    with pytest.raises(ValueError):
        MemoryManager(user_id="", project_id="p", db_path=db)


def test_rejects_empty_project_id(tmp_path):
    db = str(tmp_path / "x.sqlite")
    with pytest.raises(ValueError):
        MemoryManager(user_id="u", project_id="", db_path=db)


def test_rejects_non_string_user_id(tmp_path):
    db = str(tmp_path / "x.sqlite")
    with pytest.raises(ValueError):
        MemoryManager(user_id=42, project_id="p", db_path=db)  # type: ignore[arg-type]


def test_db_path_precedence_arg_wins(tmp_path, monkeypatch):
    # arg explicitly given > env var > default.
    monkeypatch.setenv("MADS_DB_PATH", str(tmp_path / "env.sqlite"))
    explicit = str(tmp_path / "arg.sqlite")
    m = MemoryManager(user_id="u", project_id="p", db_path=explicit)
    try:
        assert m.db_path == explicit
        # file is materialized by sqlite3.connect upon first write
        # (we just check the path getter)
    finally:
        m.close()


def test_db_path_env_var_used_when_arg_none(tmp_path, monkeypatch):
    env_path = str(tmp_path / "fromenv.sqlite")
    monkeypatch.setenv("MADS_DB_PATH", env_path)
    m = MemoryManager(user_id="u", project_id="p")
    try:
        assert m.db_path == env_path
    finally:
        m.close()


# --------------------------------------------------------------------- shared conn


def test_single_shared_connection(manager):
    # All three lazy stores share the manager's connection.
    assert manager._metric_store_._conn is manager.conn
    assert manager._episode_store_._conn is manager.conn
    assert manager._dataset_store_._conn is manager.conn


def test_close_is_idempotent(tmp_path):
    db = str(tmp_path / "close.sqlite")
    m = MemoryManager(user_id="u", project_id="p", db_path=db)
    m.close()
    # No raise on second call.
    m.close()


def test_close_releases_connection(tmp_path):
    db = str(tmp_path / "close.sqlite")
    m = MemoryManager(user_id="u", project_id="p", db_path=db)
    m.close()
    assert m._conn is None or m._closed


# --------------------------------------------------------------------- factories


def test_metric_factory_is_cached(manager):
    f1 = manager.metric
    f2 = manager.metric
    assert f1 is f2


def test_episode_factory_is_cached(manager):
    f1 = manager.episode
    f2 = manager.episode
    assert f1 is f2


def test_metric_tool_count(manager):
    assert len(manager.metric.tools()) == 5


def test_episode_tool_count(manager):
    assert len(manager.episode.tools()) == 6


# --------------------------------------------------------------------- bind_dataframe


def test_dataset_tools_requires_bind(manager):
    with pytest.raises(RuntimeError):
        manager.dataset_tools()


def test_bind_then_dataset_tools(manager, df):
    manager.bind_dataframe(df, dataset_name="churn", table_name="churn")
    tools = manager.dataset_tools()
    assert len(tools) == 6
    names = {t.name for t in tools}
    assert "register_dataset" in names
    assert "recall_dataset" in names


def test_bind_swap_invalidates_cache(manager):
    df1 = pd.DataFrame({"a": [1]})
    df2 = pd.DataFrame({"b": [2]})
    manager.bind_dataframe(df1, dataset_name="d", table_name="t")
    t1 = manager.dataset_tools()
    manager.bind_dataframe(df2, dataset_name="d", table_name="t")
    t2 = manager.dataset_tools()
    # New df identity → new tool closures.
    assert t1 is not t2
    assert t1[0] is not t2[0]


def test_bind_idempotent_within_turn(manager, df):
    manager.bind_dataframe(df, dataset_name="churn", table_name="churn")
    t1 = manager.dataset_tools()
    t2 = manager.dataset_tools()
    # Same df identity → same cached closures.
    assert t1 == t2
    assert t1[0] is t2[0]


def test_bind_rejects_empty_dataset_name(manager, df):
    with pytest.raises(ValueError):
        manager.bind_dataframe(df, dataset_name="", table_name="t")


# --------------------------------------------------------------------- all_tools


def test_all_tools_without_dataset(manager):
    # Loader's first turn has no df: 5 metric + 4 preference + 6 episode = 15.
    tools = manager.all_tools(include_dataset=False)
    assert len(tools) == 15


def test_all_tools_with_dataset_after_bind(manager, df):
    manager.bind_dataframe(df, dataset_name="churn", table_name="churn")
    tools = manager.all_tools(include_dataset=True)
    # 5 metric + 4 preference + 6 episode + 6 dataset + 1 pandas-exec = 22.
    assert len(tools) == 22
    names = {t.name for t in tools}
    assert "run_pandas_code" in names


def test_all_tools_default_include_dataset(manager, df):
    # The default is include_dataset=True; without bind it raises.
    with pytest.raises(RuntimeError):
        manager.all_tools()


# --------------------------------------------------------------------- tenancy isolation


def test_tenancy_isolated_across_managers(tmp_path):
    db = str(tmp_path / "shared.sqlite")
    m1 = MemoryManager(
        user_id="alice", project_id="proj1", db_path=db, embedder=HashEmbedder()
    )
    m2 = MemoryManager(
        user_id="alice", project_id="proj2", db_path=db, embedder=HashEmbedder()
    )
    try:
        # write a metric in proj1
        add = m1.metric.tools()[0]  # add_metric_definition
        assert "Metric 'revenue'" in add.invoke(
            {"metric_name": "revenue", "business_definition": "sum of x"}
        )
        # proj2 should not see it
        lst = m2.metric.tools()[2]  # list_metric_definitions
        out = lst.invoke({})
        assert "revenue" not in out
    finally:
        m1.close()
        m2.close()


# --------------------------------------------------------------------- preference retrieval


def test_retrieve_includes_preferences(manager):
    """``MemoryManager.retrieve`` surfaces active preferences as
    ``category="preference"`` rows alongside metric/dataset/episode
    results (closes the v1.1 retrieval-surface gap)."""
    # Write a preference via the store directly so the test doesn't
    # depend on the full tool harness.
    manager._preference_store_.set(
        "alice", "report_style", "exec summary first"
    )
    out = manager.retrieve(
        query="anything",
        user_id="alice",
        project_id="proj1",
        k=5,
    )
    pref_rows = [r for r in out if r.category == "preference"]
    assert len(pref_rows) == 1
    assert pref_rows[0].memory["preference_key"] == "report_style"
    assert pref_rows[0].score == 1.0


def test_format_context_renders_preference_blocks(manager):
    """``format_context`` emits a ``[memory:preference:KEY]`` block for
    preference RetrievedMemory rows."""
    manager._preference_store_.set("alice", "report_style", "exec summary first")
    out = manager.retrieve(
        query="anything", user_id="alice", project_id="proj1", k=5
    )
    rendered = manager.format_context(out)
    assert "[memory:preference:report_style]" in rendered
    # The value row must surface so the LLM has something actionable.
    assert "exec summary first" in rendered


# --------------------------------------------------------------------- axis-model


def test_bind_dataframe_stashes_fingerprint(manager, df):
    """``bind_dataframe`` computes and stashes the semantic fingerprint
    so dataset-scoped tools (data-quality) can be built from it."""
    assert manager.dataset_fingerprint is None
    manager.bind_dataframe(df, dataset_name="churn", table_name="churn")
    assert isinstance(manager.dataset_fingerprint, str)
    assert len(manager.dataset_fingerprint) == 64  # sha256 hex


def test_data_quality_tools_requires_bind(manager):
    """``data_quality_tools()`` raises without a bound df (no fingerprint)."""
    with pytest.raises(RuntimeError):
        manager.data_quality_tools()


def test_data_quality_tools_after_bind(manager, df):
    """``data_quality_tools()`` after bind returns 6 tools whose schemas
    expose no tenancy or fingerprint."""
    manager.bind_dataframe(df, dataset_name="churn", table_name="churn")
    tools = manager.data_quality_tools()
    assert len(tools) == 6
    for t in tools:
        schema = t.args_schema
        fields = set(getattr(schema, "model_fields", {}).keys())
        assert "user_id" not in fields
        assert "project_id" not in fields
        assert "dataset_fingerprint" not in fields


def test_data_quality_tools_isolated_across_datasets(manager, df):
    """Issues recorded on dataset A invisible after rebinding to dataset B
    (the new axis model: data quality is identity by fingerprint)."""
    manager.bind_dataframe(df, dataset_name="churn", table_name="churn")
    fp_a = manager.dataset_fingerprint
    a_tools = {t.name: t for t in manager.data_quality_tools()}
    a_tools["add_data_quality_issue"].invoke(
        {
            "table_name": "churn",
            "column_name": "TotalCharges",
            "issue_type": "encoding_issue",
            "issue_description": "stored as str, whitespace for new customers",
        }
    )

    # Switch to a different df with different schema → different fingerprint.
    other_df = pd.DataFrame({"x": [1], "y": ["a"]})
    manager.bind_dataframe(other_df, dataset_name="other", table_name="other")
    fp_b = manager.dataset_fingerprint
    assert fp_a != fp_b

    b_tools = {t.name: t for t in manager.data_quality_tools()}
    out_b = b_tools["list_data_quality_issues_for_table"].invoke(
        {"table_name": "churn"}
    )
    assert "No active data-quality issues" in out_b


def test_all_tools_still_15_and_22(manager, df):
    """Tool surface counts: 15 without df, 22 with df (post run_pandas_code).
    Data-quality tools are emitted via ``data_quality_tools()`` but are NOT
    part of ``all_tools()`` — they live on a separate surface so loader's
    pre-bind turn doesn't see them. The 22 = 15 (metric+preference+episode)
    + 6 (dataset) + 1 (run_pandas_code) breakdown holds because the new
    execution tool ships behind ``include_dataset=True`` (it needs a df)."""
    assert len(manager.all_tools(include_dataset=False)) == 15
    manager.bind_dataframe(df, dataset_name="churn", table_name="churn")
    assert len(manager.all_tools(include_dataset=True)) == 22


def test_retrieve_passes_project_id_to_preference_retriever(manager):
    """``MemoryManager.retrieve`` threads ``project_id`` through to the
    preference retriever so project-scoped prefs surface alongside global."""
    # Global pref for proj1's user.
    manager._preference_store_.set(
        "alice", "global_pref", "g"
    )
    # Project-scoped pref for proj1.
    manager._preference_store_.set(
        "alice", "project_pref", "p", project_id="proj1"
    )
    out = manager.retrieve(
        query="anything",
        user_id="alice",
        project_id="proj1",
        k=5,
    )
    pref_keys = {
        r.memory["preference_key"]
        for r in out
        if r.category == "preference"
    }
    assert "global_pref" in pref_keys
    assert "project_pref" in pref_keys


def test_retrieve_does_not_leak_other_projects_prefs(manager):
    """Project-scoped pref for proj2 must NOT surface when retrieve runs
    under proj1 (this is the cross_01 fix at the retriever level)."""
    manager._preference_store_.set(
        "alice", "rename_rule", "v_proj2", project_id="proj2"
    )
    out = manager.retrieve(
        query="anything",
        user_id="alice",
        project_id="proj1",
        k=5,
    )
    pref_values = {
        r.memory["preference_value"]
        for r in out
        if r.category == "preference"
    }
    assert "v_proj2" not in pref_values


def test_embedder_passed_to_episode_store(tmp_path):
    db = str(tmp_path / "x.sqlite")
    embedder = HashEmbedder()
    m = MemoryManager(
        user_id="u", project_id="p", db_path=db, embedder=embedder
    )
    try:
        # Trigger lazy construction
        _ = m.episode
        assert m._episode_store_._embedder is embedder
    finally:
        m.close()


# =================================================================== v1.4: schema coupling


def test_available_columns_returns_df_columns_when_bound(manager, df):
    """v1.4: _available_columns_for_session uses df.columns when bound."""
    manager.bind_dataframe(df, dataset_name="churn", table_name="churn")
    cols = manager._available_columns_for_session()
    assert cols == set(df.columns)


def test_available_columns_returns_none_without_bind(manager):
    """v1.4: No df bound, no fingerprint -> None (caller assumes applicable)."""
    cols = manager._available_columns_for_session()
    assert cols is None


def test_available_columns_falls_back_to_column_memory(manager, df):
    """v1.4: when df is unbound but the dataset_store has columns for the
    stashed fingerprint+table_name, those columns are returned."""
    # Manually register a column for the dataset fingerprint, then clear
    # the bound df to force the fallback path.
    manager.bind_dataframe(df, dataset_name="churn", table_name="churn")
    fp = manager.dataset_fingerprint
    table = manager._table_name
    # Register a couple of columns directly in the column_memory store.
    manager._dataset_store_.register_column(
        fingerprint=fp,
        table_name=table,
        column_name="tenure",
        semantic_type="int",
    )
    manager._dataset_store_.register_column(
        fingerprint=fp,
        table_name=table,
        column_name="Churn",
        semantic_type="categorical",
    )
    # Unbind the df (but preserve fingerprint + table_name).
    manager._df = None
    cols = manager._available_columns_for_session()
    assert cols == {"tenure", "Churn"}


def test_retrieve_decorates_metric_with_applicable(manager, df):
    """v1.4: A bound df makes inapplicable metrics surface with
    applicable=False + missing_columns + display snapshot."""
    # Seed a metric whose required columns are NOT in the df.
    add = manager.metric.tools()[0]
    add.invoke(
        {
            "metric_name": "active_user",
            "business_definition": "made a payment recently",
            "required_columns": ["last_payment_date"],
        }
    )
    manager.bind_dataframe(df, dataset_name="d", table_name="d")
    out = manager.retrieve(
        query="active_user", user_id="alice", project_id="proj1", k=5
    )
    metric_rows = [r for r in out if r.category == "metric"]
    assert len(metric_rows) == 1
    mem = metric_rows[0].memory
    assert mem["applicable"] is False
    assert mem["missing_columns"] == ["last_payment_date"]


def test_format_context_renders_applicable_false_annotation(manager, df):
    """v1.4: APPLICABLE: false annotation is appended to inapplicable metric blocks."""
    add = manager.metric.tools()[0]
    add.invoke(
        {
            "metric_name": "active_user",
            "business_definition": "paid recently",
            "required_columns": ["last_payment_date"],
        }
    )
    manager.bind_dataframe(df, dataset_name="d", table_name="d")
    out = manager.retrieve(
        query="active_user", user_id="alice", project_id="proj1", k=5
    )
    rendered = manager.format_context(out)
    assert "APPLICABLE: false" in rendered
    assert "last_payment_date" in rendered
    # bound df column listing should appear.
    assert "bound df has [" in rendered
    # And at least one of the df's actual columns must be listed.
    assert "churn" in rendered.lower() or "customer_id" in rendered


def test_format_context_no_annotation_when_applicable(manager, df):
    """v1.4: applicable=True metrics get no APPLICABLE annotation."""
    add = manager.metric.tools()[0]
    # Use a column the df actually has.
    add.invoke(
        {
            "metric_name": "churn_rate",
            "business_definition": "share of Churn==Yes",
            "required_columns": ["churn"],
        }
    )
    manager.bind_dataframe(df, dataset_name="d", table_name="d")
    out = manager.retrieve(
        query="churn_rate", user_id="alice", project_id="proj1", k=5
    )
    rendered = manager.format_context(out)
    assert "APPLICABLE: false" not in rendered
