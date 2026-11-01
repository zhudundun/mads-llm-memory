"""Pytest suite for the ``data_quality_issue`` memory module (v1.2 axis model).

Identity is now ``(dataset_fingerprint, normalized_table,
normalized_column, normalized_issue_type)``. ``project_id`` is AUDIT-only.
"""

from __future__ import annotations

import re
import sqlite3

import pytest

from mads.memory.data_quality import (
    DataQualityIssue,
    DataQualityStore,
    _normalize_column,
    _normalize_issue_type,
    _normalize_table,
)
from mads.memory.data_quality_tools import (
    DataQualityToolFactory,
    make_data_quality_factory,
)


# --------------------------------------------------------------------- fixtures


# Stable test fingerprints (sha256-shaped strings; the store doesn't validate
# them as hex, only requires non-empty).
FP_A = "a" * 64
FP_B = "b" * 64


@pytest.fixture
def store():
    s = DataQualityStore(db_path=":memory:")
    try:
        yield s
    finally:
        s.close()


# ----------------------------------------------------------- helper assertions


_UUID_HEX_RE = re.compile(r"^[0-9a-f]{32}$")


def _tools_by_name(tools):
    return {t.name: t for t in tools}


def _seed(
    store,
    *,
    project_id="p1",
    user_id="u1",
    dataset_fingerprint=FP_A,
    table_name="payments",
    column_name="refund_status",
    issue_type="temporal_lag",
    issue_description="refund_status lags upstream by ~24 hours",
    recommended_handling="exclude rows where ts > now()-24h",
    severity="warning",
    confidence=1.0,
):
    return store.upsert(
        user_id=user_id,
        project_id=project_id,
        dataset_fingerprint=dataset_fingerprint,
        table_name=table_name,
        column_name=column_name,
        issue_type=issue_type,
        issue_description=issue_description,
        recommended_handling=recommended_handling,
        severity=severity,
        confidence=confidence,
    )


# =================================================================== NORMALIZATION


def test_normalize_table_examples():
    assert _normalize_table("Sessions") == "sessions"
    assert _normalize_table("refund_status") == "refund_status"
    assert _normalize_table("Temporal Lag") == "temporal_lag"


def test_normalize_collapses_runs_and_strips_edges():
    assert _normalize_table("__Sessions  --  v2!!__") == "sessions_v2"
    assert _normalize_issue_type("Row-Filter (required)") == "row_filter_required"
    assert _normalize_column("Refund Status") == "refund_status"


def test_normalize_empty_raises():
    with pytest.raises(ValueError):
        _normalize_table("")
    with pytest.raises(ValueError):
        _normalize_table("!!!")
    with pytest.raises(ValueError):
        _normalize_issue_type("")
    with pytest.raises(ValueError):
        _normalize_issue_type("   ")


def test_normalize_column_passes_none_through():
    assert _normalize_column(None) is None
    assert _normalize_column("Refund Status") == "refund_status"
    with pytest.raises(ValueError):
        _normalize_column("!!!")


# =================================================================== IDENTITY


def test_identity_table_case_insensitive(store):
    """'Sessions' and 'sessions' map to the same identity within a fingerprint."""
    d1, a1 = _seed(
        store,
        table_name="Sessions",
        column_name="session_duration",
        issue_type="reliability_conditional",
        issue_description="bad on iOS 4.2",
    )
    assert a1 == "inserted"
    d2, a2 = _seed(
        store,
        table_name="sessions",
        column_name="session_duration",
        issue_type="reliability_conditional",
        issue_description="bad on iOS 4.2",
    )
    assert a2 == "noop"
    assert d2.id == d1.id


def test_identity_issue_type_case_insensitive(store):
    d1, _ = _seed(store, issue_type="Temporal Lag")
    d2, action = _seed(store, issue_type="temporal_lag")
    assert action == "noop"
    assert d2.id == d1.id


def test_dataset_scoping_isolates_same_identity(store):
    """Two datasets (different fingerprints) can have the same identity tuple
    with independent rows."""
    _seed(store, dataset_fingerprint=FP_A, issue_description="A-flavor desc")
    _seed(store, dataset_fingerprint=FP_B, issue_description="B-flavor desc")

    issues_a = store.list_active_for_table(
        dataset_fingerprint=FP_A, table_name="payments"
    )
    issues_b = store.list_active_for_table(
        dataset_fingerprint=FP_B, table_name="payments"
    )
    assert len(issues_a) == 1 and len(issues_b) == 1
    assert issues_a[0].id != issues_b[0].id
    assert "A-flavor" in issues_a[0].issue_description
    assert "B-flavor" in issues_b[0].issue_description


def test_same_dataset_two_projects_share_library(store):
    """Two projects analyzing the SAME dataset (by fingerprint) share one
    issue library — project_id is audit, not identity."""
    _seed(
        store,
        project_id="growth",
        dataset_fingerprint=FP_A,
        issue_description="shared observation",
    )
    # Same dataset, different project — must be a noop, not a fresh insert.
    _, action = _seed(
        store,
        project_id="marketing",
        dataset_fingerprint=FP_A,
        issue_description="shared observation",
    )
    assert action == "noop"


def test_table_level_and_column_level_coexist(store):
    _seed(
        store,
        table_name="users",
        column_name=None,
        issue_type="row_filter_required",
        issue_description="exclude test users",
    )
    _seed(
        store,
        table_name="users",
        column_name="email",
        issue_type="row_filter_required",
        issue_description="exclude internal emails",
    )

    table_issues = store.list_active_for_table(
        dataset_fingerprint=FP_A, table_name="users"
    )
    assert len(table_issues) == 2
    cols = sorted(
        (i.column_name if i.column_name is not None else "")
        for i in table_issues
    )
    assert cols == ["", "email"]


# =================================================================== UPSERT ACTIONS


def test_first_insert_returns_inserted(store):
    d, action = _seed(store)
    assert action == "inserted"
    assert _UUID_HEX_RE.match(d.id)
    assert isinstance(d, DataQualityIssue)
    assert d.table_name == "payments"
    assert d.normalized_table == "payments"
    assert d.valid_to is None
    assert d.user_id == "u1"
    assert d.dataset_fingerprint == FP_A

    rt = store.list_active_for_table(
        dataset_fingerprint=FP_A, table_name="payments"
    )
    assert len(rt) == 1 and rt[0].id == d.id


def test_identical_reinsert_is_noop(store):
    d1, _ = _seed(store)
    d2, action = _seed(store)
    assert action == "noop"
    assert d2.id == d1.id

    history = store.history(
        dataset_fingerprint=FP_A,
        table_name="payments",
        column_name="refund_status",
        issue_type="temporal_lag",
    )
    assert len(history) == 1


def test_severity_only_change_updates_in_place(store):
    d1, _ = _seed(store, severity="warning")
    d2, action = _seed(store, severity="critical")
    assert action == "updated_in_place"
    assert d2.id == d1.id
    assert d2.severity == "critical"

    history = store.history(
        dataset_fingerprint=FP_A,
        table_name="payments",
        column_name="refund_status",
        issue_type="temporal_lag",
    )
    assert len(history) == 1
    assert history[0].severity == "critical"


def test_confidence_only_change_updates_in_place(store):
    d1, _ = _seed(store, confidence=0.6)
    d2, action = _seed(store, confidence=0.95)
    assert action == "updated_in_place"
    assert d2.id == d1.id
    assert d2.confidence == pytest.approx(0.95)


def test_issue_description_change_versions(store):
    d1, _ = _seed(store, issue_description="v1 desc")
    d2, action = _seed(store, issue_description="v2 desc")
    assert action == "versioned"
    assert d2.id != d1.id

    history = store.history(
        dataset_fingerprint=FP_A,
        table_name="payments",
        column_name="refund_status",
        issue_type="temporal_lag",
    )
    assert len(history) == 2
    assert history[0].id == d1.id
    assert history[1].id == d2.id
    assert history[0].valid_to is not None
    assert history[1].valid_to is None


def test_recommended_handling_change_versions(store):
    _, _ = _seed(store, recommended_handling="v1 handling")
    d2, action = _seed(store, recommended_handling="v2 handling")
    assert action == "versioned"

    history = store.history(
        dataset_fingerprint=FP_A,
        table_name="payments",
        column_name="refund_status",
        issue_type="temporal_lag",
    )
    assert len(history) == 2


def test_both_definitional_fields_change_versions(store):
    _, _ = _seed(store, issue_description="v1 d", recommended_handling="v1 h")
    d2, action = _seed(
        store, issue_description="v2 d", recommended_handling="v2 h"
    )
    assert action == "versioned"

    history = store.history(
        dataset_fingerprint=FP_A,
        table_name="payments",
        column_name="refund_status",
        issue_type="temporal_lag",
    )
    assert len(history) == 2
    assert history[1].issue_description == "v2 d"
    assert history[1].recommended_handling == "v2 h"


# =================================================================== INVARIANTS


def test_versioning_invariant_one_active_row(store):
    _seed(store, issue_description="v1")
    _seed(store, issue_description="v2")
    _seed(store, issue_description="v3")

    cnt = store._conn.execute(
        """
        SELECT COUNT(*) AS c FROM data_quality_memory
         WHERE dataset_fingerprint = ? AND normalized_table = ?
           AND normalized_column = ? AND normalized_issue_type = ?
           AND valid_to IS NULL
        """,
        (FP_A, "payments", "refund_status", "temporal_lag"),
    ).fetchone()["c"]
    assert cnt == 1

    total = store._conn.execute(
        """
        SELECT COUNT(*) AS c FROM data_quality_memory
         WHERE dataset_fingerprint = ? AND normalized_table = ?
           AND normalized_column = ? AND normalized_issue_type = ?
        """,
        (FP_A, "payments", "refund_status", "temporal_lag"),
    ).fetchone()["c"]
    assert total == 3


def test_versioning_invariant_valid_to_equals_next_valid_from(store):
    _seed(store, issue_description="v1")
    _seed(store, issue_description="v2")

    history = store.history(
        dataset_fingerprint=FP_A,
        table_name="payments",
        column_name="refund_status",
        issue_type="temporal_lag",
    )
    assert len(history) == 2
    assert history[0].valid_to == history[1].valid_from


def test_partial_unique_index_rejects_duplicate_column_level(store):
    _seed(store, issue_description="legit v1")

    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute(
            """
            INSERT INTO data_quality_memory (
                id, user_id, project_id, dataset_fingerprint,
                table_name, column_name, issue_type,
                normalized_table, normalized_column, normalized_issue_type,
                issue_description, recommended_handling,
                severity, confidence,
                valid_from, valid_to, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL,
                      'warning', 1.0,
                      CURRENT_TIMESTAMP, NULL,
                      CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            (
                "00000000000000000000000000000002",
                "u1", "p1", FP_A,
                "payments", "refund_status", "temporal_lag",
                "payments", "refund_status", "temporal_lag",
                "rival v1",
            ),
        )


def test_partial_unique_index_rejects_duplicate_table_level(store):
    _seed(
        store,
        table_name="users",
        column_name=None,
        issue_type="row_filter_required",
        issue_description="exclude test users",
    )

    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute(
            """
            INSERT INTO data_quality_memory (
                id, user_id, project_id, dataset_fingerprint,
                table_name, column_name, issue_type,
                normalized_table, normalized_column, normalized_issue_type,
                issue_description, recommended_handling,
                severity, confidence,
                valid_from, valid_to, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, NULL, ?, ?, NULL, ?, ?, NULL,
                      'critical', 1.0,
                      CURRENT_TIMESTAMP, NULL,
                      CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            (
                "00000000000000000000000000000003",
                "u1", "p1", FP_A,
                "users", "row_filter_required",
                "users", "row_filter_required",
                "rival rule",
            ),
        )


# =================================================================== RETRIEVAL


def test_get_active_for_column_returns_all_issue_types(store):
    _seed(
        store,
        table_name="sessions",
        column_name="session_duration",
        issue_type="reliability_conditional",
        issue_description="bad on iOS 4.2",
    )
    _seed(
        store,
        table_name="sessions",
        column_name="session_duration",
        issue_type="outlier_pattern",
        issue_description="huge tail on weekends",
    )
    hits = store.get_active_for_column(
        dataset_fingerprint=FP_A,
        table_name="sessions",
        column_name="session_duration",
    )
    types = sorted(h.issue_type for h in hits)
    assert types == ["outlier_pattern", "reliability_conditional"]


def test_get_active_for_column_empty(store):
    assert (
        store.get_active_for_column(
            dataset_fingerprint=FP_A, table_name="ghost", column_name="x"
        )
        == []
    )


def test_list_active_for_table_includes_column_and_table_level(store):
    _seed(
        store,
        table_name="users",
        column_name=None,
        issue_type="row_filter_required",
        issue_description="exclude test users",
    )
    _seed(
        store,
        table_name="users",
        column_name="signup_date",
        issue_type="temporal_lag",
        issue_description="lags 1h",
    )
    rows = store.list_active_for_table(
        dataset_fingerprint=FP_A, table_name="users"
    )
    assert len(rows) == 2
    assert rows[0].column_name is None
    assert rows[1].column_name == "signup_date"


def test_list_active_for_table_excludes_resolved(store):
    _seed(store)
    store.resolve(
        user_id="u1",
        project_id="p1",
        dataset_fingerprint=FP_A,
        table_name="payments",
        column_name="refund_status",
        issue_type="temporal_lag",
    )
    rows = store.list_active_for_table(
        dataset_fingerprint=FP_A, table_name="payments"
    )
    assert rows == []


def test_search_active_matches_description_and_other_fields(store):
    _seed(
        store,
        table_name="events",
        column_name="campaign_id",
        issue_type="semantic_null",
        issue_description="campaign_id is null for organic traffic",
        recommended_handling="treat NULL as 'organic'",
    )
    _seed(
        store,
        table_name="sessions",
        column_name="session_duration",
        issue_type="reliability_conditional",
        issue_description="unreliable on iOS 4.2",
    )

    hits_desc = store.search_active(
        dataset_fingerprint=FP_A, query="organic"
    )
    assert len(hits_desc) == 1
    assert hits_desc[0].column_name == "campaign_id"

    hits_table = store.search_active(
        dataset_fingerprint=FP_A, query="events"
    )
    assert len(hits_table) == 1
    assert hits_table[0].table_name == "events"

    hits_type = store.search_active(
        dataset_fingerprint=FP_A, query="reliability"
    )
    assert len(hits_type) == 1

    hits_rec = store.search_active(
        dataset_fingerprint=FP_A, query="treat null"
    )
    assert len(hits_rec) == 1


def test_search_active_is_case_insensitive(store):
    _seed(
        store,
        table_name="sessions",
        column_name="session_duration",
        issue_type="reliability_conditional",
        issue_description="Unreliable on iOS 4.2",
    )
    hits_lower = store.search_active(dataset_fingerprint=FP_A, query="ios")
    hits_upper = store.search_active(dataset_fingerprint=FP_A, query="IOS")
    hits_mixed = store.search_active(dataset_fingerprint=FP_A, query="Ios")
    assert len(hits_lower) == 1
    assert len(hits_upper) == 1
    assert len(hits_mixed) == 1


def test_history_oldest_first_includes_active_and_closed(store):
    _seed(store, issue_description="v1")
    _seed(store, issue_description="v2")
    _seed(store, issue_description="v3")

    history = store.history(
        dataset_fingerprint=FP_A,
        table_name="payments",
        column_name="refund_status",
        issue_type="temporal_lag",
    )
    assert [h.issue_description for h in history] == ["v1", "v2", "v3"]
    assert history[0].valid_to is not None
    assert history[1].valid_to is not None
    assert history[2].valid_to is None


def test_history_empty_for_unknown_identity(store):
    assert (
        store.history(
            dataset_fingerprint=FP_A,
            table_name="ghost",
            column_name="x",
            issue_type="temporal_lag",
        )
        == []
    )


# =================================================================== TENANCY


def test_dataset_isolation_via_factories(store):
    """Issues on dataset A invisible to a dataset-B factory."""
    _seed(store, dataset_fingerprint=FP_A)
    f_a = make_data_quality_factory(store, "u1", "p1", FP_A)
    f_b = make_data_quality_factory(store, "u1", "p1", FP_B)
    a_tools = _tools_by_name(f_a.tools())
    b_tools = _tools_by_name(f_b.tools())

    out_a = a_tools["list_data_quality_issues_for_table"].invoke(
        {"table_name": "payments"}
    )
    out_b = b_tools["list_data_quality_issues_for_table"].invoke(
        {"table_name": "payments"}
    )
    assert "refund_status" in out_a
    assert "No active data-quality issues" in out_b


def test_users_share_dataset_library(store):
    """Two users seeing the same dataset share one library."""
    _seed(store, user_id="alice")
    f_alice = make_data_quality_factory(store, "alice", "p1", FP_A)
    f_bob = make_data_quality_factory(store, "bob", "p1", FP_A)
    a = _tools_by_name(f_alice.tools())
    b = _tools_by_name(f_bob.tools())

    out_a = a["list_data_quality_issues_for_table"].invoke(
        {"table_name": "payments"}
    )
    out_b = b["list_data_quality_issues_for_table"].invoke(
        {"table_name": "payments"}
    )
    assert "refund_status" in out_a
    assert "refund_status" in out_b

    b["add_data_quality_issue"].invoke(
        {
            "table_name": "payments",
            "column_name": "refund_status",
            "issue_type": "temporal_lag",
            "issue_description": "refund_status lags upstream by ~24 hours",
            "recommended_handling": "exclude rows where ts > now()-24h",
            "severity": "critical",
        }
    )
    out_a2 = a["list_data_quality_issues_for_table"].invoke(
        {"table_name": "payments"}
    )
    assert "critical" in out_a2


# =================================================================== VALIDATION


def test_empty_table_name_raises(store):
    with pytest.raises(ValueError):
        _seed(store, table_name="")


def test_unnormalizable_table_name_raises(store):
    with pytest.raises(ValueError):
        _seed(store, table_name="!!!")


def test_empty_issue_type_raises(store):
    with pytest.raises(ValueError):
        _seed(store, issue_type="")


def test_empty_issue_description_raises(store):
    with pytest.raises(ValueError):
        _seed(store, issue_description="")


def test_invalid_severity_raises(store):
    with pytest.raises(ValueError):
        _seed(store, severity="urgent")


def test_confidence_out_of_range_raises(store):
    with pytest.raises(ValueError):
        _seed(store, confidence=1.5)
    with pytest.raises(ValueError):
        _seed(store, confidence=-0.1)


def test_empty_dataset_fingerprint_raises(store):
    """dataset_fingerprint is REQUIRED (axis-model: IDENTITY_REQUIRED)."""
    with pytest.raises(ValueError):
        _seed(store, dataset_fingerprint="")


# =================================================================== RESOLVE


def test_resolve_active_issue_sets_valid_to(store):
    d, _ = _seed(store)
    closed = store.resolve(
        user_id="u1",
        project_id="p1",
        dataset_fingerprint=FP_A,
        table_name="payments",
        column_name="refund_status",
        issue_type="temporal_lag",
    )
    assert closed is not None
    assert closed.id == d.id
    assert closed.valid_to is not None

    assert (
        store.list_active_for_table(
            dataset_fingerprint=FP_A, table_name="payments"
        )
        == []
    )
    assert (
        store.get_active_for_column(
            dataset_fingerprint=FP_A,
            table_name="payments",
            column_name="refund_status",
        )
        == []
    )

    history = store.history(
        dataset_fingerprint=FP_A,
        table_name="payments",
        column_name="refund_status",
        issue_type="temporal_lag",
    )
    assert len(history) == 1
    assert history[0].id == d.id
    assert history[0].valid_to is not None


def test_resolve_then_readd_creates_fresh_row(store):
    d1, _ = _seed(store)
    store.resolve(
        user_id="u1",
        project_id="p1",
        dataset_fingerprint=FP_A,
        table_name="payments",
        column_name="refund_status",
        issue_type="temporal_lag",
    )
    d2, action = _seed(store)
    assert action == "inserted"
    assert d2.id != d1.id
    assert d2.valid_to is None

    history = store.history(
        dataset_fingerprint=FP_A,
        table_name="payments",
        column_name="refund_status",
        issue_type="temporal_lag",
    )
    assert len(history) == 2
    assert history[0].valid_to is not None
    assert history[1].valid_to is None


def test_resolve_nonexistent_returns_none(store):
    out = store.resolve(
        user_id="u1",
        project_id="p1",
        dataset_fingerprint=FP_A,
        table_name="payments",
        column_name="refund_status",
        issue_type="temporal_lag",
    )
    assert out is None


def test_resolve_already_resolved_returns_none(store):
    _seed(store)
    first = store.resolve(
        user_id="u1",
        project_id="p1",
        dataset_fingerprint=FP_A,
        table_name="payments",
        column_name="refund_status",
        issue_type="temporal_lag",
    )
    assert first is not None
    second = store.resolve(
        user_id="u1",
        project_id="p1",
        dataset_fingerprint=FP_A,
        table_name="payments",
        column_name="refund_status",
        issue_type="temporal_lag",
    )
    assert second is None


# =================================================================== TOOL LAYER


def test_factory_returns_exactly_six_tools(store):
    f = make_data_quality_factory(store, "u1", "p1", FP_A)
    assert isinstance(f, DataQualityToolFactory)
    tools = f.tools()
    assert len(tools) == 6
    names = sorted(t.name for t in tools)
    assert names == [
        "add_data_quality_issue",
        "get_data_quality_issue_history",
        "get_data_quality_issues_for_column",
        "list_data_quality_issues_for_table",
        "resolve_data_quality_issue",
        "search_data_quality_issues",
    ]


def test_tool_signatures_hide_tenancy_and_fingerprint(store):
    """None of the 6 tools expose user_id, project_id, or dataset_fingerprint."""
    tools = make_data_quality_factory(store, "u1", "p1", FP_A).tools()
    for t in tools:
        schema = t.args_schema.model_json_schema()
        props = set(schema.get("properties", {}).keys())
        assert "user_id" not in props, (
            f"{t.name} leaks user_id in its arg schema: {props}"
        )
        assert "project_id" not in props, (
            f"{t.name} leaks project_id in its arg schema: {props}"
        )
        assert "dataset_fingerprint" not in props, (
            f"{t.name} leaks dataset_fingerprint: {props}"
        )


def test_add_tool_inserts_then_noops(store):
    tools = _tools_by_name(
        make_data_quality_factory(store, "u1", "p1", FP_A).tools()
    )

    out1 = tools["add_data_quality_issue"].invoke(
        {
            "table_name": "payments",
            "column_name": "refund_status",
            "issue_type": "temporal_lag",
            "issue_description": "refund_status lags upstream by ~24 hours",
            "recommended_handling": "exclude rows where ts > now()-24h",
        }
    )
    assert "inserted" in out1
    assert "version 1" in out1
    assert "temporal_lag" in out1
    assert "refund_status" in out1

    out2 = tools["add_data_quality_issue"].invoke(
        {
            "table_name": "payments",
            "column_name": "refund_status",
            "issue_type": "temporal_lag",
            "issue_description": "refund_status lags upstream by ~24 hours",
            "recommended_handling": "exclude rows where ts > now()-24h",
        }
    )
    assert "noop" in out2


def test_add_tool_surfaces_versioned_and_updated_in_place(store):
    tools = _tools_by_name(
        make_data_quality_factory(store, "u1", "p1", FP_A).tools()
    )

    out_ins = tools["add_data_quality_issue"].invoke(
        {
            "table_name": "events",
            "column_name": "campaign_id",
            "issue_type": "semantic_null",
            "issue_description": "null for organic",
            "severity": "info",
        }
    )
    assert "inserted" in out_ins

    out_up = tools["add_data_quality_issue"].invoke(
        {
            "table_name": "events",
            "column_name": "campaign_id",
            "issue_type": "semantic_null",
            "issue_description": "null for organic",
            "severity": "warning",
        }
    )
    assert "updated_in_place" in out_up

    out_v = tools["add_data_quality_issue"].invoke(
        {
            "table_name": "events",
            "column_name": "campaign_id",
            "issue_type": "semantic_null",
            "issue_description": "null for organic AND direct traffic",
            "severity": "warning",
        }
    )
    assert "versioned" in out_v
    assert "version 2" in out_v

    out_n = tools["add_data_quality_issue"].invoke(
        {
            "table_name": "events",
            "column_name": "campaign_id",
            "issue_type": "semantic_null",
            "issue_description": "null for organic AND direct traffic",
            "severity": "warning",
        }
    )
    assert "noop" in out_n


def test_get_for_column_empty_then_populated(store):
    tools = _tools_by_name(
        make_data_quality_factory(store, "u1", "p1", FP_A).tools()
    )
    empty = tools["get_data_quality_issues_for_column"].invoke(
        {"table_name": "sessions", "column_name": "session_duration"}
    )
    assert "No active data-quality issues" in empty

    tools["add_data_quality_issue"].invoke(
        {
            "table_name": "sessions",
            "column_name": "session_duration",
            "issue_type": "reliability_conditional",
            "issue_description": "unreliable on iOS 4.2",
            "severity": "warning",
        }
    )
    populated = tools["get_data_quality_issues_for_column"].invoke(
        {"table_name": "sessions", "column_name": "session_duration"}
    )
    assert "reliability_conditional" in populated
    assert "iOS" in populated


def test_list_for_table_shows_table_and_column_level(store):
    tools = _tools_by_name(
        make_data_quality_factory(store, "u1", "p1", FP_A).tools()
    )
    tools["add_data_quality_issue"].invoke(
        {
            "table_name": "users",
            "issue_type": "row_filter_required",
            "issue_description": "exclude test users",
            "recommended_handling": "WHERE is_test_user = FALSE",
            "severity": "critical",
        }
    )
    tools["add_data_quality_issue"].invoke(
        {
            "table_name": "users",
            "column_name": "email",
            "issue_type": "encoding_issue",
            "issue_description": "mixed UTF-8 / Latin-1 rows",
        }
    )
    out = tools["list_data_quality_issues_for_table"].invoke(
        {"table_name": "users"}
    )
    assert "row_filter_required" in out
    assert "encoding_issue" in out
    assert "users.(table)" in out


def test_search_tool_finds_match_and_empty_state(store):
    tools = _tools_by_name(
        make_data_quality_factory(store, "u1", "p1", FP_A).tools()
    )
    tools["add_data_quality_issue"].invoke(
        {
            "table_name": "events",
            "column_name": "campaign_id",
            "issue_type": "semantic_null",
            "issue_description": "campaign_id is null for organic traffic",
            "severity": "info",
        }
    )
    out = tools["search_data_quality_issues"].invoke({"query": "organic"})
    assert "campaign_id" in out
    assert "organic" in out

    miss = tools["search_data_quality_issues"].invoke({"query": "xyzqq"})
    assert "No data-quality issues matched" in miss

    err = tools["search_data_quality_issues"].invoke({"query": ""})
    assert "Error" in err


def test_history_tool_shows_versions_and_authors(store):
    f_alice = make_data_quality_factory(store, "alice", "p1", FP_A)
    f_bob = make_data_quality_factory(store, "bob", "p1", FP_A)
    a = _tools_by_name(f_alice.tools())
    b = _tools_by_name(f_bob.tools())

    a["add_data_quality_issue"].invoke(
        {
            "table_name": "payments",
            "column_name": "refund_status",
            "issue_type": "temporal_lag",
            "issue_description": "v1 desc",
            "recommended_handling": "v1 fix",
        }
    )
    b["add_data_quality_issue"].invoke(
        {
            "table_name": "payments",
            "column_name": "refund_status",
            "issue_type": "temporal_lag",
            "issue_description": "v2 desc",
            "recommended_handling": "v2 fix",
        }
    )

    out = a["get_data_quality_issue_history"].invoke(
        {
            "table_name": "payments",
            "column_name": "refund_status",
            "issue_type": "temporal_lag",
        }
    )
    assert "v1 desc" in out
    assert "v2 desc" in out
    assert "alice" in out
    assert "bob" in out
    assert out.index("v1 desc") < out.index("v2 desc")
    assert "2 version" in out


def test_resolve_tool_then_relist_is_empty(store):
    tools = _tools_by_name(
        make_data_quality_factory(store, "u1", "p1", FP_A).tools()
    )
    tools["add_data_quality_issue"].invoke(
        {
            "table_name": "payments",
            "column_name": "refund_status",
            "issue_type": "temporal_lag",
            "issue_description": "lags 24h",
        }
    )
    out = tools["resolve_data_quality_issue"].invoke(
        {
            "table_name": "payments",
            "column_name": "refund_status",
            "issue_type": "temporal_lag",
        }
    )
    assert "resolved" in out
    listed = tools["list_data_quality_issues_for_table"].invoke(
        {"table_name": "payments"}
    )
    assert "No active data-quality issues" in listed


def test_resolve_tool_nonexistent_returns_error(store):
    tools = _tools_by_name(
        make_data_quality_factory(store, "u1", "p1", FP_A).tools()
    )
    out = tools["resolve_data_quality_issue"].invoke(
        {
            "table_name": "payments",
            "column_name": "refund_status",
            "issue_type": "temporal_lag",
        }
    )
    assert "Error" in out
    assert "no active issue" in out


def test_two_factories_different_datasets_isolated(store):
    """Factories with different dataset_fingerprint are isolated."""
    f1 = make_data_quality_factory(store, "u1", "growth", FP_A)
    f2 = make_data_quality_factory(store, "u1", "growth", FP_B)
    t1 = _tools_by_name(f1.tools())
    t2 = _tools_by_name(f2.tools())

    t1["add_data_quality_issue"].invoke(
        {
            "table_name": "payments",
            "column_name": "refund_status",
            "issue_type": "temporal_lag",
            "issue_description": "dataset-A observation",
        }
    )
    list2 = t2["list_data_quality_issues_for_table"].invoke(
        {"table_name": "payments"}
    )
    assert "No active data-quality issues" in list2


def test_factory_rejects_empty_tenancy(store):
    with pytest.raises(ValueError):
        make_data_quality_factory(store, "", "p1", FP_A)
    with pytest.raises(ValueError):
        make_data_quality_factory(store, "u1", "", FP_A)
    with pytest.raises(ValueError):
        make_data_quality_factory(store, "u1", "p1", "")
