"""Pytest suite for the ``metric_definition`` memory module.

All tests run against an in-memory SQLite store (``:memory:``). The
store-level tests exercise :class:`MetricDefinitionStore` directly; the
tool-level tests exercise the LangChain ``@tool`` callables returned by
:func:`make_metric_factory(...).tools()`.
"""

from __future__ import annotations

import re
import sqlite3

import pytest

from mads.memory.metric_definition import (
    MetricDefinition,
    MetricDefinitionStore,
    _normalize,
)
from mads.memory.metric_tools import (
    MetricToolFactory,
    make_metric_factory,
)


# --------------------------------------------------------------------- fixtures


@pytest.fixture
def store():
    """Fresh in-memory MetricDefinitionStore per test."""
    s = MetricDefinitionStore(db_path=":memory:")
    try:
        yield s
    finally:
        s.close()


# ----------------------------------------------------------- helper assertions


_UUID_HEX_RE = re.compile(r"^[0-9a-f]{32}$")


def _tools_by_name(tools):
    return {t.name: t for t in tools}


def _seed(store, *, project_id="p1", user_id="u1", metric_name="Conversion",
          business_definition="orders / sessions",
          sql_definition="SELECT orders/sessions FROM events",
          python_definition=None,
          default_filters=None,
          confidence=1.0,
          source=None):
    return store.upsert(
        user_id=user_id,
        project_id=project_id,
        metric_name=metric_name,
        business_definition=business_definition,
        sql_definition=sql_definition,
        python_definition=python_definition,
        default_filters=default_filters,
        confidence=confidence,
        source=source,
    )


# =================================================================== NORMALIZATION


def test_normalize_examples():
    """1. The three spec-mandated normalization examples."""
    assert _normalize("Active User") == "active_user"
    assert _normalize("Conversion Rate (CR)") == "conversion_rate_cr"
    assert _normalize("  retention  ") == "retention"


def test_normalize_collapses_runs_and_strips_edges():
    """2. Punctuation/whitespace runs collapse to one ``_``; edges stripped."""
    assert _normalize("__Hello  --  World!!__") == "hello_world"
    assert _normalize("MRR$$") == "mrr"
    assert _normalize("a---b") == "a_b"


def test_normalize_empty_raises():
    """3. Empty string and pure-punctuation input both raise ValueError."""
    with pytest.raises(ValueError):
        _normalize("")
    with pytest.raises(ValueError):
        _normalize("!!!")
    with pytest.raises(ValueError):
        _normalize("   ")
    with pytest.raises(ValueError):
        _normalize("___")


# =================================================================== IDENTITY


def test_identity_case_insensitive_within_project(store):
    """4. 'Active User' and 'active_user' map to the same metric in the same project."""
    d1, a1 = _seed(store, metric_name="Active User",
                   business_definition="distinct user sessions in last 28d",
                   sql_definition="SELECT ... -- v1")
    assert a1 == "inserted"

    # Re-add under the alternate casing, identical defs -> noop.
    d2, a2 = _seed(store, metric_name="active_user",
                   business_definition="distinct user sessions in last 28d",
                   sql_definition="SELECT ... -- v1")
    assert a2 == "noop"
    assert d2.id == d1.id

    active = store.list_active(project_id="p1")
    assert len(active) == 1


def test_project_scoping_isolates_same_named_metric(store):
    """5. Two projects can each have a metric named 'Revenue' with different defs."""
    _seed(store, project_id="p1", metric_name="Revenue",
          business_definition="USD sum (project p1 specific)")
    _seed(store, project_id="p2", metric_name="Revenue",
          business_definition="EUR sum (project p2 specific)")

    p1 = store.get_active(project_id="p1", metric_name="Revenue")
    p2 = store.get_active(project_id="p2", metric_name="Revenue")
    assert p1 is not None and p2 is not None
    assert p1.id != p2.id
    assert "USD" in p1.business_definition
    assert "EUR" in p2.business_definition


# =================================================================== UPSERT ACTIONS


def test_first_insert_returns_inserted(store):
    """6. First call yields ``inserted`` and persists a row visible to get_active."""
    d, action = _seed(store)
    assert action == "inserted"
    assert _UUID_HEX_RE.match(d.id)
    assert isinstance(d, MetricDefinition)
    assert d.metric_name == "Conversion"
    assert d.normalized_name == "conversion"
    assert d.valid_to is None
    assert d.user_id == "u1"

    rt = store.get_active(project_id="p1", metric_name="Conversion")
    assert rt is not None and rt.id == d.id


def test_identical_reinsert_is_noop(store):
    """7. Second call with identical fields is a no-op; history length stays 1."""
    d1, _ = _seed(store)
    d2, action = _seed(store)
    assert action == "noop"
    assert d2.id == d1.id

    history = store.history(project_id="p1", metric_name="Conversion")
    assert len(history) == 1


def test_confidence_only_change_updates_in_place(store):
    """8. Changing only confidence updates the active row in place; history len = 1."""
    d1, _ = _seed(store, confidence=0.7)
    d2, action = _seed(store, confidence=0.95)
    assert action == "updated_in_place"
    assert d2.id == d1.id  # same row
    assert d2.confidence == pytest.approx(0.95)

    history = store.history(project_id="p1", metric_name="Conversion")
    assert len(history) == 1
    assert history[0].confidence == pytest.approx(0.95)


def test_source_only_change_updates_in_place(store):
    """9. Changing only source -> updated_in_place; history len = 1."""
    d1, _ = _seed(store, source="analytics_team")
    d2, action = _seed(store, source="finance_team")
    assert action == "updated_in_place"
    assert d2.id == d1.id
    assert d2.source == "finance_team"

    history = store.history(project_id="p1", metric_name="Conversion")
    assert len(history) == 1


def test_business_definition_change_versions(store):
    """10. Changing business_definition -> versioned; history len = 2."""
    d1, _ = _seed(store, business_definition="orders / sessions")
    d2, action = _seed(store, business_definition="orders / sessions (exclude bots)")
    assert action == "versioned"
    assert d2.id != d1.id

    history = store.history(project_id="p1", metric_name="Conversion")
    assert len(history) == 2
    assert history[0].id == d1.id
    assert history[1].id == d2.id
    assert history[0].valid_to is not None
    assert history[1].valid_to is None


def test_sql_definition_change_versions(store):
    """11. Changing only sql_definition (business unchanged) -> versioned."""
    _, _ = _seed(store, sql_definition="SELECT orders/sessions FROM events")
    d2, action = _seed(store,
                       sql_definition="SELECT orders/sessions FROM events WHERE NOT is_bot")
    assert action == "versioned"
    history = store.history(project_id="p1", metric_name="Conversion")
    assert len(history) == 2


def test_python_definition_change_versions(store):
    """12. Changing only python_definition -> versioned."""
    _, _ = _seed(store, python_definition="df.orders.sum() / df.sessions.sum()")
    d2, action = _seed(store, python_definition="df.query('not is_bot').pipe(rate)")
    assert action == "versioned"
    history = store.history(project_id="p1", metric_name="Conversion")
    assert len(history) == 2


def test_default_filters_change_versions(store):
    """13. Changing only default_filters -> versioned."""
    _, _ = _seed(store, default_filters="country='US'")
    d2, action = _seed(store, default_filters="country='US' AND tier='paid'")
    assert action == "versioned"
    history = store.history(project_id="p1", metric_name="Conversion")
    assert len(history) == 2


# =================================================================== INVARIANTS


def test_versioning_invariant_one_active_row(store):
    """14. After a versioned upsert, exactly one row has valid_to IS NULL."""
    _seed(store, business_definition="v1")
    _seed(store, business_definition="v2")
    _seed(store, business_definition="v3")

    cnt = store._conn.execute(
        """
        SELECT COUNT(*) AS c FROM metric_memory
         WHERE project_id = ? AND normalized_name = ? AND valid_to IS NULL
        """,
        ("p1", "conversion"),
    ).fetchone()["c"]
    assert cnt == 1

    total = store._conn.execute(
        """
        SELECT COUNT(*) AS c FROM metric_memory
         WHERE project_id = ? AND normalized_name = ?
        """,
        ("p1", "conversion"),
    ).fetchone()["c"]
    assert total == 3


def test_versioning_invariant_valid_to_equals_next_valid_from(store):
    """15. Closed row's valid_to equals next row's valid_from (single-tx writer)."""
    _seed(store, business_definition="v1")
    _seed(store, business_definition="v2")

    history = store.history(project_id="p1", metric_name="Conversion")
    assert len(history) == 2
    # Implementation uses the same `now` for both writes inside one
    # transaction, so they must be exactly equal.
    assert history[0].valid_to == history[1].valid_from


def test_partial_unique_index_prevents_two_active_rows(store):
    """16. Manual INSERT bypassing upsert is rejected by the partial unique index."""
    _seed(store, metric_name="DupGuard", business_definition="v1")

    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute(
            """
            INSERT INTO metric_memory (
                id, user_id, project_id, metric_name, normalized_name,
                business_definition, sql_definition, python_definition,
                default_filters, valid_from, valid_to, confidence, source,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL,
                      CURRENT_TIMESTAMP, NULL, 1.0, NULL,
                      CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            (
                "00000000000000000000000000000002",
                "u1", "p1", "DupGuard", "dupguard",
                "rival v1",
            ),
        )


# =================================================================== RETRIEVAL


def test_get_active_returns_none_when_absent(store):
    """17. get_active returns None when no metric exists."""
    assert store.get_active(project_id="p1", metric_name="ghost") is None


def test_list_active_excludes_closed_rows(store):
    """18. list_active excludes versioned (closed) rows."""
    _seed(store, metric_name="A", business_definition="a1")
    _seed(store, metric_name="A", business_definition="a2")  # versions A
    _seed(store, metric_name="B", business_definition="b1")

    rows = store.list_active(project_id="p1")
    names = [r.metric_name for r in rows]
    assert names == ["A", "B"]  # case-insensitive alpha order
    # Total rows in table: 3 (2 for A, 1 for B). list_active sees 2.
    total = store._conn.execute(
        "SELECT COUNT(*) AS c FROM metric_memory WHERE project_id = ?", ("p1",)
    ).fetchone()["c"]
    assert total == 3


def test_search_matches_name_and_definition(store):
    """19. search_active hits both metric_name and business_definition substrings."""
    _seed(store, metric_name="Conversion",
          business_definition="orders divided by sessions")
    _seed(store, metric_name="Revenue",
          business_definition="sum of USD across all conversions")
    _seed(store, metric_name="Retention",
          business_definition="active users in week N+1 / week N")

    hits = store.search_active(project_id="p1", query="conv")
    names = sorted(h.metric_name for h in hits)
    # 'Conversion' hits via name; 'Revenue' hits via "conversions" in body.
    assert names == ["Conversion", "Revenue"]

    none = store.search_active(project_id="p1", query="xyzqq")
    assert none == []


def test_history_oldest_first_includes_active_and_closed(store):
    """20. history returns oldest-first; closed rows have valid_to set."""
    _seed(store, business_definition="v1")
    _seed(store, business_definition="v2")
    _seed(store, business_definition="v3")

    history = store.history(project_id="p1", metric_name="Conversion")
    assert [m.business_definition for m in history] == ["v1", "v2", "v3"]
    assert history[0].valid_to is not None
    assert history[1].valid_to is not None
    assert history[2].valid_to is None


# =================================================================== TENANCY


def test_project_isolation_get(store):
    """21. A metric defined in project A is invisible to a get_active in project B."""
    _seed(store, project_id="A", metric_name="Revenue",
          business_definition="A-revenue")
    assert store.get_active(project_id="B", metric_name="Revenue") is None
    assert store.list_active(project_id="B") == []


def test_users_share_project_library(store):
    """22. Two users in the same project see the same metric library."""
    _seed(store, user_id="alice", project_id="growth",
          metric_name="Conversion", business_definition="orders/sessions")

    f_alice = make_metric_factory(store, "alice", "growth")
    f_bob = make_metric_factory(store, "bob", "growth")
    a = _tools_by_name(f_alice.tools())
    b = _tools_by_name(f_bob.tools())

    out_a = a["get_metric_definition"].invoke({"metric_name": "Conversion"})
    out_b = b["get_metric_definition"].invoke({"metric_name": "Conversion"})
    assert "orders/sessions" in out_a
    assert "orders/sessions" in out_b

    # Bob edits the metric -> Alice sees the change.
    b["add_metric_definition"].invoke(
        {
            "metric_name": "Conversion",
            "business_definition": "orders/sessions (no bots)",
        }
    )
    out_a2 = a["get_metric_definition"].invoke({"metric_name": "Conversion"})
    assert "no bots" in out_a2


# =================================================================== VALIDATION


def test_empty_metric_name_raises(store):
    """23. metric_name='' raises ValueError at the store layer."""
    with pytest.raises(ValueError):
        _seed(store, metric_name="", business_definition="x")


def test_unnormalizable_metric_name_raises(store):
    """24. metric_name='!!!' (normalizes to empty) raises ValueError."""
    with pytest.raises(ValueError):
        _seed(store, metric_name="!!!", business_definition="x")


def test_empty_business_definition_raises(store):
    """25. business_definition='' raises ValueError."""
    with pytest.raises(ValueError):
        _seed(store, business_definition="")


def test_confidence_out_of_range_raises(store):
    """26. confidence > 1.0 raises ValueError before the DB sees it."""
    with pytest.raises(ValueError):
        _seed(store, confidence=1.5)
    with pytest.raises(ValueError):
        _seed(store, confidence=-0.1)


# =================================================================== TOOL LAYER


def test_factory_returns_exactly_five_tools(store):
    """27. Factory yields exactly 5 tools with the contracted names."""
    f = make_metric_factory(store, "u1", "p1")
    assert isinstance(f, MetricToolFactory)
    tools = f.tools()
    assert len(tools) == 5
    names = sorted(t.name for t in tools)
    assert names == [
        "add_metric_definition",
        "get_metric_definition",
        "get_metric_history",
        "list_metric_definitions",
        "search_metric_definitions",
    ]


def test_tool_signatures_hide_tenancy(store):
    """28. None of the 5 tools accept user_id or project_id as arguments."""
    tools = make_metric_factory(store, "u1", "p1").tools()
    for t in tools:
        schema = t.args_schema.model_json_schema()
        props = set(schema.get("properties", {}).keys())
        assert "user_id" not in props, (
            f"{t.name} leaks user_id in its arg schema: {props}"
        )
        assert "project_id" not in props, (
            f"{t.name} leaks project_id in its arg schema: {props}"
        )


def test_add_metric_definition_tool_inserts_then_noops(store):
    """29. add_metric_definition: first call mentions inserted; second call mentions noop."""
    tools = _tools_by_name(make_metric_factory(store, "u1", "p1").tools())

    out1 = tools["add_metric_definition"].invoke(
        {
            "metric_name": "Active User",
            "business_definition": "distinct user sessions in last 28d",
            "sql_definition": "SELECT COUNT(DISTINCT user_id) FROM sessions",
        }
    )
    assert isinstance(out1, str)
    assert "Active User" in out1
    assert "inserted" in out1
    assert "version 1" in out1

    out2 = tools["add_metric_definition"].invoke(
        {
            "metric_name": "Active User",
            "business_definition": "distinct user sessions in last 28d",
            "sql_definition": "SELECT COUNT(DISTINCT user_id) FROM sessions",
        }
    )
    assert "noop" in out2


def test_add_metric_definition_tool_versioned_label(store):
    """30. add_metric_definition: editing business_definition reports versioned + version 2."""
    tools = _tools_by_name(make_metric_factory(store, "u1", "p1").tools())
    tools["add_metric_definition"].invoke(
        {"metric_name": "Conversion", "business_definition": "v1 def"}
    )
    out = tools["add_metric_definition"].invoke(
        {"metric_name": "Conversion", "business_definition": "v2 def"}
    )
    assert "versioned" in out
    assert "version 2" in out


def test_get_metric_definition_tool_empty_state(store):
    """31. get_metric_definition for an unknown metric returns the empty-state string."""
    tools = _tools_by_name(make_metric_factory(store, "u1", "p1").tools())
    out = tools["get_metric_definition"].invoke({"metric_name": "ghost"})
    assert isinstance(out, str)
    assert "ghost" in out
    assert "No metric" in out


def test_list_and_search_tools_empty_then_populated(store):
    """32. list / search tools return empty-state strings, then a compact table once populated."""
    tools = _tools_by_name(make_metric_factory(store, "u1", "p1").tools())

    empty_list = tools["list_metric_definitions"].invoke({})
    assert "No metrics defined" in empty_list

    empty_search = tools["search_metric_definitions"].invoke({"query": "conv"})
    assert "No metrics matched" in empty_search

    tools["add_metric_definition"].invoke(
        {"metric_name": "Conversion", "business_definition": "orders/sessions"}
    )
    tools["add_metric_definition"].invoke(
        {"metric_name": "Revenue", "business_definition": "USD total"}
    )

    listed = tools["list_metric_definitions"].invoke({})
    assert "Conversion" in listed
    assert "Revenue" in listed
    # Compact header row present.
    assert "name | business_definition" in listed

    searched = tools["search_metric_definitions"].invoke({"query": "conv"})
    assert "Conversion" in searched
    assert "Revenue" not in searched


def test_history_tool_shows_authors_and_ordering(store):
    """33. get_metric_history shows oldest-first, including who edited each version."""
    f_alice = make_metric_factory(store, "alice", "p1")
    f_bob = make_metric_factory(store, "bob", "p1")
    a = _tools_by_name(f_alice.tools())
    b = _tools_by_name(f_bob.tools())

    a["add_metric_definition"].invoke(
        {"metric_name": "Conversion", "business_definition": "v1"}
    )
    b["add_metric_definition"].invoke(
        {"metric_name": "Conversion", "business_definition": "v2"}
    )

    out = a["get_metric_history"].invoke({"metric_name": "Conversion"})
    assert "v1" in out
    assert "v2" in out
    assert "alice" in out
    assert "bob" in out
    # v1 must appear before v2 in the rendered string (oldest-first).
    assert out.index("v1") < out.index("v2")
    assert "2 version" in out


def test_two_factories_different_projects_dont_see_each_other(store):
    """34. Factories with different project_id pointed at the same store are isolated."""
    f1 = make_metric_factory(store, "u1", "growth")
    f2 = make_metric_factory(store, "u1", "marketing")
    t1 = _tools_by_name(f1.tools())
    t2 = _tools_by_name(f2.tools())

    t1["add_metric_definition"].invoke(
        {"metric_name": "Conversion", "business_definition": "growth-team def"}
    )
    list2 = t2["list_metric_definitions"].invoke({})
    assert "Conversion" not in list2
    assert "No metrics defined" in list2

    get2 = t2["get_metric_definition"].invoke({"metric_name": "Conversion"})
    assert "No metric" in get2


def test_search_empty_query_returns_error_string(store):
    """35. search_metric_definitions with empty string returns an error message, not all rows."""
    tools = _tools_by_name(make_metric_factory(store, "u1", "p1").tools())
    tools["add_metric_definition"].invoke(
        {"metric_name": "Conversion", "business_definition": "orders/sessions"}
    )
    out = tools["search_metric_definitions"].invoke({"query": ""})
    assert "Error" in out
    # And does NOT dump every row as a side effect.
    assert "Conversion" not in out


def test_factory_rejects_empty_tenancy(store):
    """36. make_metric_factory rejects empty user_id / project_id."""
    with pytest.raises(ValueError):
        make_metric_factory(store, "", "p1")
    with pytest.raises(ValueError):
        make_metric_factory(store, "u1", "")


# =================================================================== B3: embeddings
# Tests for the embedding-augmented retrieval surface added in B3:
# - embedder=None graceful degradation (covered implicitly above)
# - vec INSERT on add
# - vec DELETE on versioning close
# - backfill_embeddings for migrating pre-B3 databases


def _make_store_with_embedder(tmp_path):
    """File-backed store with a HashEmbedder wired so vec writes fire.

    A file-backed (not ``:memory:``) DB keeps the test path closest to
    production conditions and exercises the WAL pragma + sqlite-vec
    extension load in the constructor.
    """
    from mads.memory.embedding import HashEmbedder

    db = str(tmp_path / "metric_b3.sqlite")
    return MetricDefinitionStore(db_path=db, embedder=HashEmbedder())


def test_store_without_embedder_skips_vec_table(store):
    """37. embedder=None: no metric_vec table is created (graceful degradation)."""
    # The base fixture has embedder=None.
    rows = store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='metric_vec'"
    ).fetchall()
    assert rows == [], "metric_vec should not exist when no embedder is wired"
    # And upsert still works (no embedding writes attempted).
    md, action = store.upsert(
        user_id="u1",
        project_id="p1",
        metric_name="net_revenue",
        business_definition="USD total",
    )
    assert action == "inserted"
    assert md.metric_name == "net_revenue"


def test_store_with_embedder_creates_vec_table(tmp_path):
    """38. embedder=HashEmbedder: metric_vec exists and INSERT writes a vec row."""
    store = _make_store_with_embedder(tmp_path)
    try:
        # metric_vec virtual table must exist.
        rows = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE name='metric_vec'"
        ).fetchall()
        assert len(rows) >= 1

        md, _ = store.upsert(
            user_id="u1",
            project_id="p1",
            metric_name="qualified_customer",
            business_definition="tenure > 12 AND MonthlyCharges > 50",
        )
        # The vec row sits at the same rowid as the metric row.
        row = store._conn.execute(
            "SELECT rowid FROM metric_memory WHERE id = ?", (md.id,)
        ).fetchone()
        assert row is not None
        vec = store._conn.execute(
            "SELECT rowid FROM metric_vec WHERE rowid = ?", (row["rowid"],)
        ).fetchone()
        assert vec is not None, "expected a metric_vec row paired with the metric"
    finally:
        store.close()


def test_versioning_deletes_old_vec_row(tmp_path):
    """39. Closing a versioned metric row also DELETEs its embedding."""
    store = _make_store_with_embedder(tmp_path)
    try:
        md_v1, _ = store.upsert(
            user_id="u1",
            project_id="p1",
            metric_name="net_revenue",
            business_definition="v1: USD total",
        )
        old_rowid = store._conn.execute(
            "SELECT rowid FROM metric_memory WHERE id = ?", (md_v1.id,)
        ).fetchone()["rowid"]

        # Trigger a versioning close.
        md_v2, action = store.upsert(
            user_id="u1",
            project_id="p1",
            metric_name="net_revenue",
            business_definition="v2: gross - refunds",
        )
        assert action == "versioned"

        # Old vec row is gone, new vec row exists at v2's rowid.
        gone = store._conn.execute(
            "SELECT rowid FROM metric_vec WHERE rowid = ?", (old_rowid,)
        ).fetchone()
        assert gone is None, "expected old vec row to be removed"
        new_rowid = store._conn.execute(
            "SELECT rowid FROM metric_memory WHERE id = ?", (md_v2.id,)
        ).fetchone()["rowid"]
        present = store._conn.execute(
            "SELECT rowid FROM metric_vec WHERE rowid = ?", (new_rowid,)
        ).fetchone()
        assert present is not None
    finally:
        store.close()


def test_in_place_update_does_not_touch_vec(tmp_path):
    """40. confidence/source-only update doesn't perturb the vec row."""
    store = _make_store_with_embedder(tmp_path)
    try:
        md, _ = store.upsert(
            user_id="u1",
            project_id="p1",
            metric_name="conversion",
            business_definition="orders/sessions",
        )
        rowid = store._conn.execute(
            "SELECT rowid FROM metric_memory WHERE id = ?", (md.id,)
        ).fetchone()["rowid"]
        # snapshot the vector bytes.
        before = store._conn.execute(
            "SELECT embedding FROM metric_vec WHERE rowid = ?", (rowid,)
        ).fetchone()["embedding"]

        _, action = store.upsert(
            user_id="u1",
            project_id="p1",
            metric_name="conversion",
            business_definition="orders/sessions",
            confidence=0.7,
            source="analyst",
        )
        assert action == "updated_in_place"
        after = store._conn.execute(
            "SELECT embedding FROM metric_vec WHERE rowid = ?", (rowid,)
        ).fetchone()["embedding"]
        assert before == after, "in-place metadata edit should not re-embed"
    finally:
        store.close()


def test_backfill_embeddings_no_op_without_embedder(store):
    """41. backfill is a no-op when no embedder is wired."""
    store.upsert(
        user_id="u1", project_id="p1",
        metric_name="x", business_definition="d",
    )
    assert store.backfill_embeddings() == 0


def test_backfill_embeddings_migrates_pre_b3_rows(tmp_path):
    """42. backfill_embeddings populates metric_vec for rows that pre-date the embedder.

    We construct a store WITHOUT an embedder, write a few rows (no vec
    table exists), close, then reopen WITH an embedder. The reopened
    store's init_schema creates metric_vec; backfill_embeddings populates
    it for the active rows.
    """
    from mads.memory.embedding import HashEmbedder

    db = str(tmp_path / "metric_backfill.sqlite")
    s_old = MetricDefinitionStore(db_path=db)
    s_old.upsert(
        user_id="u1", project_id="p1",
        metric_name="metric_a", business_definition="a-def",
    )
    s_old.upsert(
        user_id="u1", project_id="p1",
        metric_name="metric_b", business_definition="b-def",
    )
    # Close one of them via a versioning edit; the closed row should
    # NOT receive a backfilled embedding (vec table tracks active only).
    s_old.upsert(
        user_id="u1", project_id="p1",
        metric_name="metric_a", business_definition="a-def-v2",
    )
    s_old.close()

    s_new = MetricDefinitionStore(db_path=db, embedder=HashEmbedder())
    try:
        n = s_new.backfill_embeddings()
        # Two active rows (metric_a v2 + metric_b); historic metric_a v1 skipped.
        assert n == 2
        # Idempotent: re-running backfills 0 more rows.
        assert s_new.backfill_embeddings() == 0
        # Vec table holds exactly the two active row ids.
        vec_rowids = [
            r["rowid"] for r in s_new._conn.execute(
                "SELECT rowid FROM metric_vec ORDER BY rowid"
            ).fetchall()
        ]
        active_rowids = [
            r["rowid"] for r in s_new._conn.execute(
                "SELECT rowid FROM metric_memory "
                "WHERE valid_to IS NULL ORDER BY rowid"
            ).fetchall()
        ]
        assert vec_rowids == active_rowids
    finally:
        s_new.close()


def test_find_similar_returns_empty_without_embedder(store):
    """43. find_similar returns [] when no embedder is wired."""
    store.upsert(
        user_id="u1", project_id="p1",
        metric_name="conversion", business_definition="orders/sessions",
    )
    assert store.find_similar(project_id="p1", query="rate") == []


def test_find_similar_returns_hits_with_embedder(tmp_path):
    """44. find_similar surfaces stored metrics with sqlite-vec distance."""
    store = _make_store_with_embedder(tmp_path)
    try:
        store.upsert(
            user_id="u1", project_id="p1",
            metric_name="conversion", business_definition="orders/sessions",
        )
        hits = store.find_similar(project_id="p1", query="conversion", k=5)
        assert len(hits) == 1
        metric, distance = hits[0]
        assert metric.metric_name == "conversion"
        assert isinstance(distance, float)
    finally:
        store.close()


def test_search_metric_definitions_uses_retriever_for_synonym(tmp_path):
    """45. search_metric_definitions routes through the MetricRetriever
    so semantic synonyms surface in the tool's output (closes the gap
    where 'qualified customers' couldn't find the stored 'qualified_customer').

    Uses the real SentenceTransformerEmbedder; skip when unavailable
    (offline CI). The HashEmbedder test wouldn't be a meaningful
    semantic test (no synonym signal in token-bag space)."""
    try:
        from mads.memory.embedding import SentenceTransformerEmbedder

        emb = SentenceTransformerEmbedder()
        emb.embed("warmup")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"SentenceTransformerEmbedder unavailable: {exc!r}")

    store = MetricDefinitionStore(
        db_path=str(tmp_path / "metric_tool_b3.sqlite"),
        embedder=emb,
    )
    try:
        store.upsert(
            user_id="u1",
            project_id="p1",
            metric_name="qualified_customer",
            business_definition=(
                "A customer who has been with the company for more than 12 "
                "months (tenure > 12) AND pays more than $50/month."
            ),
        )
        tools = _tools_by_name(make_metric_factory(store, "u1", "p1").tools())
        out = tools["search_metric_definitions"].invoke(
            {"query": "qualified customers"}
        )
        # The match row should mention the stored metric name.
        assert "qualified_customer" in out, f"output missing metric: {out!r}"
        # And expose a score column in the new compact format.
        assert "score" in out.split("\n")[1] or "score" in out
    finally:
        store.close()


# =================================================================== v1.4: required_columns
# Schema-coupling against cat 2 (column_memory + bound df). The
# required_columns_json column persists either:
#   * NULL (unknown — assume applicable; legacy and prose-only rows)
#   * "[]" (explicit no-columns-required, e.g., a row-count metric)
#   * '["a","b"]' (applicability requires {a,b} ⊆ df.columns)


def test_required_columns_explicit_insert_and_roundtrip(store):
    """46. required_columns=['tenure','MonthlyCharges'] roundtrips via from_row."""
    md, action = store.upsert(
        user_id="u1",
        project_id="p1",
        metric_name="qualified_customer",
        business_definition="tenure > 12 AND MonthlyCharges > 50",
        required_columns=["tenure", "MonthlyCharges"],
    )
    assert action == "inserted"
    assert md.required_columns == ["tenure", "MonthlyCharges"]

    rt = store.get_active(project_id="p1", metric_name="qualified_customer")
    assert rt is not None
    assert rt.required_columns == ["tenure", "MonthlyCharges"]


def test_required_columns_none_with_no_sql_persists_null(store):
    """47. omit required_columns + empty sql_definition -> persists NULL."""
    md, _ = store.upsert(
        user_id="u1",
        project_id="p1",
        metric_name="active_user",
        business_definition="made a payment in the last 30 days",
    )
    assert md.required_columns is None
    raw = store._conn.execute(
        "SELECT required_columns_json FROM metric_memory WHERE id = ?", (md.id,)
    ).fetchone()
    assert raw["required_columns_json"] is None


def test_required_columns_empty_list_persists_explicit_empty(store):
    """48. required_columns=[] persists '[]' and round-trips to [] (NOT None)."""
    md, _ = store.upsert(
        user_id="u1",
        project_id="p1",
        metric_name="row_count",
        business_definition="total number of rows",
        required_columns=[],
    )
    assert md.required_columns == []
    raw = store._conn.execute(
        "SELECT required_columns_json FROM metric_memory WHERE id = ?", (md.id,)
    ).fetchone()
    assert raw["required_columns_json"] == "[]"


def test_required_columns_autoparsed_from_sql(store):
    """49. omit required_columns + non-empty sql -> auto-parsed."""
    md, _ = store.upsert(
        user_id="u1",
        project_id="p1",
        metric_name="loyal_customer",
        business_definition="long-tenure two-year contract customers",
        sql_definition="tenure > 36 AND Contract = 'Two year'",
    )
    # Parser yields ['tenure', 'Contract'] (the 'Two year' literal is stripped).
    assert md.required_columns == ["tenure", "Contract"]


def test_required_columns_explicit_overrides_autoparse(store):
    """50. Explicit required_columns wins over what the parser would have inferred."""
    md, _ = store.upsert(
        user_id="u1",
        project_id="p1",
        metric_name="exotic_filter",
        business_definition="custom",
        sql_definition="tenure > 12",
        required_columns=["last_payment_date"],  # NOT in sql
    )
    assert md.required_columns == ["last_payment_date"]


def test_required_columns_change_triggers_version(store):
    """51. Changing required_columns versions the metric (definitional surface)."""
    _seed(
        store,
        metric_name="m",
        business_definition="def",
        sql_definition="tenure > 12",
    )
    # Re-upsert with explicit required_columns; the auto-parse would have
    # set ['tenure'], so an explicit ['tenure','contract'] is different.
    _, action = store.upsert(
        user_id="u1",
        project_id="p1",
        metric_name="m",
        business_definition="def",
        sql_definition="tenure > 12",
        required_columns=["tenure", "contract"],
    )
    assert action == "versioned"
    history = store.history(project_id="p1", metric_name="m")
    assert len(history) == 2
    assert history[1].required_columns == ["tenure", "contract"]


def test_required_columns_invalid_type_raises(store):
    """52. required_columns must be list[str] or None."""
    import pytest

    with pytest.raises(ValueError):
        store.upsert(
            user_id="u1",
            project_id="p1",
            metric_name="m",
            business_definition="d",
            required_columns="not a list",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        store.upsert(
            user_id="u1",
            project_id="p1",
            metric_name="m",
            business_definition="d",
            required_columns=[1, 2, 3],  # type: ignore[list-item]
        )


def test_is_applicable_none_means_unknown_assume_yes(store):
    """53. is_applicable returns (True, []) when required_columns is None."""
    md, _ = store.upsert(
        user_id="u1",
        project_id="p1",
        metric_name="legacy_metric",
        business_definition="prose-only, no columns",
    )
    applicable, missing = store.is_applicable(md, {"tenure"})
    assert applicable is True
    assert missing == []


def test_is_applicable_empty_list_means_no_columns_required(store):
    """54. is_applicable returns (True, []) for explicit empty required_columns."""
    md, _ = store.upsert(
        user_id="u1",
        project_id="p1",
        metric_name="row_count",
        business_definition="just count rows",
        required_columns=[],
    )
    applicable, missing = store.is_applicable(md, set())
    assert applicable is True
    assert missing == []


def test_is_applicable_subset_match(store):
    """55. is_applicable returns (True, []) when required ⊆ available."""
    md, _ = store.upsert(
        user_id="u1",
        project_id="p1",
        metric_name="qualified_customer",
        business_definition="filter",
        required_columns=["tenure", "MonthlyCharges"],
    )
    applicable, missing = store.is_applicable(
        md, {"tenure", "MonthlyCharges", "Churn"}
    )
    assert applicable is True
    assert missing == []


def test_is_applicable_missing_column(store):
    """56. is_applicable surfaces missing columns when required ⊄ available."""
    md, _ = store.upsert(
        user_id="u1",
        project_id="p1",
        metric_name="active_user",
        business_definition="paid recently",
        required_columns=["last_payment_date", "user_id"],
    )
    applicable, missing = store.is_applicable(
        md, {"tenure", "Churn", "user_id"}
    )
    assert applicable is False
    # Sorted, missing only 'last_payment_date'.
    assert missing == ["last_payment_date"]


def test_is_applicable_accepts_list_as_available(store):
    """57. is_applicable tolerates list input as well as set (call-site sugar)."""
    md, _ = store.upsert(
        user_id="u1",
        project_id="p1",
        metric_name="m",
        business_definition="d",
        required_columns=["a"],
    )
    ok, missing = store.is_applicable(md, ["a", "b", "c"])  # type: ignore[arg-type]
    assert ok is True and missing == []


def test_backfill_required_columns_populates_legacy_rows(store):
    """58. backfill_required_columns parses sql for rows that left it NULL."""
    # First write inserts a row WITH auto-parse (sql_definition has tokens).
    # We then force the row back to NULL to simulate a legacy row, then
    # backfill should re-fill it.
    md, _ = store.upsert(
        user_id="u1",
        project_id="p1",
        metric_name="m_with_sql",
        business_definition="d",
        sql_definition="x > 1 AND y < 2",
    )
    assert md.required_columns == ["x", "y"]
    store._conn.execute(
        "UPDATE metric_memory SET required_columns_json = NULL WHERE id = ?",
        (md.id,),
    )

    # And insert a prose-only row (no sql) — backfill should leave it.
    md_prose, _ = store.upsert(
        user_id="u1",
        project_id="p1",
        metric_name="prose_only",
        business_definition="no sql here",
    )
    assert md_prose.required_columns is None

    n = store.backfill_required_columns()
    assert n == 1
    rt = store.get_active(project_id="p1", metric_name="m_with_sql")
    assert rt is not None and rt.required_columns == ["x", "y"]
    rt_prose = store.get_active(project_id="p1", metric_name="prose_only")
    assert rt_prose is not None and rt_prose.required_columns is None

    # Idempotent: second call is a no-op.
    assert store.backfill_required_columns() == 0


def test_add_metric_definition_tool_accepts_required_columns(store):
    """59. The LLM-visible add_metric_definition tool exposes required_columns."""
    f = make_metric_factory(store, "u1", "p1")
    tools = _tools_by_name(f.tools())
    schema = tools["add_metric_definition"].args_schema.model_json_schema()
    props = set(schema.get("properties", {}).keys())
    assert "required_columns" in props, (
        f"add_metric_definition is missing required_columns in its arg schema: "
        f"{props}"
    )
    # Tenancy still hidden.
    assert "user_id" not in props
    assert "project_id" not in props

    out = tools["add_metric_definition"].invoke(
        {
            "metric_name": "qualified_customer",
            "business_definition": "tenure-and-charge filter",
            "required_columns": ["tenure", "MonthlyCharges"],
        }
    )
    assert "inserted" in out
    rt = store.get_active(
        project_id="p1", metric_name="qualified_customer"
    )
    assert rt is not None
    assert rt.required_columns == ["tenure", "MonthlyCharges"]
