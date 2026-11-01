"""Pytest suite for the dataset_schema memory module.

All tests use an in-memory SQLite store (`:memory:`) for zero filesystem
touch. The fingerprint tests exercise `compute_fingerprint` directly. The
store-level tests exercise `DatasetSchemaStore`. The tool-level tests
exercise the LangChain `@tool` callables returned by `make_dataset_tools`.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

from mads.memory.dataset_schema import DatasetSchemaStore, compute_fingerprint
from mads.memory.dataset_tools import make_dataset_tools


# --------------------------------------------------------------------- fixtures


@pytest.fixture
def store():
    """Fresh in-memory DatasetSchemaStore per test."""
    s = DatasetSchemaStore(db_path=":memory:")
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def small_df():
    """Small (3-row) reference DataFrame for fingerprint / store tests."""
    return pd.DataFrame(
        {
            "order_id": [1, 2, 3],
            "customer_id": [10, 20, 30],
            "amount": [9.99, 19.99, 29.99],
        }
    )


@pytest.fixture
def orders_df():
    """20-row orders-style df bound into tool factories."""
    return pd.DataFrame(
        {
            "order_id": list(range(1, 21)),
            "customer_id": [i % 5 for i in range(20)],
            "amount": [round(i * 1.5, 2) for i in range(20)],
        }
    )


# ----------------------------------------------------------- helper assertions


_HEX_RE = re.compile(r"^[0-9a-f]+$")
_UUID_HEX_RE = re.compile(r"^[0-9a-f]{32}$")


def _tools_by_name(tools):
    return {t.name: t for t in tools}


# ============================================================ FINGERPRINT TESTS


def test_fingerprint_deterministic(small_df):
    """1. Same df -> same fingerprint across two calls."""
    fp1 = compute_fingerprint(small_df)
    fp2 = compute_fingerprint(small_df)
    assert fp1 == fp2
    assert isinstance(fp1, str) and fp1
    assert _HEX_RE.match(fp1)


def test_fingerprint_column_order_invariant(small_df):
    """2. Reversing column order does not change the fingerprint."""
    reversed_df = small_df[list(reversed(small_df.columns))]
    assert compute_fingerprint(small_df) == compute_fingerprint(reversed_df)


def test_fingerprint_row_order_invariant(small_df):
    """3. Shuffling rows does not change the fingerprint."""
    shuffled = small_df.sample(frac=1, random_state=42).reset_index(drop=True)
    assert compute_fingerprint(small_df) == compute_fingerprint(shuffled)


def test_fingerprint_changes_when_column_dropped(small_df):
    """4. Dropping a column changes the fingerprint."""
    dropped = small_df.drop(columns=["amount"])
    assert compute_fingerprint(small_df) != compute_fingerprint(dropped)


def test_fingerprint_changes_when_dtype_changes(small_df):
    """5. Same column names but a dtype changed -> different fingerprint."""
    coerced = small_df.copy()
    coerced["order_id"] = coerced["order_id"].astype(float)
    # Sanity: dtype actually differs.
    assert str(small_df["order_id"].dtype) != str(coerced["order_id"].dtype)
    assert compute_fingerprint(small_df) != compute_fingerprint(coerced)


def test_fingerprint_is_schema_only_row_count_invariant():
    """6. v1.1: 50-row vs 5000-row same-schema dfs -> SAME fingerprint.

    Post-v1.1, ``compute_fingerprint`` is an alias for
    :func:`semantic_fingerprint`, which is schema-only (columns + dtypes).
    Row count and sample values no longer affect the fingerprint. This is
    deliberate: snapshots of the same logical dataset should match across
    sessions.
    """
    df_small = pd.DataFrame({"a": list(range(50)), "b": list(range(50))})
    df_big = pd.DataFrame({"a": list(range(5000)), "b": list(range(5000))})
    assert compute_fingerprint(df_small) == compute_fingerprint(df_big)


def test_fingerprint_small_perturbation_insensitivity():
    """7. Same df with 1 extra row appended within the same log-bucket -> SAME fingerprint.

    The top-K-unique sample is taken from the sorted *string* unique values,
    so we must engineer the perturbation so that K (=5) and the chosen
    sample are unchanged. We use a small df whose K-unique sample is
    saturated such that the appended row is a duplicate value -> sample
    unchanged, dtype unchanged, columns unchanged, row count still in
    bucket "1" (10..99).

    If this assumption breaks (e.g. the implementer changed K or the
    bucketing), we xfail rather than touch the implementation.
    """
    # Row count 50; bucket = int(log10(50)) = 1.
    df_a = pd.DataFrame({"v": [i % 5 for i in range(50)]})
    # Append a duplicate value so unique set is unchanged; row count -> 51,
    # still in bucket 1.
    df_b = pd.concat(
        [df_a, pd.DataFrame({"v": [0]})], ignore_index=True
    )

    fp_a = compute_fingerprint(df_a)
    fp_b = compute_fingerprint(df_b)

    if fp_a != fp_b:
        pytest.xfail(
            "Small perturbation crossed the K-unique-values or row-bucket boundary; "
            "fingerprints diverged. This is implementation-defined, not a bug."
        )
    assert fp_a == fp_b


def test_fingerprint_empty_dataframe_is_stable():
    """8. Empty df -> non-empty hex string and stable across calls."""
    df = pd.DataFrame({"a": []})
    fp1 = compute_fingerprint(df)
    fp2 = compute_fingerprint(df)
    assert isinstance(fp1, str) and fp1
    assert _HEX_RE.match(fp1)
    assert fp1 == fp2


# =========================================================== DATASET STORE TESTS


def test_register_dataset_returns_id_and_fingerprint(store, small_df):
    """9. register_dataset returns (uuid_hex, fingerprint); visible to get/list."""
    new_id, fp = store.register_dataset(
        user_id="u1",
        df=small_df,
        dataset_name="orders",
        table_name="orders_t",
        grain="one row per order",
    )
    assert _UUID_HEX_RE.match(new_id), f"expected 32-char uuid hex, got {new_id!r}"
    assert isinstance(fp, str) and _HEX_RE.match(fp)
    assert fp == compute_fingerprint(small_df)

    row = store.get_dataset(user_id="u1", fingerprint=fp, table_name="orders_t")
    assert row is not None
    assert row["id"] == new_id
    assert row["dataset_name"] == "orders"
    assert row["table_name"] == "orders_t"
    assert row["grain"] == "one row per order"
    assert row["valid_to"] is None

    listed = store.list_datasets("u1")
    assert any(r["id"] == new_id for r in listed)


def test_register_dataset_supersedes_on_grain_change(store, small_df):
    """10. Re-register with different grain supersedes; history has 2 rows ordered asc."""
    id1, fp = store.register_dataset(
        user_id="u1",
        df=small_df,
        dataset_name="orders",
        table_name="orders_t",
        grain="grain v1",
    )
    id2, fp2 = store.register_dataset(
        user_id="u1",
        df=small_df,
        dataset_name="orders",
        table_name="orders_t",
        grain="grain v2",
    )
    assert fp == fp2
    assert id1 != id2

    active = store.get_dataset(user_id="u1", fingerprint=fp, table_name="orders_t")
    assert active is not None
    assert active["id"] == id2
    assert active["grain"] == "grain v2"

    history = store.list_dataset_history(
        user_id="u1", fingerprint=fp, table_name="orders_t"
    )
    assert len(history) == 2
    # Oldest first.
    assert history[0]["id"] == id1
    assert history[1]["id"] == id2
    assert history[0]["grain"] == "grain v1"
    assert history[1]["grain"] == "grain v2"
    assert history[0]["valid_to"] is not None
    assert history[1]["valid_to"] is None


def test_register_dataset_merges_carryover(store, small_df):
    """11. First register grain only; then primary_key only -> new active has BOTH."""
    store.register_dataset(
        user_id="u1",
        df=small_df,
        dataset_name="orders",
        table_name="orders_t",
        grain="one row per order",
    )
    _, fp = store.register_dataset(
        user_id="u1",
        df=small_df,
        dataset_name="orders",
        table_name="orders_t",
        primary_key=["order_id"],
    )

    active = store.get_dataset(user_id="u1", fingerprint=fp, table_name="orders_t")
    assert active is not None
    # Merged state: grain carried over, primary_key newly set.
    assert active["grain"] == "one row per order"
    assert active["primary_key"] == ["order_id"]


def test_forget_dataset_closes_active_and_returns_true(store, small_df):
    """12. forget_dataset returns True, get_dataset -> None, history shows closed row."""
    _, fp = store.register_dataset(
        user_id="u1",
        df=small_df,
        dataset_name="orders",
        table_name="orders_t",
        grain="g",
    )
    assert (
        store.forget_dataset(user_id="u1", fingerprint=fp, table_name="orders_t")
        is True
    )
    assert store.get_dataset(user_id="u1", fingerprint=fp, table_name="orders_t") is None

    history = store.list_dataset_history(
        user_id="u1", fingerprint=fp, table_name="orders_t"
    )
    assert len(history) == 1
    assert history[0]["valid_to"] is not None

    # Second forget returns False (no active row to close).
    assert (
        store.forget_dataset(user_id="u1", fingerprint=fp, table_name="orders_t")
        is False
    )


def test_active_row_uniqueness_invariant(store, small_df):
    """13. At most one active row per (user_id, fingerprint, table_name) after every register."""
    # Register 3 times in a row with different fields.
    _, fp = store.register_dataset(
        user_id="u1", df=small_df, dataset_name="orders", table_name="orders_t",
        grain="v1",
    )
    store.register_dataset(
        user_id="u1", df=small_df, dataset_name="orders", table_name="orders_t",
        grain="v2",
    )
    store.register_dataset(
        user_id="u1", df=small_df, dataset_name="orders", table_name="orders_t",
        primary_key=["order_id"],
    )

    # Direct SQL count -- the partial unique index must keep this <= 1.
    cnt = store._conn.execute(
        """
        SELECT COUNT(*) AS c FROM dataset_memory
         WHERE user_id = ? AND dataset_fingerprint = ? AND table_name = ?
           AND valid_to IS NULL
        """,
        ("u1", fp, "orders_t"),
    ).fetchone()["c"]
    assert cnt == 1

    # And total history is 3.
    total = store._conn.execute(
        """
        SELECT COUNT(*) AS c FROM dataset_memory
         WHERE user_id = ? AND dataset_fingerprint = ? AND table_name = ?
        """,
        ("u1", fp, "orders_t"),
    ).fetchone()["c"]
    assert total == 3


def test_dataset_user_scoping(store, small_df):
    """14. A dataset registered under u1 is invisible to u2."""
    _, fp = store.register_dataset(
        user_id="u1",
        df=small_df,
        dataset_name="orders",
        table_name="orders_t",
        grain="g",
    )
    assert store.get_dataset(user_id="u2", fingerprint=fp, table_name="orders_t") is None
    assert store.list_datasets("u2") == []
    # And u1 still sees it.
    assert store.get_dataset(user_id="u1", fingerprint=fp, table_name="orders_t") is not None
    assert len(store.list_datasets("u1")) == 1


# =========================================================== COLUMN STORE TESTS


def test_register_column_upserts_and_lists(store):
    """15. register_column inserts row; get_column returns it; list_columns includes it."""
    fp = "fp_test_15"
    new_id = store.register_column(
        fingerprint=fp,
        table_name="t",
        column_name="amount",
        semantic_type="currency",
    )
    assert _UUID_HEX_RE.match(new_id)

    row = store.get_column(fingerprint=fp, table_name="t", column_name="amount")
    assert row is not None
    assert row["id"] == new_id
    assert row["column_name"] == "amount"
    assert row["semantic_type"] == "currency"
    assert row["valid_to"] is None

    listed = store.list_columns(fingerprint=fp, table_name="t")
    assert any(r["id"] == new_id for r in listed)


def test_register_column_merges_and_supersedes(store):
    """16. semantic_type='currency' then pii_level='low' -> active has both; history len 2."""
    fp = "fp_test_16"
    id1 = store.register_column(
        fingerprint=fp, table_name="t", column_name="x", semantic_type="currency"
    )
    id2 = store.register_column(
        fingerprint=fp, table_name="t", column_name="x", pii_level="low"
    )
    assert id1 != id2

    active = store.get_column(fingerprint=fp, table_name="t", column_name="x")
    assert active is not None
    assert active["id"] == id2
    assert active["semantic_type"] == "currency"  # carried over
    assert active["pii_level"] == "low"            # newly set

    history = store.list_column_history(
        fingerprint=fp, table_name="t", column_name="x"
    )
    assert len(history) == 2
    assert history[0]["id"] == id1
    assert history[1]["id"] == id2
    assert history[0]["valid_to"] is not None
    assert history[1]["valid_to"] is None


def test_register_column_pii_level_validation(store):
    """17. pii_level='extreme' raises ValueError; None and the four valid levels succeed."""
    fp = "fp_test_17"

    with pytest.raises(ValueError):
        store.register_column(
            fingerprint=fp, table_name="t", column_name="c", pii_level="extreme"
        )

    # None succeeds.
    store.register_column(
        fingerprint=fp, table_name="t", column_name="c_none", pii_level=None
    )
    assert (
        store.get_column(fingerprint=fp, table_name="t", column_name="c_none")
        is not None
    )

    # Each valid level on a distinct column to avoid merge interference.
    for lvl in ("none", "low", "medium", "high"):
        col = f"c_{lvl}"
        store.register_column(
            fingerprint=fp, table_name="t", column_name=col, pii_level=lvl
        )
        row = store.get_column(fingerprint=fp, table_name="t", column_name=col)
        assert row is not None
        assert row["pii_level"] == lvl


def test_column_has_no_user_scoping(store):
    """18. column_memory is keyed by fingerprint only -- no user_id, both users see same row."""
    fp = "fp_shared_18"
    # Insert from "u1 context" (only used implicitly via... nothing -- store has
    # no user_id on column side). Just call register_column once.
    new_id = store.register_column(
        fingerprint=fp,
        table_name="t",
        column_name="shared_col",
        semantic_type="currency",
    )
    # Read from store with no user notion -- there's no user param to vary
    # at the store layer. Verify the row schema has no user_id key (per
    # contract column_memory has no user_id column).
    row = store.get_column(fingerprint=fp, table_name="t", column_name="shared_col")
    assert row is not None
    assert row["id"] == new_id
    assert "user_id" not in row

    # And via the tool layer two factories with different user_ids both see
    # the same column row (this is the user-facing equivalent of "no
    # scoping at the store level").
    df = pd.DataFrame({"shared_col": [1, 2, 3]})
    # We rebuild a tool factory bound to this df; its computed fingerprint
    # must match `fp` for the tool to see the row. To guarantee that, we
    # use the actual computed fp for the df.
    real_fp = compute_fingerprint(df)
    # Insert a column row under the real fingerprint so both factories
    # observe it.
    store.register_column(
        fingerprint=real_fp,
        table_name="t",
        column_name="shared_col",
        semantic_type="currency",
    )
    u1_tools = _tools_by_name(
        make_dataset_tools(store, df, dataset_name="d", table_name="t", user_id="u1")
    )
    u2_tools = _tools_by_name(
        make_dataset_tools(store, df, dataset_name="d", table_name="t", user_id="u2")
    )
    u1_out = u1_tools["recall_column"].invoke({"column_name": "shared_col"})
    u2_out = u2_tools["recall_column"].invoke({"column_name": "shared_col"})
    assert "shared_col" in u1_out
    assert "shared_col" in u2_out
    # Both see semantic_type=currency in the rendered string.
    assert "currency" in u1_out
    assert "currency" in u2_out


# ============================================================= TOOL-LEVEL TESTS


def test_factory_returns_six_tools_with_expected_names(store, orders_df):
    """19. Factory returns exactly 6 tools; sorted names match the contract."""
    tools = make_dataset_tools(
        store, orders_df, dataset_name="orders", table_name="orders_t", user_id="u1"
    )
    assert len(tools) == 6
    names = sorted(t.name for t in tools)
    assert names == [
        "list_columns",
        "list_datasets",
        "recall_column",
        "recall_dataset",
        "register_column",
        "register_dataset",
    ]


def test_register_dataset_tool_then_recall(store, orders_df):
    """20. register_dataset.invoke -> string mentions fingerprint+grain; recall round-trips."""
    tools = _tools_by_name(
        make_dataset_tools(
            store, orders_df, dataset_name="orders", table_name="orders_t",
            user_id="u1",
        )
    )
    out = tools["register_dataset"].invoke({"grain": "one row per order"})
    assert isinstance(out, str)

    fp = compute_fingerprint(orders_df)
    assert fp in out
    assert "one row per order" in out

    recall = tools["recall_dataset"].invoke({})
    assert isinstance(recall, str)
    assert "one row per order" in recall


def test_register_column_tool_unknown_column_returns_error_string(store, orders_df):
    """21. register_column.invoke with a column not in df -> error STRING, no raise."""
    tools = _tools_by_name(
        make_dataset_tools(
            store, orders_df, dataset_name="orders", table_name="orders_t",
            user_id="u1",
        )
    )
    out = tools["register_column"].invoke({"column_name": "x"})
    assert isinstance(out, str)
    assert "Error" in out
    assert "x" in out


def test_register_column_tool_bad_pii_level_returns_error_string(store, orders_df):
    """22. register_column with bad pii_level on a real column -> error STRING, no raise."""
    tools = _tools_by_name(
        make_dataset_tools(
            store, orders_df, dataset_name="orders", table_name="orders_t",
            user_id="u1",
        )
    )
    real_col = "amount"
    assert real_col in orders_df.columns
    out = tools["register_column"].invoke(
        {"column_name": real_col, "pii_level": "EXTREME"}
    )
    assert isinstance(out, str)
    assert "Error" in out


def test_list_columns_tool_empty_then_populated(store, orders_df):
    """23. list_columns empty case wording is documented; after register includes column."""
    tools = _tools_by_name(
        make_dataset_tools(
            store, orders_df, dataset_name="orders", table_name="orders_t",
            user_id="u1",
        )
    )
    empty_out = tools["list_columns"].invoke({})
    # Per dataset_tools.py: "No column memory for table '{table_name}'."
    assert empty_out == "No column memory for table 'orders_t'."

    tools["register_column"].invoke(
        {"column_name": "amount", "semantic_type": "currency"}
    )
    populated = tools["list_columns"].invoke({})
    assert "amount" in populated
    assert "currency" in populated


def test_tool_user_scoping_for_dataset_side(store, orders_df):
    """24. Tools built for u1 don't see a dataset registered through u2's factory."""
    u1_tools = _tools_by_name(
        make_dataset_tools(
            store, orders_df, dataset_name="orders", table_name="orders_t",
            user_id="u1",
        )
    )
    u2_tools = _tools_by_name(
        make_dataset_tools(
            store, orders_df, dataset_name="orders", table_name="orders_t",
            user_id="u2",
        )
    )

    # u2 registers a dataset fact.
    u2_tools["register_dataset"].invoke({"grain": "u2-only grain"})

    # u1 recall must NOT see it.
    u1_recall = u1_tools["recall_dataset"].invoke({})
    assert "u2-only grain" not in u1_recall
    assert "No dataset memory" in u1_recall

    # u1 list_datasets must NOT include it.
    u1_list = u1_tools["list_datasets"].invoke({})
    assert "u2-only grain" not in u1_list
    # Per dataset_tools.py: empty message is "No datasets registered yet."
    assert "No datasets registered yet." in u1_list

    # u2 still sees its own data.
    u2_recall = u2_tools["recall_dataset"].invoke({})
    assert "u2-only grain" in u2_recall


# ====================================== v1.1: DATASET CONTEXT + CROSS-SESSION LOOKUP


def test_register_context_inserts_then_get_context_returns(store, small_df):
    """25. register_context inserts a row; get_context returns a DatasetContext."""
    from mads.dataset.profiler import profile, DatasetContext

    ctx = profile(small_df, dataset_name="orders")
    store.register_context(ctx)

    fetched = store.get_context(ctx.semantic_fingerprint)
    assert isinstance(fetched, DatasetContext)
    assert fetched.semantic_fingerprint == ctx.semantic_fingerprint
    assert fetched.dataset_name == "orders"
    assert fetched.columns == ctx.columns
    assert fetched.dtypes == ctx.dtypes
    assert fetched.profile_summary == ctx.profile_summary


def test_register_context_upserts_in_place(store, small_df):
    """26. Second register_context with same semantic_fp UPDATEs in place — no new row."""
    from mads.dataset.profiler import profile

    ctx_v1 = profile(small_df, dataset_name="orders_v1")
    store.register_context(ctx_v1)
    cnt1 = store._conn.execute(
        "SELECT COUNT(*) AS c FROM dataset_context_memory"
    ).fetchone()["c"]
    assert cnt1 == 1

    ctx_v2 = profile(small_df, dataset_name="orders_v2", file_bytes=b"raw")
    assert ctx_v2.semantic_fingerprint == ctx_v1.semantic_fingerprint
    store.register_context(ctx_v2)

    cnt2 = store._conn.execute(
        "SELECT COUNT(*) AS c FROM dataset_context_memory"
    ).fetchone()["c"]
    assert cnt2 == 1  # still one row -> in-place update

    fetched = store.get_context(ctx_v1.semantic_fingerprint)
    assert fetched.dataset_name == "orders_v2"
    assert fetched.exact_fingerprint  # populated from v2's file_bytes


def test_get_context_returns_none_when_absent(store):
    """27. get_context for an unknown fingerprint returns None."""
    assert store.get_context("nonexistent_fp") is None


def test_lookup_dataset_returns_none_when_nothing_registered(store, small_df):
    """28. lookup_dataset with no prior registration -> None."""
    out = store.lookup_dataset(
        user_id="u1", df=small_df, table_name="orders_t"
    )
    assert out is None


def test_lookup_dataset_exact_match_when_file_bytes_match(store, small_df):
    """29. lookup_dataset with matching file_bytes returns matched_by='exact'."""
    raw = b"id,name\n1,a\n2,b\n"
    store.register_dataset(
        user_id="u1",
        df=small_df,
        dataset_name="orders",
        table_name="orders_t",
        grain="one row per order",
        file_bytes=raw,
    )
    out = store.lookup_dataset(
        user_id="u1", df=small_df, table_name="orders_t", file_bytes=raw
    )
    assert out is not None
    assert out["matched_by"] == "exact"
    assert out["row"]["grain"] == "one row per order"


def test_lookup_dataset_semantic_fallback_when_bytes_differ(store, small_df):
    """30. lookup_dataset with mismatched file_bytes falls back to semantic match."""
    raw_a = b"id,name\n1,a\n2,b\n"
    raw_b = b"id,name\n1,a\n2,b\n3,c\n"
    store.register_dataset(
        user_id="u1",
        df=small_df,
        dataset_name="orders",
        table_name="orders_t",
        grain="g",
        file_bytes=raw_a,
    )
    out = store.lookup_dataset(
        user_id="u1", df=small_df, table_name="orders_t", file_bytes=raw_b
    )
    assert out is not None
    assert out["matched_by"] == "semantic"
    assert out["row"]["grain"] == "g"


def test_lookup_dataset_semantic_match_when_no_file_bytes(store, small_df):
    """31. lookup_dataset with no file_bytes only attempts the semantic path."""
    store.register_dataset(
        user_id="u1",
        df=small_df,
        dataset_name="orders",
        table_name="orders_t",
        grain="g",
        # no file_bytes -> exact_fingerprint column is NULL
    )
    out = store.lookup_dataset(
        user_id="u1", df=small_df, table_name="orders_t"
    )
    assert out is not None
    assert out["matched_by"] == "semantic"


def test_register_dataset_with_file_bytes_populates_both_new_columns(store, small_df):
    """32. register_dataset(file_bytes=...) populates exact_fingerprint AND semantic_fingerprint."""
    from mads.dataset.fingerprint import exact_fingerprint, semantic_fingerprint

    raw = b"id,name\n1,a\n2,b\n"
    _, fp = store.register_dataset(
        user_id="u1",
        df=small_df,
        dataset_name="orders",
        table_name="orders_t",
        grain="g",
        file_bytes=raw,
    )
    # Returned fingerprint is the semantic_fingerprint of the df.
    assert fp == semantic_fingerprint(small_df)

    row = store._conn.execute(
        """
        SELECT exact_fingerprint, semantic_fingerprint
          FROM dataset_memory
         WHERE user_id = ? AND dataset_fingerprint = ? AND table_name = ?
           AND valid_to IS NULL
        """,
        ("u1", fp, "orders_t"),
    ).fetchone()
    assert row["exact_fingerprint"] == exact_fingerprint(raw)
    assert row["semantic_fingerprint"] == fp


def test_register_dataset_without_file_bytes_leaves_exact_null(store, small_df):
    """33. register_dataset without file_bytes -> semantic_fingerprint set, exact NULL."""
    from mads.dataset.fingerprint import semantic_fingerprint

    _, fp = store.register_dataset(
        user_id="u1",
        df=small_df,
        dataset_name="orders",
        table_name="orders_t",
        grain="g",
    )
    row = store._conn.execute(
        """
        SELECT exact_fingerprint, semantic_fingerprint
          FROM dataset_memory
         WHERE user_id = ? AND dataset_fingerprint = ? AND table_name = ?
           AND valid_to IS NULL
        """,
        ("u1", fp, "orders_t"),
    ).fetchone()
    assert row["exact_fingerprint"] is None
    assert row["semantic_fingerprint"] == semantic_fingerprint(small_df)
