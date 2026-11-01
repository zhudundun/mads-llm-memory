"""Pytest suite for :class:`mads.memory.retrievers.DatasetRetriever`.

The retriever wraps :meth:`DatasetSchemaStore.lookup_dataset`'s
exact-then-semantic fallback. These tests exercise the documented
scoring + reason strings, the "no artifact" short-circuit, and user
isolation.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mads.memory.dataset_schema import DatasetSchemaStore
from mads.memory.retrievers import DatasetRetriever, RetrievedMemory


# --------------------------------------------------------------------- fixtures


@pytest.fixture
def store():
    s = DatasetSchemaStore(db_path=":memory:")
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def retriever(store):
    return DatasetRetriever(store)


@pytest.fixture
def small_df():
    return pd.DataFrame(
        {
            "order_id": [1, 2, 3],
            "customer_id": [10, 20, 30],
            "amount": [9.99, 19.99, 29.99],
        }
    )


# --------------------------------------------------------------------- tests


def test_exact_fingerprint_match_scores_one(store, retriever, small_df):
    """1. Matching file_bytes -> score=1.0 with the exact_fingerprint reason."""
    raw = b"id,name\n1,a\n2,b\n"
    store.register_dataset(
        user_id="u1",
        df=small_df,
        dataset_name="orders",
        table_name="orders_t",
        grain="g",
        file_bytes=raw,
    )
    out = retriever.retrieve(
        user_id="u1",
        table_name="orders_t",
        df=small_df,
        file_bytes=raw,
    )
    assert len(out) == 1
    assert isinstance(out[0], RetrievedMemory)
    assert out[0].score == 1.0
    assert out[0].reason == "exact_fingerprint match"
    assert out[0].category == "dataset"


def test_semantic_fingerprint_match_scores_point_seven(store, retriever, small_df):
    """2. Schema-only match -> score=0.7 with the semantic reason."""
    # Register WITHOUT file_bytes so exact_fingerprint is NULL; the lookup
    # then falls back to semantic.
    store.register_dataset(
        user_id="u1",
        df=small_df,
        dataset_name="orders",
        table_name="orders_t",
        grain="g",
    )
    out = retriever.retrieve(
        user_id="u1",
        table_name="orders_t",
        df=small_df,
    )
    assert len(out) == 1
    assert out[0].score == 0.7
    assert out[0].reason == "semantic_fingerprint match (logical dataset)"
    assert out[0].category == "dataset"


def test_no_match_returns_empty(retriever, small_df):
    """3. No prior registration -> empty result."""
    out = retriever.retrieve(
        user_id="u1",
        table_name="orders_t",
        df=small_df,
    )
    assert out == []


def test_neither_df_nor_file_bytes_returns_empty(retriever):
    """4. Calling without any fingerprintable artifact returns []."""
    out = retriever.retrieve(user_id="u1", table_name="orders_t")
    assert out == []


def test_user_isolation(store, retriever, small_df):
    """5. A dataset registered for user A is invisible to user B."""
    store.register_dataset(
        user_id="A",
        df=small_df,
        dataset_name="orders",
        table_name="orders_t",
        grain="g",
    )
    out_b = retriever.retrieve(user_id="B", table_name="orders_t", df=small_df)
    assert out_b == []
    out_a = retriever.retrieve(user_id="A", table_name="orders_t", df=small_df)
    assert len(out_a) == 1


def test_result_count_is_at_most_one(store, retriever, small_df):
    """6. Even with k>1 the result count is 0 or 1."""
    store.register_dataset(
        user_id="u1",
        df=small_df,
        dataset_name="orders",
        table_name="orders_t",
        grain="g",
    )
    out = retriever.retrieve(
        user_id="u1", table_name="orders_t", df=small_df, k=10
    )
    assert len(out) == 1


def test_category_tag_is_dataset(store, retriever, small_df):
    """7. Every returned RetrievedMemory has category='dataset'."""
    store.register_dataset(
        user_id="u1",
        df=small_df,
        dataset_name="orders",
        table_name="orders_t",
        grain="g",
    )
    out = retriever.retrieve(user_id="u1", table_name="orders_t", df=small_df)
    assert all(r.category == "dataset" for r in out)


def test_reason_strings_match_spec_exactly(store, retriever, small_df):
    """8. The two reason strings are exactly as documented in the spec."""
    raw = b"id,name\n1,a\n2,b\n"
    store.register_dataset(
        user_id="u1",
        df=small_df,
        dataset_name="orders",
        table_name="orders_t",
        grain="g",
        file_bytes=raw,
    )
    exact_hit = retriever.retrieve(
        user_id="u1", table_name="orders_t", df=small_df, file_bytes=raw
    )
    assert exact_hit[0].reason == "exact_fingerprint match"
    # Mismatched bytes => semantic fallback.
    semantic_hit = retriever.retrieve(
        user_id="u1",
        table_name="orders_t",
        df=small_df,
        file_bytes=b"different bytes",
    )
    assert semantic_hit[0].reason == "semantic_fingerprint match (logical dataset)"
