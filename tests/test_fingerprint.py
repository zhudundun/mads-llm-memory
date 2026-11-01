"""Pytest suite for the dual-fingerprint surface in :mod:`mads.dataset.fingerprint`.

Covers:

* :func:`exact_fingerprint` — deterministic SHA256 of raw bytes.
* :func:`semantic_fingerprint` — deterministic SHA256 of sorted
  columns + dtypes, row-count-invariant.
* The backward-compat aliasing of ``compute_fingerprint`` to
  ``semantic_fingerprint``.
"""

from __future__ import annotations

import re

import pandas as pd

from mads.dataset.fingerprint import (
    compute_fingerprint,
    exact_fingerprint,
    semantic_fingerprint,
)

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


# =================================================================== exact_fingerprint


def test_exact_fingerprint_deterministic_and_64_char_hex():
    """exact_fingerprint(b'hello') is deterministic and 64-char lowercase hex."""
    fp1 = exact_fingerprint(b"hello")
    fp2 = exact_fingerprint(b"hello")
    assert fp1 == fp2
    assert _HEX64_RE.match(fp1), f"expected 64-char lowercase hex, got {fp1!r}"


def test_exact_fingerprint_differs_for_distinct_bytes():
    """exact_fingerprint differs for distinct bytes."""
    assert exact_fingerprint(b"hello") != exact_fingerprint(b"world")
    # Single-byte change.
    assert exact_fingerprint(b"hello") != exact_fingerprint(b"hellO")


def test_exact_fingerprint_empty_bytes_returns_64_char_hex():
    """exact_fingerprint(b'') returns a 64-char hex digest (sha256 of empty)."""
    fp = exact_fingerprint(b"")
    assert _HEX64_RE.match(fp)
    # The known sha256 of the empty input.
    assert fp == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


# ================================================================ semantic_fingerprint


def test_semantic_fingerprint_64_char_lowercase_hex():
    """semantic_fingerprint returns 64-char lowercase hex."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    fp = semantic_fingerprint(df)
    assert _HEX64_RE.match(fp)


def test_semantic_fingerprint_same_for_different_row_counts_same_schema():
    """Two dfs with same cols+dtypes but different row counts -> SAME fingerprint."""
    df_small = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    df_big = pd.DataFrame(
        {"a": list(range(1000)), "b": [f"v{i}" for i in range(1000)]}
    )
    assert semantic_fingerprint(df_small) == semantic_fingerprint(df_big)


def test_semantic_fingerprint_differs_for_different_schemas():
    """Different column names -> different fingerprint."""
    df1 = pd.DataFrame({"a": [1, 2, 3]})
    df2 = pd.DataFrame({"b": [1, 2, 3]})
    assert semantic_fingerprint(df1) != semantic_fingerprint(df2)


def test_semantic_fingerprint_differs_for_different_dtypes():
    """Same column names but different dtypes -> different fingerprint."""
    df1 = pd.DataFrame({"a": [1, 2, 3]})
    df2 = df1.copy()
    df2["a"] = df2["a"].astype(float)
    assert str(df1["a"].dtype) != str(df2["a"].dtype)
    assert semantic_fingerprint(df1) != semantic_fingerprint(df2)


def test_semantic_fingerprint_column_order_invariant():
    """Reversing column order does not change the fingerprint."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    df_rev = df[list(reversed(df.columns))]
    assert semantic_fingerprint(df) == semantic_fingerprint(df_rev)


def test_semantic_fingerprint_empty_df_does_not_raise():
    """Empty df returns a stable 64-char hex digest, does not raise."""
    df = pd.DataFrame({"a": pd.Series([], dtype="int64")})
    fp = semantic_fingerprint(df)
    assert _HEX64_RE.match(fp)
    # Stable across calls.
    assert fp == semantic_fingerprint(df)


# =========================================================== backward-compat alias


def test_compute_fingerprint_is_semantic_fingerprint():
    """compute_fingerprint IS semantic_fingerprint (function identity check).

    Confirms the v1.1 aliasing: every legacy caller of compute_fingerprint
    transparently gets the new schema-only fingerprint.
    """
    assert compute_fingerprint is semantic_fingerprint


def test_compute_fingerprint_alias_produces_same_hash_as_semantic():
    """compute_fingerprint(df) == semantic_fingerprint(df) for an arbitrary df."""
    df = pd.DataFrame({"x": [1.0, 2.0], "y": ["a", "b"]})
    assert compute_fingerprint(df) == semantic_fingerprint(df)
