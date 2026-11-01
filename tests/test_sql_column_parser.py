"""Tests for :func:`mads.memory.sql_column_parser.extract_columns_from_sql`.

The parser is intentionally simplistic (regex-based, no SQL grammar). These
tests lock down the contract the v1.4 schema-coupling layer relies on:

* identifier-shaped tokens come through preserved
* SQL keywords / aggregates / booleans are rejected
* string literals are stripped before tokenizing (so quoted column-like
  text doesn't pollute the result)
* deduplication preserves first-occurrence order
* edge cases (empty, whitespace, comments, parens) don't crash
"""

from __future__ import annotations

from mads.memory.sql_column_parser import extract_columns_from_sql


def test_simple_predicate():
    assert extract_columns_from_sql("tenure > 12 AND MonthlyCharges > 50") == [
        "tenure",
        "MonthlyCharges",
    ]


def test_select_count_with_quoted_literal():
    """Quoted literal 'Yes' is stripped; table name 'churn_data' is a
    tolerated false-positive — the docstring documents this explicitly."""
    out = extract_columns_from_sql(
        "SELECT COUNT(*) FROM churn_data WHERE Churn = 'Yes'"
    )
    assert out == ["churn_data", "Churn"]


def test_no_whitespace_quoted_literal():
    out = extract_columns_from_sql("tenure>12 and PaymentMethod='Credit card'")
    assert out == ["tenure", "PaymentMethod"]


def test_empty_returns_empty():
    assert extract_columns_from_sql("") == []


def test_whitespace_only_returns_empty():
    assert extract_columns_from_sql("   \t  \n  ") == []


def test_non_string_input_returns_empty():
    assert extract_columns_from_sql(None) == []  # type: ignore[arg-type]
    assert extract_columns_from_sql(123) == []  # type: ignore[arg-type]


def test_keywords_rejected_case_insensitive():
    out = extract_columns_from_sql("SELECT a FROM t WHERE b IS NOT NULL")
    # No SELECT, FROM, WHERE, IS, NOT, NULL in the output.
    assert "SELECT" not in out and "select" not in out
    assert set(out) == {"a", "t", "b"}


def test_aggregate_function_names_rejected():
    out = extract_columns_from_sql("SUM(amount) + AVG(price) + MAX(qty)")
    assert out == ["amount", "price", "qty"]


def test_dedup_preserves_first_occurrence_order():
    out = extract_columns_from_sql("a > 1 AND b < 2 AND a > 3 AND c IS NOT NULL")
    assert out == ["a", "b", "c"]


def test_parenthesized_predicate():
    out = extract_columns_from_sql("(tenure > 12) AND (MonthlyCharges > 50)")
    assert out == ["tenure", "MonthlyCharges"]


def test_multi_line_with_comment():
    sql = """
    SELECT
      -- ignored: status check
      customer_id,
      MonthlyCharges
    FROM customers
    WHERE Churn = 'No'
    """
    out = extract_columns_from_sql(sql)
    # Note: 'status' and 'check' (from the line comment) must NOT leak.
    assert "status" not in out
    assert "check" not in out
    assert "customer_id" in out
    assert "MonthlyCharges" in out


def test_block_comment_stripped():
    sql = "/* leading note: amount */ SELECT a FROM t"
    out = extract_columns_from_sql(sql)
    assert "leading" not in out
    assert "note" not in out
    assert "amount" not in out
    assert out == ["a", "t"]


def test_boolean_literals_rejected():
    out = extract_columns_from_sql("is_paid = TRUE AND is_trial = FALSE")
    assert out == ["is_paid", "is_trial"]


def test_numeric_only_tokens_rejected():
    # No bare integer should ever appear as a column.
    out = extract_columns_from_sql("colA = 100 AND colB = 3.14")
    assert "100" not in out
    assert "3" not in out
    assert "14" not in out
    assert set(out) == {"colA", "colB"}


def test_quoted_text_with_keyword_does_not_leak():
    """String literal contains 'FROM' — it must not slip into the output."""
    out = extract_columns_from_sql("notes = 'data FROM the warehouse' AND id = 1")
    assert "FROM" not in out and "from" not in out
    assert "data" not in out
    assert "warehouse" not in out
    assert out == ["notes", "id"]
