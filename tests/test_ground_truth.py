"""Unit tests for eval.ground_truth — the broad-eval helper library."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import pytest  # noqa: E402

from ground_truth import (  # noqa: E402
    SYNTHETIC_PREMIUM_ORDER_MEAN_AMOUNT,
    TELCO_CHURN_PCT,
    TELCO_CHURN_BY_CONTRACT_PCT,
    TELCO_CONTRACT_VALUES,
    TELCO_FILTERED_CHURN_BY_CONTRACT,
    TELCO_FIBER_CHURN_PCT,
    TELCO_QUALIFIED_CUSTOMER_CHURN,
    TELCO_ROW_COUNT,
    TELCO_TOTAL_CHARGES_BLANK_ROWS,
    assert_all_values_present,
    assert_at_least_n_present,
    assert_consistent_numbers,
    assert_no_value_present,
    assert_number_matches,
    assert_pattern_absent,
    assert_pattern_present,
    compute_filtered_churn_by_contract,
    compute_qualified_customer_churn,
    count_pattern_hits,
    extract_numbers,
    extract_percentages,
)


# -----------------------------------------------------------------------------
# Ground-truth constant sanity (catches regressions in the CSV path)
# -----------------------------------------------------------------------------
def test_telco_constants_match_known_truth():
    assert TELCO_ROW_COUNT == 7043
    assert TELCO_CHURN_PCT == pytest.approx(26.5370, abs=0.01)
    assert TELCO_TOTAL_CHARGES_BLANK_ROWS == 11
    assert set(TELCO_CONTRACT_VALUES) == {"Month-to-month", "One year", "Two year"}
    assert TELCO_CHURN_BY_CONTRACT_PCT["Month-to-month"] == pytest.approx(42.71, abs=0.05)
    assert TELCO_CHURN_BY_CONTRACT_PCT["One year"] == pytest.approx(11.27, abs=0.05)
    assert TELCO_CHURN_BY_CONTRACT_PCT["Two year"] == pytest.approx(2.83, abs=0.05)
    assert TELCO_FIBER_CHURN_PCT == pytest.approx(41.89, abs=0.05)


def test_qualified_customer_churn_matches_baked_value():
    assert TELCO_QUALIFIED_CUSTOMER_CHURN == pytest.approx(21.71, abs=0.05)
    # Computed-on-call returns the same value.
    assert compute_qualified_customer_churn() == pytest.approx(21.71, abs=0.05)


def test_loyal_customer_count_matches_known_truth():
    """v1.4 adv_02 ground truth: count of rows with tenure>36 AND
    Contract=='Two year' on the Telco churn dataset."""
    from ground_truth import TELCO_LOYAL_CUSTOMER_COUNT, compute_loyal_customer_count

    # 1441 rows match on the standard Telco snapshot.
    assert TELCO_LOYAL_CUSTOMER_COUNT == 1441
    assert compute_loyal_customer_count() == 1441


def test_synthetic_premium_order_mean_amount_deterministic():
    """multi_03 ground truth: mean amount of premium_order rows (amount>300
    AND discount_pct<10) on the seed=42 synthetic sales dataset. The value
    is computed from the generator at import time; we sanity-check it lies
    in the dataset's value range and is reproducible."""
    # ~$399.39 for seed=42; allow a small tolerance against any rare future
    # numpy seed-stream shift.
    assert SYNTHETIC_PREMIUM_ORDER_MEAN_AMOUNT == pytest.approx(399.39, abs=1.0)


def test_filtered_churn_by_contract_matches_baked_values():
    out = compute_filtered_churn_by_contract()
    assert set(out) == {"Month-to-month", "One year", "Two year"}
    assert out["Month-to-month"] == pytest.approx(42.71, abs=0.05)
    assert out["One year"] == pytest.approx(11.28, abs=0.05)
    assert out["Two year"] == pytest.approx(2.85, abs=0.05)
    # Module-level constant matches.
    assert TELCO_FILTERED_CHURN_BY_CONTRACT == out


# -----------------------------------------------------------------------------
# Numeric extraction
# -----------------------------------------------------------------------------
def test_extract_percentages_basic():
    text = "The churn rate is 26.5%, while the One year segment is 11.3%."
    assert extract_percentages(text) == [26.5, 11.3]


def test_extract_percentages_handles_integer_and_decimals():
    assert extract_percentages("100% and 0.1% and 42%") == [100.0, 0.1, 42.0]


def test_extract_numbers_skips_percent_signs():
    # Bare-number extractor pulls raw tokens (the 26.5 inside "26.5%" is still
    # extracted because '%' is not part of the number itself).
    out = extract_numbers("answer is 7043 rows, 11 blanks")
    assert 7043.0 in out
    assert 11.0 in out


# -----------------------------------------------------------------------------
# assert_number_matches
# -----------------------------------------------------------------------------
def test_assert_number_matches_within_tolerance():
    ok, _ = assert_number_matches("rate is 26.5%", expected=26.5)
    assert ok


def test_assert_number_matches_outside_tolerance():
    ok, reason = assert_number_matches("rate is 73.5%", expected=26.5, tolerance=0.5)
    assert not ok
    assert "73.5" in reason


def test_assert_number_matches_no_percentage_present():
    ok, reason = assert_number_matches("no figures here", expected=26.5)
    assert not ok
    assert "no candidate" in reason


# -----------------------------------------------------------------------------
# assert_all_values_present / assert_no_value_present
# -----------------------------------------------------------------------------
def test_assert_all_values_present_match():
    ok, _ = assert_all_values_present(
        "Month-to-month, One year, Two year customers",
        TELCO_CONTRACT_VALUES,
    )
    assert ok


def test_assert_all_values_present_missing():
    ok, reason = assert_all_values_present(
        "Month-to-month and One year only",
        TELCO_CONTRACT_VALUES,
    )
    assert not ok
    assert "Two year" in reason


def test_assert_no_value_present_clean():
    ok, _ = assert_no_value_present("synthetic columns: amount, region", ["customerID"])
    assert ok


def test_assert_no_value_present_bleed():
    ok, reason = assert_no_value_present(
        "we always rename customer_id to ...",
        ["customer_id", "customerID"],
    )
    assert not ok
    assert "customer_id" in reason


# -----------------------------------------------------------------------------
# assert_pattern_present / assert_pattern_absent
# -----------------------------------------------------------------------------
def test_assert_pattern_present_match():
    ok, _ = assert_pattern_present("they paid recently", r"paid|payment")
    assert ok


def test_assert_pattern_absent_clean():
    ok, _ = assert_pattern_absent("no signup here", r"never\s+signed\s+up")
    assert ok


def test_assert_pattern_absent_violation():
    ok, _ = assert_pattern_absent("they signed up", r"signed?\s*up")
    assert not ok


# -----------------------------------------------------------------------------
# assert_consistent_numbers
# -----------------------------------------------------------------------------
def test_assert_consistent_numbers_agree():
    ok, _ = assert_consistent_numbers("A says 26.5%", "B reports 26.4%", tolerance=0.5)
    assert ok


def test_assert_consistent_numbers_disagree():
    ok, _ = assert_consistent_numbers("A says 26.5%", "B reports 73.5%", tolerance=0.5)
    assert not ok


# -----------------------------------------------------------------------------
# count_pattern_hits / assert_at_least_n_present
# -----------------------------------------------------------------------------
def test_count_pattern_hits_returns_distinct_count():
    text = "tenure is in months, contract grouping, totalcharges coerce"
    n = count_pattern_hits(
        text,
        [r"tenure.*month", r"contract", r"totalcharges.*coerce", r"never_here"],
    )
    assert n == 3


def test_assert_at_least_n_present_pass():
    text = "tenure is in months, contract grouping, totalcharges coerce"
    ok, _ = assert_at_least_n_present(
        text,
        [r"tenure.*month", r"contract", r"totalcharges.*coerce"],
        n=3,
        label="facts",
    )
    assert ok


def test_assert_at_least_n_present_fail():
    text = "only one fact: contract"
    ok, reason = assert_at_least_n_present(
        text,
        [r"tenure.*month", r"contract", r"totalcharges.*coerce"],
        n=3,
    )
    assert not ok
    assert "1/3" in reason or "1 / 3" in reason or "missing" in reason
