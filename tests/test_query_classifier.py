"""Pytest suite for :func:`mads.memory.retrievers.query_classifier.is_data_question`.

The classifier is a deterministic keyword/substring check used by
:class:`HybridRetriever` to gate memory retrieval on non-analytical queries.

Design contract under test:

1. ``CREATIVE_PATTERNS`` substrings → ``False`` (denylist precedence).
2. ``ANALYTICAL_PATTERNS`` substrings → ``True``.
3. Default → ``True`` (bias toward not breaking the data flow).
4. Empty / whitespace-only → ``True``.

The denylist runs FIRST, so a query containing BOTH a creative token and
an analytical token returns ``False``.
"""

from __future__ import annotations

import pytest

from mads.memory.retrievers.query_classifier import is_data_question


# --------------------------------------------------------------- creative → False


def test_poem_returns_false():
    assert is_data_question("Write me a short poem about telecoms") is False


def test_joke_returns_false():
    assert is_data_question("Tell me a joke") is False


def test_haiku_returns_false():
    assert is_data_question("Compose a haiku about data") is False


def test_song_returns_false():
    assert is_data_question("Sing me a song about analytics") is False


def test_greeting_returns_false():
    assert is_data_question("hello") is False


def test_who_are_you_returns_false():
    assert is_data_question("Who are you and what do you do?") is False


def test_creative_with_data_words_still_false():
    """Denylist precedence: 'poem' wins over 'churn'/'analysis'."""
    assert is_data_question("Write me a poem about churn analysis") is False


def test_neg_01_query_returns_false():
    """Verbatim prompt from neg_01_unrelated_query — the bug we're closing."""
    assert is_data_question("Write me a short poem about telecoms.") is False


# --------------------------------------------------------------- analytical → True


def test_show_returns_true():
    assert is_data_question("Show me churn by Contract") is True


def test_compute_returns_true():
    assert is_data_question("Compute conversion rate") is True


def test_plot_returns_true():
    assert is_data_question("Plot the distribution of TotalCharges") is True


def test_describe_returns_true():
    assert is_data_question("Describe the dataset") is True


def test_load_csv_returns_true():
    assert is_data_question("Load /tmp/foo.csv and show the first 5 rows") is True


def test_average_returns_true():
    assert is_data_question("Compute the average MonthlyCharges") is True


def test_eda_term_returns_true():
    assert is_data_question("Do EDA on this table") is True


# --------------------------------------------------------------- default → True


def test_empty_string_returns_true():
    assert is_data_question("") is True


def test_whitespace_only_returns_true():
    assert is_data_question("   \n\t  ") is True


def test_unknown_returns_true():
    """Ambiguous / unclassified queries default to True (data-flow bias)."""
    assert is_data_question("What's the weather today") is True


def test_just_data_term_returns_true():
    assert is_data_question("churn") is True


# --------------------------------------------------------------- case insensitivity


def test_case_insensitive_creative():
    assert is_data_question("WRITE ME A POEM") is False


def test_case_insensitive_analytical():
    assert is_data_question("SHOW ME THE DATA") is True


# --------------------------------------------------------------- pattern surface


def test_creative_patterns_nonempty():
    from mads.memory.retrievers.query_classifier import CREATIVE_PATTERNS

    assert "poem" in CREATIVE_PATTERNS
    assert "joke" in CREATIVE_PATTERNS


def test_analytical_patterns_nonempty():
    from mads.memory.retrievers.query_classifier import ANALYTICAL_PATTERNS

    assert "show" in ANALYTICAL_PATTERNS
    assert "churn" in ANALYTICAL_PATTERNS
