"""Tests for the heuristic :func:`mads.router.route` classifier."""

from __future__ import annotations

import pytest

from mads.router import route


# --------------------------------------------------------------------- loader


def test_loader_load_csv():
    assert route("Load /tmp/x.csv") == "loader"


def test_loader_preview():
    assert route("Preview the data file") == "loader"


def test_loader_first_n_rows():
    assert route("show me the first 5 rows") == "loader"


def test_loader_list_columns():
    assert route("list the columns") == "loader"


# --------------------------------------------------------------------- eda


def test_eda_describe_is_residual():
    assert route("Describe the dataset") == "eda"


def test_eda_missing_values():
    assert route("How many missing values are in each column?") == "eda"


def test_eda_correlation():
    assert route("What is the correlation between A and B") == "eda"


# --------------------------------------------------------------------- viz


def test_viz_plot():
    assert route("Plot churn by tenure") == "viz"


def test_viz_histogram():
    assert route("histogram of monthly_charges") == "viz"


def test_viz_scatter():
    assert route("scatter customerID vs charges") == "viz"


# --------------------------------------------------------------------- priority


def test_viz_wins_over_loader_on_plot_columns():
    # "plot the columns of x.csv" mentions both signals: viz must win.
    assert route("plot the columns of x.csv") == "viz"


def test_viz_wins_over_loader_on_chart_after_load():
    assert route("After loading, chart the distribution") == "viz"


# --------------------------------------------------------------------- shape


def test_case_insensitive():
    assert route("LOAD DATA.CSV") == "loader"
    assert route("PLOT STUFF") == "viz"


def test_deterministic_idempotent():
    q = "What is the correlation matrix?"
    assert route(q) == route(q) == route(q)


def test_non_string_raises():
    with pytest.raises(TypeError):
        route(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        route(42)  # type: ignore[arg-type]
