"""Unit tests for eval.synthetic_dataset — determinism and schema."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import pandas as pd  # noqa: E402

from synthetic_dataset import (  # noqa: E402
    SYNTHETIC_COLUMNS,
    SYNTHETIC_CUSTOMER_SEGMENTS,
    SYNTHETIC_PRODUCT_CATEGORIES,
    SYNTHETIC_REGIONS,
    SYNTHETIC_ROWS,
    generate_synthetic_sales,
    schema_summary,
    write_synthetic_csv,
)


def test_generate_is_deterministic():
    df1 = generate_synthetic_sales()
    df2 = generate_synthetic_sales()
    pd.testing.assert_frame_equal(df1, df2)


def test_schema_matches_spec():
    df = generate_synthetic_sales()
    assert len(df) == SYNTHETIC_ROWS
    assert tuple(df.columns) == SYNTHETIC_COLUMNS
    assert set(df["product_category"].unique()).issubset(set(SYNTHETIC_PRODUCT_CATEGORIES))
    assert set(df["customer_segment"].unique()).issubset(set(SYNTHETIC_CUSTOMER_SEGMENTS))
    assert set(df["region"].unique()).issubset(set(SYNTHETIC_REGIONS))
    # order_id must be unique
    assert df["order_id"].is_unique
    # amount must be in [10, 500]
    assert df["amount"].min() >= 10.0
    assert df["amount"].max() <= 500.0


def test_write_synthetic_csv_roundtrip(tmp_path):
    out = tmp_path / "syn.csv"
    p = write_synthetic_csv(out)
    assert p.exists()
    df = pd.read_csv(p)
    assert len(df) == SYNTHETIC_ROWS
    assert set(df.columns) == set(SYNTHETIC_COLUMNS)


def test_schema_summary_includes_known_keys():
    summary = schema_summary()
    assert "rows = 120" in summary
    assert "order_id" in summary
    assert "product_category" in summary
