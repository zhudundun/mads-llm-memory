"""Deterministic synthetic sales dataset for cross-dataset isolation scenarios.

Generated programmatically (no URL downloads, no real public datasets).
Same seed → same dataframe → same ground-truth aggregates. The dataset is
written to a temp CSV path on demand so the eval harness can load it the
same way it loads ``baseline/data/churn_data.csv``.

Schema (exactly 120 rows, 7 columns):

    order_id         int64        unique 1001..1120
    product_category category     A / B / C / D
    customer_segment category     small / medium / large
    amount           float64      ~Uniform(10, 500)
    order_date       datetime     last 90 days from 2026-06-01 (deterministic)
    region           category     N / S / E / W
    discount_pct     float64      0.0 .. 25.0

Computed ground-truth aggregates (asserted in test_synthetic_dataset.py):

    rows = 120
    n_columns = 7
    mean_amount ≈ {value baked in below}
    distinct product_category = {A, B, C, D}
    distinct customer_segment = {small, medium, large}
    distinct region = {N, S, E, W}
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SYNTHETIC_SEED: int = 42
SYNTHETIC_ROWS: int = 120
SYNTHETIC_COLUMNS: tuple[str, ...] = (
    "order_id",
    "product_category",
    "customer_segment",
    "amount",
    "order_date",
    "region",
    "discount_pct",
)
SYNTHETIC_PRODUCT_CATEGORIES: tuple[str, ...] = ("A", "B", "C", "D")
SYNTHETIC_CUSTOMER_SEGMENTS: tuple[str, ...] = ("small", "medium", "large")
SYNTHETIC_REGIONS: tuple[str, ...] = ("N", "S", "E", "W")
SYNTHETIC_DATASET_NAME: str = "synthetic_sales"


def generate_synthetic_sales(seed: int = SYNTHETIC_SEED) -> pd.DataFrame:
    """Return the deterministic 120-row sales dataframe. Same seed → same df."""
    rng = np.random.default_rng(seed)
    n = SYNTHETIC_ROWS
    df = pd.DataFrame(
        {
            "order_id": np.arange(1001, 1001 + n, dtype=np.int64),
            "product_category": rng.choice(
                SYNTHETIC_PRODUCT_CATEGORIES, size=n
            ),
            "customer_segment": rng.choice(
                SYNTHETIC_CUSTOMER_SEGMENTS, size=n
            ),
            "amount": rng.uniform(10.0, 500.0, size=n).round(2),
            # 90 days of orders ending at a fixed anchor date; offsets are
            # integer days drawn deterministically.
            "order_date": pd.to_datetime("2026-06-01")
            - pd.to_timedelta(rng.integers(0, 90, size=n), unit="D"),
            "region": rng.choice(SYNTHETIC_REGIONS, size=n),
            "discount_pct": rng.uniform(0.0, 25.0, size=n).round(2),
        }
    )
    return df


def _tmp_csv_path() -> Path:
    """Path under the OS tmp dir; safe to overwrite per run."""
    return Path(tempfile.gettempdir()) / "mads_synthetic_sales.csv"


def write_synthetic_csv(path: str | Path | None = None) -> Path:
    """Materialize the dataset to disk; default path is in tmp.

    Returns the path written. Idempotent across runs given the same seed
    (file bytes will be identical because the dataframe is identical)."""
    target = Path(path) if path is not None else _tmp_csv_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_sales()
    df.to_csv(target, index=False)
    return target


def schema_summary() -> str:
    """Human-readable schema block to embed in the report."""
    df = generate_synthetic_sales()
    lines = [
        f"rows = {len(df)}",
        f"columns = {list(df.columns)}",
        "dtypes:",
    ]
    for c, dt in df.dtypes.items():
        lines.append(f"  {c}: {dt}")
    lines.append(f"distinct product_category: {sorted(df['product_category'].unique().tolist())}")
    lines.append(f"distinct customer_segment: {sorted(df['customer_segment'].unique().tolist())}")
    lines.append(f"distinct region: {sorted(df['region'].unique().tolist())}")
    lines.append(f"mean amount: {df['amount'].mean():.2f}")
    lines.append(f"mean discount_pct: {df['discount_pct'].mean():.2f}")
    return "\n".join(lines)


__all__ = [
    "SYNTHETIC_SEED",
    "SYNTHETIC_ROWS",
    "SYNTHETIC_COLUMNS",
    "SYNTHETIC_PRODUCT_CATEGORIES",
    "SYNTHETIC_CUSTOMER_SEGMENTS",
    "SYNTHETIC_REGIONS",
    "SYNTHETIC_DATASET_NAME",
    "generate_synthetic_sales",
    "write_synthetic_csv",
    "schema_summary",
]
