"""Deterministic DataFrame and raw-file fingerprinting.

Two complementary fingerprints support cross-session dataset recognition:

* :func:`exact_fingerprint` — sha256 of the raw file bytes. Bytes-equal
  files produce identical fingerprints; any byte-level change (a single
  appended row, a re-saved CSV with different line endings, even resaving
  through pandas) breaks the match. Requires only the raw bytes, no
  parsing.
* :func:`semantic_fingerprint` — sha256 of sorted column names + sorted
  dtypes. Stable across snapshots of the same *logical* dataset (different
  row counts, different sample values, etc.) as long as the schema is
  unchanged. Order-invariant in column ordering.

For backward compatibility, ``compute_fingerprint`` is an alias for
:func:`semantic_fingerprint` — historical callers (the dataset_memory
store, dataset_tools, demos) get the schema-only semantics with no code
changes.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd


def exact_fingerprint(file_bytes: bytes) -> str:
    """Return a sha256 hex digest of the raw file bytes.

    Deterministic on identical bytes. No parsing required.
    Returns a 64-char lowercase hex digest.
    """
    return hashlib.sha256(file_bytes).hexdigest()


def semantic_fingerprint(df: pd.DataFrame) -> str:
    """Return a sha256 hex digest of the dataframe's schema only.

    Built from the sorted column names plus the dtypes aligned to that
    sorted order. Stable across snapshots of the same logical dataset
    (different row counts, different sample values). Order-invariant in
    column ordering. Returns a 64-char lowercase hex digest.
    """
    cols = sorted(map(str, df.columns))
    dtypes = [str(df[c].dtype) for c in cols]
    payload = {"columns": cols, "dtypes": dtypes}
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# Backward-compat alias. Pre-v1.1 callers (dataset_memory store, dataset_tools,
# demos, the old fingerprint tests) all called ``compute_fingerprint(df)``
# expecting an order-invariant schema-aware hash. Aliasing to the schema-only
# fingerprint preserves the order-invariance guarantees while dropping the
# row-bucket + per-column-sample components that made the old fingerprint
# sample-sensitive.
compute_fingerprint = semantic_fingerprint


__all__ = ["compute_fingerprint", "exact_fingerprint", "semantic_fingerprint"]
