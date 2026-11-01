"""Dataset profiler: produce a :class:`DatasetContext` from a DataFrame.

Implements the v1.1 "dataset profiler + dual-fingerprint" surface. The
profiler is pure-Python and deterministic — no LLM call, no API key needed.
The output drives cross-session dataset recognition by attaching a profile
plus dual fingerprints (exact / semantic) to every logical dataset.

Conservative heuristics (see CLAUDE.md / spec):

* ``primary_key_candidates`` — fully unique AND no nulls.
* ``join_key_candidates`` — name ends in ``_id``/``_key``/``Id``/``Key``
  OR high-cardinality and name ends in ``_id``/``_key``; minus PK
  candidates.
* ``time_columns`` — pandas datetime64-like dtype only; no string-date
  parsing.
* ``numeric_columns`` — numeric AND not boolean (booleans → categorical).
* ``categorical_columns`` — object/string/bool AND nunique(dropna=True)
  <= 20.

Overlap between numeric and categorical is allowed by design (low-card
ints are both).
"""
from __future__ import annotations

import re
from typing import Optional

import pandas as pd
from pydantic import BaseModel

from mads.dataset.fingerprint import exact_fingerprint, semantic_fingerprint


_JOIN_KEY_NAME_RE = re.compile(r"(_id|_key|Id|Key)$")
_JOIN_KEY_SUFFIXES = ("_id", "_key")
_CATEGORICAL_MAX_CARD = 20


class DatasetContext(BaseModel):
    """Structured profile of a dataset, keyed by ``semantic_fingerprint``.

    All list fields preserve the dataframe's column ordering except where
    noted. ``profile_summary`` is a deterministic, template-rendered string
    suitable for prompt injection.
    """

    dataset_name: str
    exact_fingerprint: str
    semantic_fingerprint: str
    columns: list[str]
    dtypes: dict[str, str]
    row_count: int
    primary_key_candidates: list[str]
    join_key_candidates: list[str]
    time_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    profile_summary: str


# --------------------------------------------------------------------- heuristics


def _primary_key_candidates(df: pd.DataFrame) -> list[str]:
    """Columns that are fully unique AND have no nulls. Column-order."""
    if len(df) == 0:
        return []
    n = len(df)
    out: list[str] = []
    for c in df.columns:
        col = df[c]
        try:
            if int(col.isna().sum()) != 0:
                continue
            if int(col.nunique(dropna=False)) != n:
                continue
        except Exception:
            continue
        out.append(str(c))
    return out


def _join_key_candidates(df: pd.DataFrame, pk: list[str]) -> list[str]:
    """Name-suffix match (case-sensitive endings) PLUS high-card-with-_id/_key.

    Excludes columns already in ``pk``. Output sorted by column-order.
    """
    pk_set = set(pk)
    n = len(df)
    out: list[str] = []
    for c in df.columns:
        name = str(c)
        if name in pk_set:
            continue
        matched = False
        if _JOIN_KEY_NAME_RE.search(name):
            matched = True
        else:
            # The "high cardinality + _id/_key suffix" rule. We already
            # know the regex didn't match, but the spec specifies that
            # the suffix check here is the lowercased `_id`/`_key` form
            # (the regex already covers Id/Key and snake_case forms).
            if n > 0 and name.endswith(_JOIN_KEY_SUFFIXES):
                try:
                    if int(df[c].nunique(dropna=True)) > 0.5 * n:
                        matched = True
                except Exception:
                    pass
        if matched:
            out.append(name)
    return out


def _time_columns(df: pd.DataFrame) -> list[str]:
    """Columns whose dtype is datetime64-like. NO string-date parsing."""
    out: list[str] = []
    for c in df.columns:
        col = df[c]
        if pd.api.types.is_datetime64_any_dtype(col):
            out.append(str(c))
            continue
        if str(col.dtype).startswith("datetime"):
            out.append(str(c))
    return out


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    """Numeric AND not boolean. Booleans are routed to categorical."""
    out: list[str] = []
    for c in df.columns:
        col = df[c]
        if pd.api.types.is_bool_dtype(col):
            continue
        if pd.api.types.is_numeric_dtype(col):
            out.append(str(c))
    return out


def _categorical_columns(
    df: pd.DataFrame, numeric: list[str], time: list[str]
) -> list[str]:
    """object / string / bool with nunique(dropna=True) <= 20, column-order."""
    numeric_set = set(numeric)
    time_set = set(time)
    out: list[str] = []
    for c in df.columns:
        name = str(c)
        if name in numeric_set or name in time_set:
            continue
        col = df[c]
        is_candidate = (
            pd.api.types.is_object_dtype(col)
            or pd.api.types.is_string_dtype(col)
            or pd.api.types.is_bool_dtype(col)
        )
        if not is_candidate:
            continue
        try:
            card = int(col.nunique(dropna=True))
        except Exception:
            continue
        if card <= _CATEGORICAL_MAX_CARD:
            out.append(name)
    return out


# ----------------------------------------------------------------- profile_summary


def _format_list(items: list[str]) -> str:
    if not items:
        return "none"
    return ", ".join(items)


def _render_summary(
    *,
    dataset_name: str,
    row_count: int,
    n_cols: int,
    n_numeric: int,
    n_time: int,
    n_categorical: int,
    primary_key_candidates: list[str],
    join_key_candidates: list[str],
    time_columns: list[str],
    exact_fp: str,
    semantic_fp: str,
) -> str:
    """Render the deterministic template. Exactly 6 lines, no trailing newline."""
    exact_prefix = (exact_fp[:8] if exact_fp else "") + "..."
    semantic_prefix = (semantic_fp[:8] if semantic_fp else "") + "..."
    lines = [
        f"Dataset: {dataset_name}",
        (
            f"{row_count:,} rows, {n_cols} columns "
            f"({n_numeric} numeric, {n_time} datetime, {n_categorical} categorical)."
        ),
        f"PK candidates: {_format_list(primary_key_candidates)}.",
        f"Join keys: {_format_list(join_key_candidates)}.",
        f"Time columns: {_format_list(time_columns)}.",
        f"exact_fingerprint: {exact_prefix}  semantic_fingerprint: {semantic_prefix}",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------------- entry


def profile(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    file_bytes: Optional[bytes] = None,
) -> DatasetContext:
    """Compute a :class:`DatasetContext` from ``df``.

    ``file_bytes`` is required for ``exact_fingerprint``. If omitted, the
    exact_fingerprint is the empty string ``""`` — the caller can fill it
    in later by re-profiling with bytes available.
    """
    columns = [str(c) for c in df.columns]
    dtypes = {str(c): str(df[c].dtype) for c in df.columns}
    row_count = int(len(df))

    pk = _primary_key_candidates(df)
    jk = _join_key_candidates(df, pk)
    time = _time_columns(df)
    numeric = _numeric_columns(df)
    categorical = _categorical_columns(df, numeric, time)

    sem_fp = semantic_fingerprint(df)
    exact_fp = exact_fingerprint(file_bytes) if file_bytes is not None else ""

    summary = _render_summary(
        dataset_name=dataset_name,
        row_count=row_count,
        n_cols=len(columns),
        n_numeric=len(numeric),
        n_time=len(time),
        n_categorical=len(categorical),
        primary_key_candidates=pk,
        join_key_candidates=jk,
        time_columns=time,
        exact_fp=exact_fp,
        semantic_fp=sem_fp,
    )

    return DatasetContext(
        dataset_name=dataset_name,
        exact_fingerprint=exact_fp,
        semantic_fingerprint=sem_fp,
        columns=columns,
        dtypes=dtypes,
        row_count=row_count,
        primary_key_candidates=pk,
        join_key_candidates=jk,
        time_columns=time,
        numeric_columns=numeric,
        categorical_columns=categorical,
        profile_summary=summary,
    )


__all__ = ["DatasetContext", "profile"]
