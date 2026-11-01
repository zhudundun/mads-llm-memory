"""LangChain tool factory for the dataset_schema memory category.

The factory closes over ``(store, df, dataset_name, table_name, user_id,
project_id)`` and pre-computes the dataset fingerprint once. The LLM never
sees user_id, project_id, dataset_name, table_name, or df.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from langchain_core.tools import BaseTool, tool

from mads.memory.dataset_schema import DatasetSchemaStore, compute_fingerprint


def _format_dataset(row: dict) -> str:
    """Pretty-print an active dataset row as a single human-readable block."""
    parts = [
        f"Dataset '{row.get('dataset_name')}' (table={row.get('table_name')})",
        f"  fingerprint: {row.get('dataset_fingerprint')}",
    ]
    if row.get("project_id"):
        parts.append(f"  project_id: {row['project_id']}")
    if row.get("grain"):
        parts.append(f"  grain: {row['grain']}")
    if row.get("primary_key"):
        parts.append(f"  primary_key: {row['primary_key']}")
    if row.get("join_keys"):
        parts.append(f"  join_keys: {row['join_keys']}")
    if row.get("time_columns"):
        parts.append(f"  time_columns: {row['time_columns']}")
    if row.get("target_columns"):
        parts.append(f"  target_columns: {row['target_columns']}")
    if row.get("known_issues"):
        parts.append(f"  known_issues: {row['known_issues']}")
    parts.append(f"  valid_from: {row.get('valid_from')}")
    return "\n".join(parts)


def _format_column(row: dict) -> str:
    """Pretty-print an active column row."""
    parts = [
        f"Column '{row.get('column_name')}' (table={row.get('table_name')})",
    ]
    if row.get("semantic_type"):
        parts.append(f"  semantic_type: {row['semantic_type']}")
    if row.get("business_meaning"):
        parts.append(f"  business_meaning: {row['business_meaning']}")
    if row.get("pii_level"):
        parts.append(f"  pii_level: {row['pii_level']}")
    if row.get("example_values") is not None:
        parts.append(f"  example_values: {row['example_values']}")
    if row.get("null_behavior"):
        parts.append(f"  null_behavior: {row['null_behavior']}")
    parts.append(f"  valid_from: {row.get('valid_from')}")
    return "\n".join(parts)


def make_dataset_tools(
    store: DatasetSchemaStore,
    df: pd.DataFrame,
    *,
    dataset_name: str,
    table_name: str,
    user_id: str,
    project_id: Optional[str] = None,
) -> list[BaseTool]:
    """Return 6 ``@tool`` callables bound via closure to (store, df, names, user_id).

    The fingerprint is computed once at factory time and reused by every tool.
    The LLM never sees ``user_id``, ``project_id``, ``dataset_name``,
    ``table_name``, or the ``df`` itself.
    """
    fingerprint = compute_fingerprint(df)
    bound_columns = {str(c) for c in df.columns}

    @tool
    def register_dataset(
        grain: Optional[str] = None,
        primary_key: Optional[list[str]] = None,
        join_keys: Optional[list[dict]] = None,
        time_columns: Optional[list[str]] = None,
        target_columns: Optional[list[str]] = None,
        known_issues: Optional[list[str]] = None,
    ) -> str:
        """Record dataset-level facts about the current table (grain, primary key,
        join keys, time/target columns, known issues). Supersedes any prior
        active record for this dataset. Pass only fields you want to update;
        omitted fields keep their prior values."""
        new_id, fp = store.register_dataset(
            user_id=user_id,
            df=df,
            dataset_name=dataset_name,
            table_name=table_name,
            project_id=project_id,
            grain=grain,
            primary_key=primary_key,
            join_keys=join_keys,
            time_columns=time_columns,
            target_columns=target_columns,
            known_issues=known_issues,
        )
        set_fields = []
        for label, val in [
            ("grain", grain),
            ("primary_key", primary_key),
            ("join_keys", join_keys),
            ("time_columns", time_columns),
            ("target_columns", target_columns),
            ("known_issues", known_issues),
        ]:
            if val is not None:
                set_fields.append(f"{label}={val}")
        suffix = "; ".join(set_fields) if set_fields else "no fields set (record created)"
        return (
            f"Registered dataset '{dataset_name}' (table={table_name}, "
            f"fingerprint={fp}, id={new_id}). {suffix}."
        )

    @tool
    def recall_dataset() -> str:
        """Look up the recorded dataset-level facts for the current table.
        Returns grain, primary key, join keys, time/target columns, known issues."""
        row = store.get_dataset(
            user_id=user_id, fingerprint=fingerprint, table_name=table_name
        )
        if row is None:
            return f"No dataset memory for '{dataset_name}' (table={table_name})."
        return _format_dataset(row)

    @tool
    def list_datasets() -> str:
        """List every dataset the user has registered facts about, one per line."""
        rows = store.list_datasets(user_id)
        if not rows:
            return "No datasets registered yet."
        lines = [f"You have memory for {len(rows)} dataset(s):"]
        for r in rows:
            lines.append(
                f"- {r['dataset_name']} (table={r['table_name']}, "
                f"fingerprint={r['dataset_fingerprint']}, grain={r.get('grain')})"
            )
        return "\n".join(lines)

    @tool
    def register_column(
        column_name: str,
        semantic_type: Optional[str] = None,
        business_meaning: Optional[str] = None,
        pii_level: Optional[str] = None,
        example_values: Optional[list] = None,
        null_behavior: Optional[str] = None,
    ) -> str:
        """Record column-level facts (semantic type, business meaning, PII level,
        example values, null behavior) for one column in the current table.
        ``pii_level`` must be one of 'none', 'low', 'medium', 'high' if provided.
        Supersedes any prior active record for this column."""
        if column_name not in bound_columns:
            return (
                f"Error: column '{column_name}' is not in the current table "
                f"'{table_name}'. Available columns: {sorted(bound_columns)}."
            )
        try:
            new_id = store.register_column(
                fingerprint=fingerprint,
                table_name=table_name,
                column_name=column_name,
                semantic_type=semantic_type,
                business_meaning=business_meaning,
                pii_level=pii_level,
                example_values=example_values,
                null_behavior=null_behavior,
            )
        except ValueError as exc:
            return f"Error: {exc}"
        return (
            f"Registered column '{column_name}' on table '{table_name}' "
            f"(id={new_id})."
        )

    @tool
    def recall_column(column_name: str) -> str:
        """Look up the recorded facts for a single column in the current table."""
        row = store.get_column(
            fingerprint=fingerprint,
            table_name=table_name,
            column_name=column_name,
        )
        if row is None:
            return f"No column memory for '{column_name}'."
        return _format_column(row)

    @tool
    def list_columns() -> str:
        """List every column with recorded facts for the current table,
        sorted alphabetically by column name."""
        rows = store.list_columns(fingerprint=fingerprint, table_name=table_name)
        if not rows:
            return f"No column memory for table '{table_name}'."
        lines = [f"You have memory for {len(rows)} column(s) on '{table_name}':"]
        for r in rows:
            lines.append(
                f"- {r['column_name']}: semantic_type={r.get('semantic_type')}, "
                f"pii_level={r.get('pii_level')}, "
                f"business_meaning={r.get('business_meaning')}"
            )
        return "\n".join(lines)

    return [
        register_dataset,
        recall_dataset,
        list_datasets,
        register_column,
        recall_column,
        list_columns,
    ]
