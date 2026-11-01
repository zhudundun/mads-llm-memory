"""LangChain tool factory for the ``data_quality_issue`` memory category.

The factory closes over ``(store, user_id, project_id, dataset_fingerprint)``
so the LLM never sees tenancy or fingerprint fields. Under the v1.2 axis
model, ``dataset_fingerprint`` is IDENTITY_REQUIRED — the fingerprint
must be bound from the manager's currently-bound df before any
data-quality tools are emitted. ``MemoryManager.all_tools`` excludes
data-quality tools when no df is bound (``include_dataset=False``).
"""

from __future__ import annotations

from typing import Literal, Optional

from langchain_core.tools import BaseTool, tool

from mads.memory.data_quality import (
    DataQualityIssue,
    DataQualityStore,
)


# ---------------------------------------------------------------- formatting


def _one_line(text: Optional[str], width: int = 80) -> str:
    if text is None:
        return ""
    flat = " ".join(text.split())
    if len(flat) > width:
        return flat[: width - 1] + "…"
    return flat


def _format_full(issue: DataQualityIssue) -> str:
    col = issue.column_name if issue.column_name is not None else "(table-level)"
    parts = [
        f"Issue '{issue.issue_type}' on {issue.table_name}.{col} "
        f"(project={issue.project_id})",
        f"  severity: {issue.severity}",
        f"  description: {issue.issue_description}",
    ]
    if issue.recommended_handling:
        parts.append(f"  recommended_handling: {issue.recommended_handling}")
    parts.append(f"  confidence: {issue.confidence}")
    if issue.dataset_fingerprint:
        parts.append(f"  dataset_fingerprint: {issue.dataset_fingerprint}")
    parts.append(f"  valid_from: {issue.valid_from}")
    parts.append(f"  last_edited_by: {issue.user_id}")
    return "\n".join(parts)


def _format_table(rows: list[DataQualityIssue]) -> str:
    lines = ["table.column | issue_type | severity | description"]
    for r in rows:
        col = r.column_name if r.column_name is not None else "(table)"
        lines.append(
            f"{r.table_name}.{col} | {r.issue_type} | {r.severity} "
            f"| {_one_line(r.issue_description)}"
        )
    return "\n".join(lines)


def _version_number(history: list[DataQualityIssue], target_id: str) -> int:
    for idx, issue in enumerate(history, start=1):
        if issue.id == target_id:
            return idx
    return len(history)


# ---------------------------------------------------------------- factory


class DataQualityToolFactory:
    """Holds (store, user_id, project_id, dataset_fingerprint) and emits 6 tools.

    The LLM-visible signatures of the tools contain NONE of the
    tenancy / fingerprint fields; all four are closure-bound here.
    """

    def __init__(
        self,
        store: DataQualityStore,
        user_id: str,
        project_id: str,
        dataset_fingerprint: str,
    ) -> None:
        if not isinstance(user_id, str) or not user_id:
            raise ValueError("user_id must be a non-empty string")
        if not isinstance(project_id, str) or not project_id:
            raise ValueError("project_id must be a non-empty string")
        if (
            not isinstance(dataset_fingerprint, str)
            or not dataset_fingerprint
        ):
            raise ValueError(
                "dataset_fingerprint must be a non-empty string"
            )
        self._store = store
        self._user_id = user_id
        self._project_id = project_id
        self._dataset_fingerprint = dataset_fingerprint

    @property
    def user_id(self) -> str:
        return self._user_id

    @property
    def project_id(self) -> str:
        return self._project_id

    @property
    def dataset_fingerprint(self) -> str:
        return self._dataset_fingerprint

    def tools(self) -> list[BaseTool]:
        store = self._store
        user_id = self._user_id
        project_id = self._project_id
        dataset_fingerprint = self._dataset_fingerprint

        @tool
        def add_data_quality_issue(
            table_name: str,
            issue_type: str,
            issue_description: str,
            column_name: Optional[str] = None,
            recommended_handling: Optional[str] = None,
            severity: Literal["info", "warning", "critical"] = "warning",
            confidence: float = 1.0,
        ) -> str:
            """Record a data-quality observation about a table or column.

            Identity is (table_name, column_name, issue_type) within the CURRENT
            DATASET (identified by its semantic fingerprint). If an active issue
            with the same identity exists and the definitional fields
            (issue_description, recommended_handling) changed, a new version is
            created and the previous version is closed. If only severity /
            confidence changed, the current version is updated in place.
            issue_type is free-form but conventionally one of: temporal_lag,
            reliability_conditional, semantic_null, row_filter_required,
            outlier_pattern, encoding_issue, duplicate_pattern,
            missing_data_pattern. Pass column_name=None for table-level issues."""
            try:
                issue, action = store.upsert(
                    user_id=user_id,
                    project_id=project_id,
                    dataset_fingerprint=dataset_fingerprint,
                    table_name=table_name,
                    column_name=column_name,
                    issue_type=issue_type,
                    issue_description=issue_description,
                    recommended_handling=recommended_handling,
                    severity=severity,
                    confidence=confidence,
                )
            except ValueError as exc:
                return f"Error: {exc}"

            history = store.history(
                dataset_fingerprint=dataset_fingerprint,
                table_name=table_name,
                column_name=column_name,
                issue_type=issue_type,
            )
            version = _version_number(history, issue.id)
            col = (
                issue.column_name
                if issue.column_name is not None
                else "(table-level)"
            )
            return (
                f"Issue '{issue.issue_type}' on {issue.table_name}.{col} "
                f"{action} (version {version}, id={issue.id}, "
                f"severity={issue.severity}, confidence={issue.confidence})."
            )

        @tool
        def get_data_quality_issues_for_column(
            table_name: str,
            column_name: str,
        ) -> str:
            """Return all currently-active data-quality issues recorded against a
            specific (table_name, column_name) for the current dataset. Returns
            one entry per active issue_type. Empty-state message if none."""
            try:
                rows = store.get_active_for_column(
                    dataset_fingerprint=dataset_fingerprint,
                    table_name=table_name,
                    column_name=column_name,
                )
            except ValueError as exc:
                return f"Error: {exc}"
            if not rows:
                return (
                    f"No active data-quality issues on "
                    f"{table_name}.{column_name} for this dataset."
                )
            header = (
                f"Active data-quality issues on "
                f"{table_name}.{column_name} for this dataset "
                f"({len(rows)}):"
            )
            return header + "\n" + _format_table(rows)

        @tool
        def list_data_quality_issues_for_table(
            table_name: str,
        ) -> str:
            """Return all currently-active data-quality issues anchored to this
            table for the current dataset, including both column-level
            (column_name is set) and table-level (column_name is null).
            Empty-state message if none."""
            try:
                rows = store.list_active_for_table(
                    dataset_fingerprint=dataset_fingerprint,
                    table_name=table_name,
                )
            except ValueError as exc:
                return f"Error: {exc}"
            if not rows:
                return (
                    f"No active data-quality issues on table "
                    f"'{table_name}' for this dataset."
                )
            header = (
                f"Active data-quality issues on table '{table_name}' "
                f"for this dataset ({len(rows)}):"
            )
            return header + "\n" + _format_table(rows)

        @tool
        def search_data_quality_issues(query: str) -> str:
            """Substring search (case-insensitive) across active issues for the
            current dataset. Matches against issue_description,
            recommended_handling, table_name, column_name, and issue_type."""
            if not isinstance(query, str) or not query:
                return "Error: query must be a non-empty string."
            rows = store.search_active(
                dataset_fingerprint=dataset_fingerprint, query=query
            )
            if not rows:
                return f"No data-quality issues matched '{query}'."
            header = (
                f"Matches for '{query}' on this dataset ({len(rows)}):"
            )
            return header + "\n" + _format_table(rows)

        @tool
        def get_data_quality_issue_history(
            table_name: str,
            issue_type: str,
            column_name: Optional[str] = None,
        ) -> str:
            """Return the full version history of a data-quality issue on the
            current dataset, oldest first. Includes both active and closed
            (resolved) rows. Pass column_name=None to look up table-level issue
            history."""
            try:
                rows = store.history(
                    dataset_fingerprint=dataset_fingerprint,
                    table_name=table_name,
                    column_name=column_name,
                    issue_type=issue_type,
                )
            except ValueError as exc:
                return f"Error: {exc}"
            col = column_name if column_name is not None else "(table-level)"
            if not rows:
                return (
                    f"No history for issue '{issue_type}' on "
                    f"{table_name}.{col} for this dataset."
                )
            lines = [
                f"History for issue '{issue_type}' on {table_name}.{col} "
                f"for this dataset ({len(rows)} version(s)):"
            ]
            for i, r in enumerate(rows, start=1):
                lines.append(
                    f"  v{i}: {r.valid_from} -> {r.valid_to or 'active'} "
                    f"| severity={r.severity} | by {r.user_id} "
                    f"| {_one_line(r.issue_description)}"
                )
            return "\n".join(lines)

        @tool
        def resolve_data_quality_issue(
            table_name: str,
            issue_type: str,
            column_name: Optional[str] = None,
        ) -> str:
            """Mark a currently-active data-quality issue on the current dataset
            as resolved. Sets valid_to=CURRENT_TIMESTAMP on the active row with
            NO replacement row. The issue exits the active-issues list but
            remains in history. Returns an error message if no active issue
            matches the identity tuple."""
            try:
                closed = store.resolve(
                    user_id=user_id,
                    project_id=project_id,
                    dataset_fingerprint=dataset_fingerprint,
                    table_name=table_name,
                    column_name=column_name,
                    issue_type=issue_type,
                )
            except ValueError as exc:
                return f"Error: {exc}"
            col = column_name if column_name is not None else "(table-level)"
            if closed is None:
                return (
                    f"Error: no active issue '{issue_type}' on "
                    f"{table_name}.{col} for this dataset."
                )
            return (
                f"Issue '{closed.issue_type}' on {closed.table_name}.{col} "
                f"resolved (id={closed.id}, valid_to={closed.valid_to})."
            )

        return [
            add_data_quality_issue,
            get_data_quality_issues_for_column,
            list_data_quality_issues_for_table,
            search_data_quality_issues,
            get_data_quality_issue_history,
            resolve_data_quality_issue,
        ]


def make_data_quality_factory(
    store: DataQualityStore,
    user_id: str,
    project_id: str,
    dataset_fingerprint: str,
) -> DataQualityToolFactory:
    """Construct a :class:`DataQualityToolFactory` bound to
    ``(store, user_id, project_id, dataset_fingerprint)``.

    Call ``.tools()`` on the result to get the list of 6 LangChain tools.
    """
    return DataQualityToolFactory(
        store, user_id, project_id, dataset_fingerprint
    )
