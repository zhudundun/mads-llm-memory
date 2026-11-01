"""LangChain tool factory for the ``metric_definition`` memory category.

The factory closes over ``(store, user_id, project_id)`` so the LLM never
sees tenancy fields. Multiple users in the same project share one metric
library — only ``user_id`` differs across their tool closures and it is
recorded on every write for audit, not for visibility filtering.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import BaseTool, tool

from mads.memory.metric_definition import (
    MetricDefinition,
    MetricDefinitionStore,
)
from mads.memory.retrievers.metric_retriever import MetricRetriever
from mads.memory.retrievers.types import RetrievedMemory


# ---------------------------------------------------------------- formatting


def _one_line(text: Optional[str], width: int = 80) -> str:
    """Collapse whitespace to a single space; truncate with an ellipsis."""
    if text is None:
        return ""
    flat = " ".join(text.split())
    if len(flat) > width:
        return flat[: width - 1] + "…"
    return flat


def _format_full(m: MetricDefinition) -> str:
    """Pretty-print the full active definition of one metric."""
    parts = [
        f"Metric '{m.metric_name}' (project={m.project_id})",
        f"  normalized: {m.normalized_name}",
        f"  business_definition: {m.business_definition}",
    ]
    if m.sql_definition:
        parts.append(f"  sql_definition: {m.sql_definition}")
    if m.python_definition:
        parts.append(f"  python_definition: {m.python_definition}")
    if m.default_filters:
        parts.append(f"  default_filters: {m.default_filters}")
    parts.append(f"  confidence: {m.confidence}")
    if m.source:
        parts.append(f"  source: {m.source}")
    parts.append(f"  valid_from: {m.valid_from}")
    parts.append(f"  last_edited_by: {m.user_id}")
    return "\n".join(parts)


def _format_table(rows: list[MetricDefinition]) -> str:
    """Compact table: name | one-line business definition | confidence | source."""
    lines = ["name | business_definition | confidence | source"]
    for m in rows:
        lines.append(
            f"{m.metric_name} | {_one_line(m.business_definition)} "
            f"| {m.confidence} | {m.source or '-'}"
        )
    return "\n".join(lines)


def _format_retrieved_table(retrieved: list[RetrievedMemory]) -> str:
    """Compact table for retriever output: includes score + reason columns.

    Mirrors :func:`_format_table` but prepends ``score`` and ``reason`` so
    the LLM sees *why* each match surfaced (exact / substring / semantic).
    """
    lines = ["score | name | business_definition | confidence | source | reason"]
    for rm in retrieved:
        mem = rm.memory or {}
        lines.append(
            f"{rm.score:.2f} | {mem.get('metric_name', '?')} "
            f"| {_one_line(mem.get('business_definition'))} "
            f"| {mem.get('confidence', '-')} "
            f"| {mem.get('source') or '-'} "
            f"| {rm.reason}"
        )
    return "\n".join(lines)


def _version_number(history: list[MetricDefinition], target_id: str) -> int:
    """Return the 1-based index of ``target_id`` in oldest-first history."""
    for idx, m in enumerate(history, start=1):
        if m.id == target_id:
            return idx
    # Should not happen — the row we just wrote must be in history.
    return len(history)


# ---------------------------------------------------------------- factory


class MetricToolFactory:
    """Holds a (store, user_id, project_id) triple and emits the 5 LangChain tools.

    The LLM-visible signatures of the tools contain neither ``user_id`` nor
    ``project_id``; both are captured in the closures returned by
    :meth:`tools`.
    """

    def __init__(
        self,
        store: MetricDefinitionStore,
        user_id: str,
        project_id: str,
        *,
        retriever: Optional[MetricRetriever] = None,
    ) -> None:
        if not isinstance(user_id, str) or not user_id:
            raise ValueError("user_id must be a non-empty string")
        if not isinstance(project_id, str) or not project_id:
            raise ValueError("project_id must be a non-empty string")
        self._store = store
        self._user_id = user_id
        self._project_id = project_id
        # B3: search_metric_definitions routes through MetricRetriever so
        # both reactive (tool-call) and proactive (HybridRetriever)
        # retrieval share one ranking implementation. If the caller
        # didn't pass a retriever, build one off the store (it'll
        # inherit the store's embedder if one is wired).
        self._retriever: MetricRetriever = retriever or MetricRetriever(store)

    @property
    def user_id(self) -> str:
        return self._user_id

    @property
    def project_id(self) -> str:
        return self._project_id

    def tools(self) -> list[BaseTool]:
        store = self._store
        user_id = self._user_id
        project_id = self._project_id
        retriever = self._retriever

        @tool
        def add_metric_definition(
            metric_name: str,
            business_definition: str,
            sql_definition: Optional[str] = None,
            python_definition: Optional[str] = None,
            default_filters: Optional[str] = None,
            confidence: float = 1.0,
            source: Optional[str] = None,
            required_columns: Optional[list[str]] = None,
        ) -> str:
            """Register or update a metric. If a metric with the same name already exists
            in this project and any of the definitional fields (business_definition,
            sql_definition, python_definition, default_filters, required_columns) changed,
            a new version is created and the previous version is closed. If only
            confidence/source changed, the current version is updated in place. Returns
            a human-readable confirmation including the version number.

            If your metric depends on specific dataframe columns (e.g. a filter
            ``tenure > 12`` needs ``tenure``), pass them as
            ``required_columns=['tenure']`` so the memory layer can flag the
            metric as inapplicable when those columns are absent from the
            bound dataset. If you omit ``required_columns`` AND your
            ``sql_definition`` is non-empty, the store auto-parses identifier
            tokens out of the SQL as a best-effort fallback. If your
            definition is prose-only and column-dependent (e.g. "made a payment
            in the last 30 days" needs ``last_payment_date``), passing
            ``required_columns`` explicitly is highly recommended — without it,
            the system cannot detect schema mismatches and may silently
            substitute a different definition."""
            try:
                definition, action = store.upsert(
                    user_id=user_id,
                    project_id=project_id,
                    metric_name=metric_name,
                    business_definition=business_definition,
                    sql_definition=sql_definition,
                    python_definition=python_definition,
                    default_filters=default_filters,
                    confidence=confidence,
                    source=source,
                    required_columns=required_columns,
                )
            except ValueError as exc:
                return f"Error: {exc}"

            history = store.history(
                project_id=project_id, metric_name=metric_name
            )
            version = _version_number(history, definition.id)
            return (
                f"Metric '{definition.metric_name}' {action} "
                f"(version {version}, id={definition.id}, "
                f"confidence={definition.confidence}, "
                f"source={definition.source or '-'})."
            )

        @tool
        def get_metric_definition(metric_name: str) -> str:
            """Return the current (active) definition of a metric in this project,
            formatted for analyst consumption. Returns an empty-state message if no
            metric by that name exists."""
            try:
                m = store.get_active(
                    project_id=project_id, metric_name=metric_name
                )
            except ValueError as exc:
                return f"Error: {exc}"
            if m is None:
                return f"No metric named '{metric_name}' in this project."
            return _format_full(m)

        @tool
        def list_metric_definitions() -> str:
            """List all currently-active metrics in this project as a compact table:
            name | one-line business definition | confidence | source. Returns an
            empty-state message if no metrics are defined."""
            rows = store.list_active(project_id=project_id)
            if not rows:
                return "No metrics defined in this project."
            header = f"Active metrics in project '{project_id}' ({len(rows)}):"
            return header + "\n" + _format_table(rows)

        @tool
        def search_metric_definitions(query: str) -> str:
            """Substring or semantic search over active metrics in this project;
            returns up to 5 ranked matches with score and reason. Matches by
            exact name, substring of the metric name, tokenized whole-token
            match against the query, OR semantic similarity over the
            (metric_name + business_definition) embedding when an embedder is
            wired. Best match first."""
            if not isinstance(query, str) or not query:
                return "Error: query must be a non-empty string."
            results = retriever.retrieve(query, project_id=project_id, k=5)
            if not results:
                return f"No metrics matched '{query}'."
            header = f"Matches for '{query}' in project '{project_id}' ({len(results)}):"
            return header + "\n" + _format_retrieved_table(results)

        @tool
        def get_metric_history(metric_name: str) -> str:
            """Return the full version history of a metric in this project, oldest
            first, showing valid_from / valid_to / business_definition / who changed
            it (user_id). Returns empty-state message if no history exists."""
            try:
                rows = store.history(
                    project_id=project_id, metric_name=metric_name
                )
            except ValueError as exc:
                return f"Error: {exc}"
            if not rows:
                return f"No history for metric '{metric_name}' in this project."
            lines = [
                f"History for '{metric_name}' in project '{project_id}' "
                f"({len(rows)} version(s)):"
            ]
            for i, m in enumerate(rows, start=1):
                lines.append(
                    f"  v{i}: {m.valid_from} -> {m.valid_to or 'active'} "
                    f"| by {m.user_id} | {_one_line(m.business_definition)}"
                )
            return "\n".join(lines)

        return [
            add_metric_definition,
            get_metric_definition,
            list_metric_definitions,
            search_metric_definitions,
            get_metric_history,
        ]


def make_metric_factory(
    store: MetricDefinitionStore,
    user_id: str,
    project_id: str,
    *,
    retriever: Optional[MetricRetriever] = None,
) -> MetricToolFactory:
    """Construct a :class:`MetricToolFactory` bound to (store, user_id, project_id).

    Optionally accepts a pre-built :class:`MetricRetriever` (the
    MemoryManager passes one so the tool's ``search_metric_definitions``
    and the proactive HybridRetriever share the same hybrid scoring).
    When omitted, the factory builds a retriever from the store —
    inheriting the store's embedder if one is wired.

    Call ``.tools()`` on the result to get the list of 5 LangChain tools.
    """
    return MetricToolFactory(store, user_id, project_id, retriever=retriever)
