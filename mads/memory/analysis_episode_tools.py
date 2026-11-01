"""LangChain tool factory for the ``analysis_episode`` memory category.

The factory closes over ``(store, user_id, project_id)`` so the LLM never
sees tenancy fields. Analysis episodes are a property of the project, not
the user — multiple users in the same project share one episode library.
``user_id`` is recorded on every write for audit, not for filtering.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import BaseTool, tool

from mads.memory.analysis_episode import (
    AnalysisEpisode,
    AnalysisEpisodeStore,
)


# ---------------------------------------------------------------- formatting


def _one_line(text: Optional[str], width: int = 80) -> str:
    """Collapse whitespace to a single space; truncate with an ellipsis."""
    if text is None:
        return ""
    flat = " ".join(text.split())
    if len(flat) > width:
        return flat[: width - 1] + "…"
    return flat


def _short_id(episode_id: str) -> str:
    return episode_id[:8] if len(episode_id) >= 8 else episode_id


def _format_full(ep: AnalysisEpisode) -> str:
    """Pretty-print one episode for the ``get`` tool."""
    lines = [
        f"Episode {ep.id} (project={ep.project_id})",
        f"  created_at:    {ep.created_at}",
        f"  updated_at:    {ep.updated_at}",
        f"  recorded_by:   {ep.user_id}",
        f"  question:      {ep.question}",
        f"  conclusion:    {ep.conclusion}",
    ]
    if ep.analysis_type:
        lines.append(f"  analysis_type: {ep.analysis_type}")
    if ep.datasets_used:
        lines.append(f"  datasets_used: {ep.datasets_used}")
    if ep.metrics_used:
        lines.append(f"  metrics_used:  {ep.metrics_used}")
    if ep.methods_used:
        lines.append(f"  methods_used:  {ep.methods_used}")
    if ep.code_refs:
        lines.append(f"  code_refs:     {ep.code_refs}")
    if ep.chart_refs:
        lines.append(f"  chart_refs:    {ep.chart_refs}")
    if ep.assumptions:
        lines.append(f"  assumptions:   {ep.assumptions}")
    if ep.user_feedback:
        lines.append(f"  user_feedback: {ep.user_feedback}")
    return "\n".join(lines)


def _format_compact_table(eps: list[AnalysisEpisode]) -> str:
    """Compact table: id (short) | created_at | question | conclusion."""
    lines = ["id        | created_at                  | question | conclusion"]
    for e in eps:
        lines.append(
            f"{_short_id(e.id)}  | {e.created_at} | "
            f"{_one_line(e.question, 60)} | {_one_line(e.conclusion, 60)}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------- factory


class AnalysisEpisodeToolFactory:
    """Holds a (store, user_id, project_id) triple and emits 6 LangChain tools.

    The LLM-visible signatures of the tools contain neither ``user_id`` nor
    ``project_id``; both are captured in the closures returned by
    :meth:`tools`.
    """

    def __init__(
        self,
        store: AnalysisEpisodeStore,
        user_id: str,
        project_id: str,
    ) -> None:
        if not isinstance(user_id, str) or not user_id:
            raise ValueError("user_id must be a non-empty string")
        if not isinstance(project_id, str) or not project_id:
            raise ValueError("project_id must be a non-empty string")
        self._store = store
        self._user_id = user_id
        self._project_id = project_id

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

        @tool
        def record_analysis_episode(
            question: str,
            conclusion: str,
            analysis_type: Optional[str] = None,
            datasets_used: Optional[str] = None,
            metrics_used: Optional[str] = None,
            methods_used: Optional[str] = None,
            code_refs: Optional[str] = None,
            chart_refs: Optional[str] = None,
            assumptions: Optional[str] = None,
            user_feedback: Optional[str] = None,
        ) -> str:
            """Record a new analysis episode (immutable snapshot of an analytical
            investigation). The question and conclusion are required; everything
            else is optional. Free-text fields (question, conclusion, assumptions,
            user_feedback) are concatenated and embedded for semantic retrieval.
            Returns the new episode's id."""
            try:
                ep = store.record(
                    user_id=user_id,
                    project_id=project_id,
                    question=question,
                    conclusion=conclusion,
                    analysis_type=analysis_type,
                    datasets_used=datasets_used,
                    metrics_used=metrics_used,
                    methods_used=methods_used,
                    code_refs=code_refs,
                    chart_refs=chart_refs,
                    assumptions=assumptions,
                    user_feedback=user_feedback,
                )
            except ValueError as exc:
                return f"Error: {exc}"
            return (
                f"Recorded analysis episode {ep.id} "
                f"(project={ep.project_id}, created_at={ep.created_at})."
            )

        @tool
        def get_analysis_episode(episode_id: str) -> str:
            """Fetch a single episode by id, formatted for review. Returns an
            empty-state message if no episode by that id exists in this project."""
            if not isinstance(episode_id, str) or not episode_id:
                return "Error: episode_id must be a non-empty string."
            ep = store.get(project_id=project_id, episode_id=episode_id)
            if ep is None:
                return (
                    f"No analysis episode with id '{episode_id}' "
                    f"in project '{project_id}'."
                )
            return _format_full(ep)

        @tool
        def list_recent_analyses(limit: int = 10) -> str:
            """List the most recent episodes in this project, newest first,
            capped at `limit`. Returns a compact table: id (short) | created_at |
            question one-liner | conclusion one-liner."""
            if not isinstance(limit, int) or limit <= 0:
                return "Error: limit must be a positive integer."
            rows = store.list_recent(project_id=project_id, limit=limit)
            if not rows:
                return (
                    f"No analysis episodes recorded yet in "
                    f"project '{project_id}'."
                )
            header = (
                f"Most recent analysis episodes in project '{project_id}' "
                f"({len(rows)}):"
            )
            return header + "\n" + _format_compact_table(rows)

        @tool
        def find_similar_analyses(query: str, k: int = 5) -> str:
            """Semantic search: embed `query` and return the k most similar
            episodes by cosine similarity over the embedded text (question +
            conclusion + assumptions + user_feedback). Use this when you want
            analyses similar in MEANING; use search_analyses for exact-term
            lookup."""
            if not isinstance(query, str) or not query.strip():
                return "Error: query must be a non-empty string."
            if not isinstance(k, int) or k <= 0:
                return "Error: k must be a positive integer."
            hits = store.find_similar(
                project_id=project_id, query=query, k=k
            )
            if not hits:
                return (
                    f"No analysis episodes matched '{query}' "
                    f"in project '{project_id}'."
                )
            lines = [
                f"Semantic matches for '{query}' in project '{project_id}' "
                f"({len(hits)}):"
            ]
            for ep, dist in hits:
                lines.append(
                    f"  {_short_id(ep.id)}  d={dist:.4f}  | "
                    f"{_one_line(ep.question, 60)} -> "
                    f"{_one_line(ep.conclusion, 60)}"
                )
            return "\n".join(lines)

        @tool
        def search_analyses(query: str) -> str:
            """Substring search (case-insensitive) over question, conclusion,
            assumptions, and user_feedback. Returns all matching episodes (no k
            limit). Use this for exact-term retrieval; for semantic similarity
            use find_similar_analyses."""
            if not isinstance(query, str) or not query:
                return "Error: query must be a non-empty string."
            rows = store.search(project_id=project_id, query=query)
            if not rows:
                return (
                    f"No analysis episodes matched '{query}' "
                    f"in project '{project_id}'."
                )
            header = (
                f"Substring matches for '{query}' in project '{project_id}' "
                f"({len(rows)}):"
            )
            return header + "\n" + _format_compact_table(rows)

        @tool
        def update_user_feedback(episode_id: str, feedback: str) -> str:
            """Update the user_feedback field on an existing episode and bump
            updated_at. This is the ONLY field that can be edited after creation.
            Triggers a re-embedding (since user_feedback is part of the embedded
            text). Returns confirmation including the new updated_at."""
            try:
                ep = store.update_feedback(
                    user_id=user_id,
                    project_id=project_id,
                    episode_id=episode_id,
                    feedback=feedback,
                )
            except ValueError as exc:
                return f"Error: {exc}"
            if ep is None:
                return (
                    f"Error: no analysis episode with id '{episode_id}' "
                    f"in project '{project_id}'."
                )
            return (
                f"Updated user_feedback on episode {ep.id} "
                f"(updated_at={ep.updated_at})."
            )

        return [
            record_analysis_episode,
            get_analysis_episode,
            list_recent_analyses,
            find_similar_analyses,
            search_analyses,
            update_user_feedback,
        ]


def make_analysis_episode_factory(
    store: AnalysisEpisodeStore,
    user_id: str,
    project_id: str,
) -> AnalysisEpisodeToolFactory:
    """Construct an :class:`AnalysisEpisodeToolFactory` bound to
    (store, user_id, project_id).

    Call ``.tools()`` on the result to get the list of 6 LangChain tools.
    """
    return AnalysisEpisodeToolFactory(store, user_id, project_id)


__all__ = [
    "AnalysisEpisodeToolFactory",
    "make_analysis_episode_factory",
]
