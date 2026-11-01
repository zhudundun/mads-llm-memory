"""LangChain tool factory for the user_preference memory category.

The factory closes over ``(store, user_id, project_id,
get_dataset_fingerprint)`` so the LLM never sees tenancy fields or the
dataset fingerprint.

Scope semantics (per the v1.3 axis model):

* Preferences default to GLOBAL scope
  (``project_id=None, dataset_fingerprint=None``) — they apply to YOU
  as a user across every project and every dataset. This is the right
  default for personal style: column naming conventions, output format,
  chart preferences.
* Passing ``project_scoped=True`` on ``remember_preference`` attaches
  the *current project's* ``project_id`` to the row, so the preference
  is scoped to this project only and will NOT bleed into other
  projects.
* Passing ``dataset_scoped=True`` on ``remember_preference`` attaches
  the *current dataset's* ``semantic_fingerprint`` to the row, so the
  preference is scoped to this dataset only and will NOT bleed into
  other datasets.
* The two flags COMPOSE: a row with both ``project_id`` and
  ``dataset_fingerprint`` populated applies only to that
  (project, dataset) pair.
* On retrieval, the MOST-SPECIFIC eligible row wins
  (dataset+project > dataset > project > global).

Write-side downgrade (v1.3 behavior B'):

* If ``dataset_scoped=True`` but no df is currently bound to the
  manager (``get_dataset_fingerprint() is None``), the row is stored
  WITHOUT a fingerprint (so it ends up at the next-most-specific
  scope: project-only if ``project_scoped=True``, else global). The
  returned string includes an explicit warning so the LLM can re-record
  once a dataset is loaded.
* The downgrade NEVER raises; this is a soft fallback designed for
  loader's first turn where the LLM may set preferences before the df
  is bound.

The factory's ``project_id`` parameter is the closure-bound current
project; ``get_dataset_fingerprint`` is a closure that returns the
CURRENT semantic fingerprint of the manager's bound df (or ``None``).
Both are NEVER exposed to the LLM — the LLM picks scope by setting
the keywords ``project_scoped: bool`` and ``dataset_scoped: bool``.
"""

from __future__ import annotations

from typing import Callable, Optional

from langchain_core.tools import BaseTool, tool

from mads.memory.user_preference import PreferenceStore


def _scope_label(project_id: Optional[str], dataset_fp: Optional[str]) -> str:
    """Human-readable label for the LLM-facing reason strings."""
    if project_id is not None and dataset_fp is not None:
        return "project+dataset"
    if dataset_fp is not None:
        return "dataset"
    if project_id is not None:
        return "project"
    return "global"


def make_preference_tools(
    store: PreferenceStore,
    user_id: str,
    project_id: Optional[str] = None,
    *,
    get_dataset_fingerprint: Optional[Callable[[], Optional[str]]] = None,
) -> list[BaseTool]:
    """Return 4 @tool-decorated functions bound to
    ``(store, user_id, project_id, get_dataset_fingerprint)``.

    The LLM never sees ``user_id``, ``project_id``, or the dataset
    fingerprint. It can choose scope by passing ``project_scoped: bool``
    and/or ``dataset_scoped: bool`` to the relevant tools.

    ``get_dataset_fingerprint`` is a callable returning the CURRENT
    fingerprint of the bound df at tool-call time (live read, not
    snapshot), so a ``bind_dataframe`` call mid-conversation is
    reflected without rebuilding the tools. When ``None`` (default), no
    dataset is bound and ``dataset_scoped=True`` downgrades with a
    warning string.

    If ``project_id`` is None (no current project bound), passing
    ``project_scoped=True`` still works but writes ``project_id=None``
    (effectively a global pref); in practice the manager always passes
    a non-empty ``project_id`` so this is only relevant in unit tests.
    """

    # Normalise the dataset-fingerprint accessor so the tool callables
    # below can call it unconditionally.
    if get_dataset_fingerprint is None:
        def _get_fp() -> Optional[str]:
            return None
    else:
        _get_fp = get_dataset_fingerprint

    @tool
    def remember_preference(
        key: str,
        value: str,
        project_scoped: bool = False,
        dataset_scoped: bool = False,
        source: str = "user_explicit",
        confidence: float = 1.0,
    ) -> str:
        """Save a user preference as a key/value pair.

        Scope:
          - Default (project_scoped=False, dataset_scoped=False) → GLOBAL
            preference: applies to this user across ALL projects and ALL
            datasets. Use for PERSONAL STYLE that should follow the user
            everywhere: preferred chart library, preferred output
            format, color schemes, report length.
          - project_scoped=True → PROJECT-SCOPED preference: applies ONLY
            to the current project; will NOT bleed into other projects.
            Use for project-specific business rules and conventions.
          - dataset_scoped=True → DATASET-SCOPED preference: applies ONLY
            when the current dataset is bound; will NOT bleed onto
            other datasets. Use for dataset-specific rules:
            "rename customerID to customer_id on the churn dataset",
            "treat MonthlyCharges as USD on this dataset".
          - The two flags COMPOSE: project+dataset is the most specific.

        If you set dataset_scoped=True but no dataset is loaded yet, the
        preference is recorded at the next-most-specific scope (project
        if also project_scoped, else global) and the return value will
        explicitly say so. You may re-record once the dataset is loaded
        if you actually want dataset-scoped behavior.

        Choosing well matters: a dataset-specific rule saved globally
        will surface on UNRELATED datasets and confuse the agent.

        Supersedes any prior value for the same (key, scope) tuple.
        Rows in other scope combinations are NOT affected. On retrieval
        the most-specific row wins (dataset+project > dataset > project
        > global).
        """
        actual_project = project_id if project_scoped else None
        live_fp = _get_fp() if dataset_scoped else None

        downgraded = dataset_scoped and live_fp is None

        new_id = store.set(
            user_id,
            key,
            value,
            project_id=actual_project,
            dataset_fingerprint=live_fp,
            source=source,
            confidence=confidence,
        )

        if downgraded:
            # B' fallback: requested dataset scope but no df bound.
            # Stored at the next-most-specific scope; warn explicitly.
            downgrade_target = "project" if actual_project is not None else "global"
            return (
                f"Saved preference '{key}' = '{value}' (id={new_id}). "
                f"recorded (downgraded to {downgrade_target} scope — no "
                f"dataset was bound; load a dataset and re-record if you "
                f"want dataset-scoped)."
            )

        scope_label = _scope_label(actual_project, live_fp)
        return (
            f"Saved preference '{key}' = '{value}' "
            f"(scope: {scope_label}, source={source}, "
            f"confidence={confidence}, id={new_id})."
        )

    @tool
    def recall_preference(key: str) -> str:
        """Look up the current value of a single user preference by key.

        Returns the MOST-SPECIFIC matching preference for the current
        scope (dataset+project > dataset > project > global), or a 'not
        set' message. The scope used is reported alongside the value so
        callers can tell which level applied.
        """
        live_fp = _get_fp()
        row = store.get(
            user_id,
            key,
            project_id=project_id,
            dataset_fingerprint=live_fp,
        )
        if row is None:
            return f"No preference set for '{key}'."
        scope_label = _scope_label(row.get("project_id"), row.get("dataset_fingerprint"))
        return (
            f"Your preference for '{key}' is: {row['preference_value']} "
            f"(scope: {scope_label})."
        )

    @tool
    def list_preferences() -> str:
        """List every applicable preference for the current user, one per line.

        Returns the EFFECTIVE view: one row per ``preference_key``,
        picking the most-specific applicable scope (dataset+project >
        dataset > project > global). Each line includes the effective
        scope so the user can see which level applied.
        """
        live_fp = _get_fp()
        rows = store.list_active(
            user_id,
            project_id=project_id,
            dataset_fingerprint=live_fp,
        )
        if not rows:
            return "No preferences set yet."
        lines = [f"You have {len(rows)} preferences:"]
        for r in rows:
            scope_label = _scope_label(
                r.get("project_id"), r.get("dataset_fingerprint")
            )
            lines.append(
                f"- {r['preference_key']}: {r['preference_value']} "
                f"(scope: {scope_label}, source={r['source']}, "
                f"confidence={r['confidence']})"
            )
        return "\n".join(lines)

    @tool
    def forget_preference(
        key: str,
        project_scoped: bool = False,
        dataset_scoped: bool = False,
    ) -> str:
        """Forget (close) the active preference for the given key.

        Scope flags:
          - Default (both False) → forget the GLOBAL pref for this key.
          - project_scoped=True → forget the CURRENT PROJECT's row
            (leaves the global pref untouched).
          - dataset_scoped=True → forget the CURRENT DATASET's row
            (leaves the global and project-scoped rows untouched). If
            no dataset is bound, this is treated as an unbound-dataset
            no-op rather than silently forgetting the project/global
            row.
          - Both flags True → forget the project+dataset row.

        Does not insert a replacement; history is retained for audit.
        """
        actual_project = project_id if project_scoped else None
        live_fp = _get_fp() if dataset_scoped else None

        if dataset_scoped and live_fp is None:
            # Treat as a no-op rather than fall through and silently
            # close a less-specific row — that would be surprising.
            return (
                f"No active preference for '{key}' at the requested "
                f"dataset scope (no dataset is currently bound)."
            )

        ok = store.forget(
            user_id,
            key,
            project_id=actual_project,
            dataset_fingerprint=live_fp,
        )
        if ok:
            scope_label = _scope_label(actual_project, live_fp)
            return f"Forgot preference '{key}' (scope: {scope_label})."
        return f"No active preference for '{key}'."

    return [
        remember_preference,
        recall_preference,
        list_preferences,
        forget_preference,
    ]
