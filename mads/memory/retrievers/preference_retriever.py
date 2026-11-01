"""preference_retriever — surface user preferences as user-state.

Unlike the other per-category retrievers, this one is *not* query-driven.
Preferences are user-state: they always apply (a stored "always slice by
tenure" preference is true on the next turn regardless of what the user
asks). So the retriever ignores any query string and returns active rows
for ``user_id`` (filtered by IDENTITY_OPTIONAL project_id and
IDENTITY_OPTIONAL dataset_fingerprint) with a uniform ``score=1.0``.

Scope semantics (v1.3 axis model):

* ``user_id`` is IDENTITY_REQUIRED.
* ``project_id`` is IDENTITY_OPTIONAL — when None, the session has no
  project bound, so only rows with ``project_id IS NULL`` are eligible;
  when set, rows with ``project_id IS NULL`` OR ``project_id = ?`` are
  eligible (the IDENTITY_OPTIONAL fan-in).
* ``dataset_fingerprint`` is IDENTITY_OPTIONAL — when None, the session
  has no df bound, so only rows with ``dataset_fingerprint IS NULL``
  are eligible; when set, rows with ``dataset_fingerprint IS NULL`` OR
  ``dataset_fingerprint = ?`` are eligible.

Most-specific wins. When two eligible rows share the same
``preference_key`` (e.g. a global pref AND a project-scoped pref), we
return only the MOST-SPECIFIC of them so the LLM doesn't see three
copies of the same key with different scopes.

Why score 1.0 across the board:

* There is no concept of relevance against the query; the value is "I
  remember this about you".
* The :class:`HybridRetriever` already sorts by ``(-score, category_order)``;
  giving preferences ``score=1.0`` and ranking them *first* among score-1.0
  rows ensures they sit at the top of the injected context block, which is
  where the LLM is most likely to read them.

The retriever is internal infrastructure — tenancy is closure-bound at the
manager / wrapper level, never an LLM-visible argument.
"""

from __future__ import annotations

from typing import Optional

from mads.memory.retrievers.types import RetrievedMemory
from mads.memory.user_preference import PreferenceStore


def _scope_label(project_id: Optional[str], dataset_fp: Optional[str]) -> str:
    if project_id is not None and dataset_fp is not None:
        return "project+dataset"
    if dataset_fp is not None:
        return "dataset"
    if project_id is not None:
        return "project"
    return "global"


class PreferenceRetriever:
    """Retrieves applicable preferences for a user (and optionally a
    project and dataset).

    The result list is sorted by ``preference_key`` ascending. The
    underlying :meth:`PreferenceStore.list_active` collapses to one
    row per key (most-specific wins) so the retriever doesn't surface
    duplicate keys.
    """

    def __init__(self, store: PreferenceStore) -> None:
        self._store = store

    def retrieve(
        self,
        *,
        user_id: str,
        project_id: Optional[str] = None,
        dataset_fingerprint: Optional[str] = None,
        k: Optional[int] = None,
    ) -> list[RetrievedMemory]:
        """Return applicable preferences with ``score=1.0``.

        Behaviour:

        * ``project_id is None AND dataset_fingerprint is None`` → return
          ONLY global preferences (both columns IS NULL on the row).
        * ``project_id`` set, ``dataset_fingerprint`` None → return rows
          that match the project (or are global), excluding
          dataset-scoped rows.
        * ``project_id`` None, ``dataset_fingerprint`` set → return rows
          that match the dataset (or are global), excluding
          project-scoped rows.
        * Both set → return rows that match either axis (or both), with
          the most-specific eligible row per key surfaced.

        ``k`` is accepted for signature compatibility with the other
        retrievers but is intentionally ignored when ``None`` —
        preferences are always-on user state. If a caller does pass a
        positive integer ``k``, the result is truncated *after* the full
        list is built so deterministic ordering is preserved.

        The ``reason`` field labels the effective scope of the surfaced
        row (``global`` / ``project`` / ``dataset`` / ``project+dataset``)
        so the LLM can see which level was matched.
        """
        if not isinstance(user_id, str) or not user_id:
            return []

        rows = self._store.list_active(
            user_id,
            project_id=project_id,
            dataset_fingerprint=dataset_fingerprint,
        )
        results: list[RetrievedMemory] = []
        for row in rows:
            key = row.get("preference_key") or "?"
            scope = _scope_label(
                row.get("project_id"), row.get("dataset_fingerprint")
            )
            results.append(
                RetrievedMemory(
                    memory=row,
                    score=1.0,
                    reason=f"active user preference: '{key}' [scope: {scope}]",
                    category="preference",
                )
            )

        if isinstance(k, int) and k > 0:
            return results[:k]
        return results


__all__ = ["PreferenceRetriever"]
