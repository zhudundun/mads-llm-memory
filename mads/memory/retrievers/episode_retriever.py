"""episode_retriever — vector KNN with optional post-filter metadata.

Calls :meth:`AnalysisEpisodeStore.find_similar` (which returns sqlite-vec
distance tuples), over-fetches ``k*3`` candidates to leave room for the
post-filter, then applies the metadata filters and trims to ``k``.

Score normalization: ``score = 1.0 - clamp(distance, 0, 2) / 2.0``.
sqlite-vec returns L2 distance over unit-norm embeddings, so distances
are bounded in ``[0, 2]``; clamping is defensive.
"""

from __future__ import annotations

from typing import Optional

from mads.memory.analysis_episode import AnalysisEpisodeStore
from mads.memory.retrievers.types import RetrievedMemory


class EpisodeRetriever:
    """Retrieves analysis episodes ranked by semantic similarity."""

    def __init__(self, store: AnalysisEpisodeStore) -> None:
        self._store = store

    def retrieve(
        self,
        query: str,
        *,
        project_id: str,
        k: int = 5,
        analysis_type: Optional[str] = None,
        datasets_used_contains: Optional[str] = None,
        created_after: Optional[str] = None,  # ISO-8601 string
    ) -> list[RetrievedMemory]:
        """Vector KNN with post-filter metadata.

        * Over-fetches ``k * 3`` candidates from the store to give the
          post-filter room to discard mismatches.
        * Filters:
            - ``analysis_type``: exact match.
            - ``datasets_used_contains``: case-insensitive substring on
              the raw ``datasets_used`` string.
            - ``created_after``: keep rows where ``created_at >=`` the
              given ISO-8601 string (lexicographic comparison is correct
              for ISO-8601).
        * Empty query returns ``[]``.
        """
        if not isinstance(query, str) or not query.strip():
            return []
        if not isinstance(k, int) or k <= 0:
            return []

        # Over-fetch to leave room for the post-filter; never less than k.
        fetch_k = max(k * 3, k)

        hits = self._store.find_similar(
            project_id=project_id, query=query, k=fetch_k
        )
        if not hits:
            return []

        # Build the filter-note suffix once — it's the same string for
        # every surviving candidate in this call.
        filter_notes: list[str] = []
        if analysis_type is not None:
            filter_notes.append(f"analysis_type={analysis_type}")
        if datasets_used_contains is not None:
            filter_notes.append(
                f"datasets_used~='{datasets_used_contains}'"
            )
        if created_after is not None:
            filter_notes.append(f"created_after={created_after}")
        suffix = (
            f"; filtered: {', '.join(filter_notes)}" if filter_notes else ""
        )

        dsu_needle = (
            datasets_used_contains.lower()
            if datasets_used_contains is not None
            else None
        )

        out: list[RetrievedMemory] = []
        for episode, distance in hits:
            # Apply filters.
            if analysis_type is not None and episode.analysis_type != analysis_type:
                continue
            if dsu_needle is not None:
                haystack = (episode.datasets_used or "").lower()
                if dsu_needle not in haystack:
                    continue
            if created_after is not None and episode.created_at < created_after:
                continue

            # Score: 1.0 - clamp(distance, 0, 2) / 2.0  -> [0, 1].
            d_clamped = max(0.0, min(2.0, float(distance)))
            score = 1.0 - d_clamped / 2.0
            reason = f"semantic match (distance={distance:.2f}){suffix}"

            out.append(
                RetrievedMemory(
                    memory=episode.model_dump(),
                    score=score,
                    reason=reason,
                    category="episode",
                )
            )

        # find_similar already orders by ascending distance (so descending
        # score). After filtering the order is preserved, but sort
        # explicitly for safety + to satisfy the documented contract.
        out.sort(key=lambda rm: -rm.score)
        return out[:k]


__all__ = ["EpisodeRetriever"]
