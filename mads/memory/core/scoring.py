"""scoring — v1.1 architectural scaffold.

Design intent: confidence-aware ranking for retrieval. Today each store's
list/search/find_similar returns rows ordered by a single signal (recency
for list, vector distance for find_similar, lexical match for search).
The scoring module will combine multiple signals — recency, confidence
score on the row, embedding distance, lexical hit count, user-feedback
boost on episodes — into a single rank used by the
:class:`mads.memory.retrievers.hybrid_retriever.HybridRetriever`.

v1.1 trigger: when single-signal retrieval visibly mis-ranks in the
with-memory eval (e.g. an outdated but recent metric definition outranking
the user's authoritative one).

Status: STUB — interface only, no behavior. See docs/architecture.md.
"""
from __future__ import annotations

from typing import Any


class Scorer:
    """Combines multiple retrieval signals into a single confidence-aware rank."""

    def score(self, candidates: list[dict[str, Any]], *, query: str) -> list[tuple[dict[str, Any], float]]:
        """Return ``(row, score)`` pairs sorted descending by score.

        Inputs: a list of candidate rows (already pre-filtered by tenancy
        and category) and the query text. Output: the same rows with a
        scalar score attached, sorted highest-first.
        """
        raise NotImplementedError(
            "scoring is a v1.1 scaffold; not yet implemented."
        )


__all__ = ["Scorer"]
