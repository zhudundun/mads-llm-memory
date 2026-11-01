"""vector_store — v1.1 architectural scaffold.

Design intent: abstract base for the sqlite-vec virtual-table usage that
currently lives inline in :mod:`mads.memory.analysis_episode`. Wraps the
``vec0`` virtual table creation, the per-row insert with an attached
embedding, and the ``MATCH``-based KNN query. Lets cat 6
(``workflow_recipe``, which will embed recipe descriptions) and a future
metric-definition embedding share the same code path without re-implementing
the sqlite-vec dance.

v1.1 trigger: when the *second* category needs vector search. Today only
``analysis_episode`` does, so factoring out the base would be speculative.

Status: STUB — interface only, no behavior. See docs/architecture.md.
"""
from __future__ import annotations

from typing import Any


class BaseVectorStore:
    """Abstract base for sqlite-vec-backed vector stores."""

    def index(self, row_id: str, embedding: list[float]) -> None:
        """Insert ``embedding`` into the vector virtual table keyed by ``row_id``."""
        raise NotImplementedError(
            "vector_store is a v1.1 scaffold; not yet implemented."
        )

    def search(self, query: list[float], *, k: int = 5) -> list[tuple[str, float]]:
        """Return the top-k ``(row_id, distance)`` pairs nearest to ``query``."""
        raise NotImplementedError(
            "vector_store is a v1.1 scaffold; not yet implemented."
        )


__all__ = ["BaseVectorStore"]
