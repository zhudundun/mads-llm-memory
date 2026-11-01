"""Shared types for the specialized retrievers (v1.1).

Each retriever returns ``list[RetrievedMemory]``: a uniform shape carrying
the source row, a normalized relevance ``score`` in ``[0.0, 1.0]``, a
one-line human-readable ``reason``, and a ``category`` tag identifying
which retriever produced the row. The hybrid retriever concatenates lists
from every per-category retriever and sorts by ``score`` descending.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol


Category = Literal["metric", "dataset", "episode", "preference"]


@dataclass
class RetrievedMemory:
    """One result row with provenance.

    Attributes
    ----------
    memory:
        The source row as a plain ``dict`` (so callers don't need to
        know the per-category pydantic model).
    score:
        Normalized relevance in ``[0.0, 1.0]`` — higher = more relevant.
    reason:
        One-line human-readable explanation of *why* this row was
        retrieved (e.g. ``"exact name match"`` or
        ``"semantic match (distance=0.12); filtered: analysis_type=root_cause"``).
    category:
        Which retriever produced this row. The hybrid retriever uses
        this both as a tie-breaker and as a passthrough tag for the
        caller.
    """

    memory: dict
    score: float
    reason: str
    category: Category


class Retriever(Protocol):
    """Marker protocol for the per-category retrievers.

    The per-category retrievers all have specialized ``retrieve``
    signatures (metric takes a query string + project; dataset takes a
    fingerprintable artifact; episode takes a query + filters;
    preference takes only ``user_id``). The one invariant shared across
    them all: ``retrieve`` returns a ``list[RetrievedMemory]`` sorted by
    ``score`` descending (preferences are uniformly ``score=1.0``).
    """


__all__ = ["RetrievedMemory", "Category", "Retriever"]
