"""dataset_retriever — fingerprint-based lookup.

Backed by :meth:`DatasetSchemaStore.lookup_dataset`, which already
implements the v1.1 ``exact-fingerprint → semantic-fingerprint``
fallback. We thinly wrap its output into the common
:class:`RetrievedMemory` shape so the hybrid retriever can fuse it
alongside the metric and episode scores.

Result count is 0 or 1 — ``lookup_dataset`` returns at most one match.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from mads.memory.dataset_schema import DatasetSchemaStore
from mads.memory.retrievers.types import RetrievedMemory


class DatasetRetriever:
    """Retrieves dataset memory for a fingerprintable artifact."""

    def __init__(self, store: DatasetSchemaStore) -> None:
        self._store = store

    def retrieve(
        self,
        *,
        user_id: str,
        table_name: str,
        df: Optional[pd.DataFrame] = None,
        file_bytes: Optional[bytes] = None,
        k: int = 5,
    ) -> list[RetrievedMemory]:
        """Return at most one :class:`RetrievedMemory` for the dataset.

        ``k`` is accepted for signature parity with the other retrievers
        but practically caps at 1 in v1.1 — ``lookup_dataset`` returns
        a single best match. If neither ``df`` nor ``file_bytes`` is
        supplied, returns ``[]``.
        """
        if df is None and file_bytes is None:
            return []
        if not isinstance(k, int) or k <= 0:
            return []
        if not isinstance(table_name, str) or not table_name:
            return []
        # The store API requires a df to compute the semantic_fingerprint;
        # we can't run any lookup without one. file_bytes alone is not
        # enough to drive lookup_dataset because the semantic path is
        # df-derived.
        if df is None:
            return []

        match = self._store.lookup_dataset(
            user_id=user_id,
            df=df,
            table_name=table_name,
            file_bytes=file_bytes,
        )
        if match is None:
            return []

        matched_by = match.get("matched_by")
        row = match.get("row") or {}
        if matched_by == "exact":
            score = 1.0
            reason = "exact_fingerprint match"
        elif matched_by == "semantic":
            score = 0.7
            reason = "semantic_fingerprint match (logical dataset)"
        else:
            # Unknown classifier — be defensive: skip rather than guess.
            return []

        return [
            RetrievedMemory(
                memory=row,
                score=score,
                reason=reason,
                category="dataset",
            )
        ]


__all__ = ["DatasetRetriever"]
