"""hybrid_retriever — thin dispatcher across the per-category retrievers.

v1.1: no RRF, no score re-normalization, no LLM. Each per-category
retriever already returns scores on the same ``[0.0, 1.0]`` scale, so
we simply concatenate and sort. Ties are broken by category order
(``preference < metric < dataset < episode``) — preferences come first
among score-1.0 rows because they are always-on user state (i.e. they
apply to every turn regardless of query), so the LLM should see them at
the top of the injected memory block. The remaining order
(``metric < dataset < episode``) reflects how deterministic each signal
is (an exact-name match is a stronger signal than a fingerprint match,
which is stronger than a semantic match).

Preferences are returned *in addition to* the ``k`` cap on relevance-
ranked rows — they are user-state, not query-context, so capping them
by ``k`` would arbitrarily drop user-state that always applies. The
implementation slices metric+dataset+episode at ``k`` and then appends
the full preference list before the final sort.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from mads.memory.retrievers.dataset_retriever import DatasetRetriever
from mads.memory.retrievers.episode_retriever import EpisodeRetriever
from mads.memory.retrievers.metric_retriever import MetricRetriever
from mads.memory.retrievers.preference_retriever import PreferenceRetriever
from mads.memory.retrievers.query_classifier import is_data_question
from mads.memory.retrievers.types import RetrievedMemory


# Preferences rank first among ties because they are always-applicable
# user state; the remaining order reflects signal determinism.
_CATEGORY_ORDER = {"preference": 0, "metric": 1, "dataset": 2, "episode": 3}


class HybridRetriever:
    """Fans out to all four per-category retrievers and merges results."""

    def __init__(
        self,
        metric: MetricRetriever,
        dataset: DatasetRetriever,
        episode: EpisodeRetriever,
        preference: PreferenceRetriever,
    ) -> None:
        self._metric = metric
        self._dataset = dataset
        self._episode = episode
        self._preference = preference

    def retrieve(
        self,
        query: str,
        *,
        user_id: str,
        project_id: str,
        k: int = 5,
        # episode-side passthrough filters:
        analysis_type: Optional[str] = None,
        datasets_used_contains: Optional[str] = None,
        created_after: Optional[str] = None,
        # dataset-side passthrough:
        df: Optional[pd.DataFrame] = None,
        file_bytes: Optional[bytes] = None,
        table_name: Optional[str] = None,
        # v1.4 metric-side passthrough — schema coupling against cat 2.
        # Passed through to MetricRetriever so each result is decorated
        # with applicable/missing_columns; None skips the check.
        available_columns: Optional[set[str]] = None,
    ) -> list[RetrievedMemory]:
        """Dispatch + concatenate + sort.

        Returns ``[]`` when the query is not a data-analysis question (per
        :func:`mads.memory.retrievers.query_classifier.is_data_question`).
        This prevents preference / dataset / episode / metric memories from
        polluting prompts for off-topic queries (poems, creative writing,
        generic chat) — the gate fans out uniformly to all four
        sub-retrievers, so none of them fire when the gate trips. See
        ``neg_01_unrelated_query`` in the broad eval for the failure mode
        this fix closes.

        Otherwise:

        * Always calls the metric and episode retrievers with ``query``.
        * Calls the dataset retriever only if a fingerprintable artifact
          (``df`` or ``file_bytes``) AND a ``table_name`` are provided.
        * Always calls the preference retriever with ``user_id`` — its
          result list is independent of the query.
        * Concatenates metric+dataset+episode results, sorts by ``score``
          desc (ties broken by category order), caps at ``k``, then
          appends ALL preferences (additional to k — preferences are
          user-state, not relevance-ranked context).
        * Final sort by ``(-score, category_order)`` so the score-1.0
          preference rows surface above other score-1.0 rows.
        """
        # Query-relevance gate (closes neg_01 over-retrieval on creative
        # / off-topic prompts). Fires uniformly across all sub-retrievers.
        if not is_data_question(query):
            return []

        if not isinstance(k, int) or k <= 0:
            return []

        # Relevance-ranked categories — these are capped at k.
        ranked: list[RetrievedMemory] = []
        ranked.extend(
            self._metric.retrieve(
                query,
                project_id=project_id,
                k=k,
                available_columns=available_columns,
            )
        )

        # Dataset retriever only fires when we actually have an artifact
        # and a table_name to scope it to.
        if (df is not None or file_bytes is not None) and table_name is not None:
            ranked.extend(
                self._dataset.retrieve(
                    user_id=user_id,
                    table_name=table_name,
                    df=df,
                    file_bytes=file_bytes,
                    k=1,
                )
            )

        ranked.extend(
            self._episode.retrieve(
                query,
                project_id=project_id,
                k=k,
                analysis_type=analysis_type,
                datasets_used_contains=datasets_used_contains,
                created_after=created_after,
            )
        )

        # Sort + cap the relevance-ranked list.
        ranked.sort(
            key=lambda rm: (-rm.score, _CATEGORY_ORDER.get(rm.category, 99))
        )
        ranked = ranked[:k]

        # Preferences are user-state — always returned, NOT counted against k.
        # Thread project_id AND dataset_fingerprint through so the
        # IDENTITY_OPTIONAL scope filter on BOTH axes picks up scoped
        # prefs in addition to global ones. When the session has no df
        # bound (``df is None`` and no datasets_used_contains), we use
        # the same fingerprint the episode/dataset retrievers see via
        # ``datasets_used_contains`` so all sub-retrievers agree on the
        # active dataset.
        if df is not None:
            from mads.dataset.fingerprint import semantic_fingerprint

            ds_fp_for_prefs: Optional[str] = semantic_fingerprint(df)
        else:
            ds_fp_for_prefs = datasets_used_contains
        preferences = self._preference.retrieve(
            user_id=user_id,
            project_id=project_id,
            dataset_fingerprint=ds_fp_for_prefs,
        )

        merged = ranked + preferences

        # Final sort: score desc, then category order. Preferences (all
        # score=1.0, category_order=0) end up above any other score=1.0
        # rows, with the rest of the merged list keeping its prior order.
        merged.sort(
            key=lambda rm: (-rm.score, _CATEGORY_ORDER.get(rm.category, 99))
        )
        return merged


__all__ = ["HybridRetriever"]
