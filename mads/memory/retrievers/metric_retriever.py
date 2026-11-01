"""metric_retriever — string + semantic matching on metric names.

Three string-match paths produce a "string_score" in ``[0.5, 1.0]``:

* Exact match on the normalized metric name → ``score=1.0``,
  ``reason="exact name match"``.
* Case-insensitive substring match on the normalized name →
  ``score = clamp(0.5 + 0.4 * (len(query) / len(name)), 0.5, 0.9)``,
  ``reason=f"substring match on name: '<normalized_name>'"``.
* Tokenized whole-token match — the normalized metric name appears as a
  contiguous run of whole tokens in the (token-split) query. Guards
  against false positives like ``"ion"`` matching ``"conversion"`` that
  pure substring would produce going in the opposite direction. Score
  ``= clamp(0.5 + 0.3 * (len(name) / len(query)), 0.5, 0.8)``,
  reason ``"tokenized match: '<name>' found in query as whole token(s)"``.

**B3 — semantic match (added 2026-06-06).** When the store is wired with
an :class:`~mads.memory.embedding.Embedder`, the retriever also embeds the
query and runs a ``vec0`` KNN scan over the per-project active metric
vectors. Each metric's score is the **max of string_score and
embedding_score** (a metric that exact-matches by name keeps the 1.0 — the
embedding leg can only lift the floor for synonym/plural cases like
``"qualified customers"`` matching the stored ``qualified_customer``).
Embedding scores below ``embedding_floor`` (default 0.6) are clamped to 0
so noise doesn't pollute the hybrid score. When both signals are zero the
metric is dropped.

The reason string is informative — callers see exactly why a row's score
is what it is: ``"semantic match (similarity=0.88)"`` or
``"substring match on name: 'net_revenue' + semantic (similarity=0.85)"``
when both signals fire.

Graceful degradation: if the store was built without an embedder, the
retriever still works in string-only mode (existing tests stay green).
"""

from __future__ import annotations

from typing import Optional

from mads.memory.embedding import Embedder
from mads.memory.metric_definition import (
    MetricDefinition,
    MetricDefinitionStore,
    _normalize,
)
from mads.memory.retrievers.types import RetrievedMemory


# Default similarity floor: scores below this are treated as noise.
# Calibrated empirically against the multi_01 case and the neg_01-style
# off-topic query, both with all-MiniLM-L6-v2:
#
#   "what's the weather today in tokyo" vs the qualified-customer row:
#       distance ≈ 1.45  →  sim ≈ 0.275
#   "What percent of qualified customers churn on this dataset?" vs the
#   qualified-customer row:
#       distance ≈ 1.04  →  sim ≈ 0.48
#   "whats our MRR?" vs an MRR row:
#       distance ≈ 0.96  →  sim ≈ 0.52
#
# A floor of 0.40 surfaces the synonym/plural cases the string fast-path
# misses while keeping the obvious noise out. Tunable per-instance via
# the constructor — if a downstream caller wants stricter matching they
# can dial it up.
_DEFAULT_EMBEDDING_FLOOR = 0.40


def _is_whole_token_subseq(name: str, query_tokens: list[str]) -> bool:
    """Return True iff ``name`` (split on ``_``) is a contiguous whole-token
    sub-sequence of ``query_tokens``.

    Examples:
      * ``_is_whole_token_subseq("conversion", ["show", "conversion", "by"])``
        → ``True`` (single token at index 1).
      * ``_is_whole_token_subseq("active_user", ["active", "user", "count"])``
        → ``True`` (two tokens at indices 0–1).
      * ``_is_whole_token_subseq("conversion", ["ion"])`` → ``False``
        (would require "conversion" as a complete token, but query has
        only "ion").
      * ``_is_whole_token_subseq("conversion", ["what", "is", "the",
        "weather", "today"])`` → ``False``.
    """
    if not name or not query_tokens:
        return False
    name_tokens = name.split("_")
    n, m = len(name_tokens), len(query_tokens)
    if n == 0 or n > m:
        return False
    # Slide a window of length n over the query tokens.
    for i in range(m - n + 1):
        if query_tokens[i : i + n] == name_tokens:
            return True
    return False


class MetricRetriever:
    """Retrieves metric definitions for a tenancy-scoped query.

    Combines fast-path string matching with optional vec0 semantic search
    via a max-of hybrid. The string fast path stays as-is (don't refactor
    `_is_whole_token_subseq` etc — they are the lower bound the embedder
    can only lift, not replace).
    """

    def __init__(
        self,
        store: MetricDefinitionStore,
        *,
        embedder: Optional[Embedder] = None,
        embedding_floor: float = _DEFAULT_EMBEDDING_FLOOR,
    ) -> None:
        self._store = store
        # Embedder may come from a kwarg OR from the store itself (the
        # MemoryManager wires it onto the store; tests construct the
        # retriever directly). Prefer the explicit kwarg so callers can
        # override for measurement.
        self._embedder: Optional[Embedder] = embedder or getattr(
            store, "_embedder", None
        )
        # Floor must be in [0, 1]; clamp defensively.
        self._embedding_floor = max(0.0, min(1.0, float(embedding_floor)))

    def retrieve(
        self,
        query: str,
        *,
        project_id: str,
        k: int = 5,
        available_columns: Optional[set[str]] = None,
    ) -> list[RetrievedMemory]:
        """Return up to ``k`` metrics ranked by max(string, semantic) score.

        Empty query, empty active set, or no matches all return ``[]``.
        Ties on score are broken by ``metric_name`` ascending so the
        ordering is deterministic.

        v1.4 ``available_columns`` (optional ``set[str]``): when provided,
        each result row is decorated with ``applicable: bool`` and
        ``missing_columns: list[str]`` on its ``memory`` dict. When
        ``available_columns is None`` (caller has no df bound and no
        column_memory snapshot), the applicability check is SKIPPED — the
        row is left without ``applicable`` / ``missing_columns`` keys. The
        formatter treats absence-of-key as "no annotation to surface,"
        which is identical in effect to ``applicable=True``.
        """
        if not isinstance(query, str) or not query.strip():
            return []
        if not isinstance(k, int) or k <= 0:
            return []

        # Normalize the query the same way the store normalizes metric names
        # so e.g. "Net Revenue" matches the stored normalized "net_revenue".
        try:
            normalized_query = _normalize(query)
        except ValueError:
            return []

        actives = self._store.list_active(project_id=project_id)
        if not actives:
            return []

        # Pre-tokenize the query once for the tokenized match path.
        query_tokens = normalized_query.split("_") if normalized_query else []

        # ---------- pass 1: collect string scores per metric.
        # Keyed by normalized_name (== identity for active rows in a project).
        string_scores: dict[str, tuple[float, str, dict]] = {}
        for metric in actives:
            name = metric.normalized_name
            mem_dict = metric.model_dump()
            if name == normalized_query:
                string_scores[name] = (1.0, "exact name match", mem_dict)
                continue
            if normalized_query in name:
                if len(name) == 0:
                    continue
                raw = 0.5 + 0.4 * (len(normalized_query) / len(name))
                s = max(0.5, min(0.9, raw))
                string_scores[name] = (
                    s,
                    f"substring match on name: '{name}'",
                    mem_dict,
                )
                continue
            if _is_whole_token_subseq(name, query_tokens):
                if len(normalized_query) == 0:
                    continue
                raw = 0.5 + 0.3 * (len(name) / len(normalized_query))
                s = max(0.5, min(0.8, raw))
                string_scores[name] = (
                    s,
                    f"tokenized match: '{name}' found in query as whole token(s)",
                    mem_dict,
                )

        # ---------- pass 2: collect semantic scores per metric.
        # Use the ORIGINAL query (not normalized) for embedding so the
        # embedder sees natural language. find_similar returns
        # ascending-distance tuples; convert to similarity in [0, 1].
        # NOTE: sqlite-vec's MATCH wants a k cap; over-fetch a bit so the
        # max-of merge has room.
        semantic_scores: dict[str, tuple[float, dict]] = {}
        if self._embedder is not None:
            fetch_k = max(k * 3, k)
            hits = self._store.find_similar(
                project_id=project_id, query=query, k=fetch_k
            )
            for metric, distance in hits:
                # Convert L2 over unit-norm vectors to similarity in [0,1].
                d_clamped = max(0.0, min(2.0, float(distance)))
                sim = 1.0 - d_clamped / 2.0
                if sim < self._embedding_floor:
                    continue
                semantic_scores[metric.normalized_name] = (
                    sim,
                    metric.model_dump(),
                )

        # ---------- pass 3: hybrid max-of merge.
        # Index actives by normalized_name for the applicability lookup
        # below — we need the typed MetricDefinition for is_applicable.
        actives_by_name: dict[str, "MetricDefinition"] = {
            m.normalized_name: m for m in actives
        }

        merged: list[RetrievedMemory] = []
        all_names = set(string_scores) | set(semantic_scores)
        for name in all_names:
            s_entry = string_scores.get(name)
            v_entry = semantic_scores.get(name)
            s_score = s_entry[0] if s_entry else 0.0
            v_score = v_entry[0] if v_entry else 0.0
            final = max(s_score, v_score)
            if final <= 0.0:
                continue

            # Compose a reason string that surfaces both signals.
            if s_entry and v_entry:
                # Show both — keep the string reason core and tack on
                # the semantic similarity for transparency.
                base_reason = s_entry[1]
                reason = (
                    f"{base_reason} + semantic (similarity={v_entry[0]:.2f})"
                )
            elif s_entry:
                reason = s_entry[1]
            else:
                reason = f"semantic match (similarity={v_entry[0]:.2f})"

            mem_dict = (s_entry[2] if s_entry else v_entry[1])  # type: ignore[index]

            # v1.4: applicability decoration. Skip the check when the
            # caller didn't pass column info (we can't decide, so the
            # formatter falls back to its existing "applicable" default).
            if available_columns is not None:
                metric_obj = actives_by_name.get(name)
                if metric_obj is not None:
                    applicable, missing = self._store.is_applicable(
                        metric_obj, available_columns
                    )
                    mem_dict["applicable"] = applicable
                    mem_dict["missing_columns"] = missing
                    # Stash a sorted snapshot of the bound df's columns so
                    # the formatter can show "bound df has [...]" without
                    # needing to reach back into the manager. Sorted for
                    # determinism — the LLM sees a stable ordering.
                    mem_dict["_available_columns_for_display"] = sorted(
                        available_columns
                    )

            merged.append(
                RetrievedMemory(
                    memory=mem_dict,
                    score=final,
                    reason=reason,
                    category="metric",
                )
            )

        # Sort by score desc, then by metric_name asc for a deterministic tie-break.
        merged.sort(key=lambda rm: (-rm.score, rm.memory["metric_name"]))
        return merged[:k]


__all__ = ["MetricRetriever"]
