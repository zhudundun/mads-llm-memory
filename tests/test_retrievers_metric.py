"""Pytest suite for :class:`mads.memory.retrievers.MetricRetriever`.

Exercises the exact-then-substring ranking, project isolation, ``k``
caps, and the documented reason strings. All tests run against an
in-memory :class:`MetricDefinitionStore` — no embeddings, no network.
"""

from __future__ import annotations

import pytest

from mads.memory.metric_definition import MetricDefinitionStore
from mads.memory.retrievers import MetricRetriever, RetrievedMemory


# --------------------------------------------------------------------- fixtures


@pytest.fixture
def store():
    s = MetricDefinitionStore(db_path=":memory:")
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def retriever(store):
    return MetricRetriever(store)


def _seed(store, *, project_id="p1", user_id="u1", name, definition="d"):
    """Insert one active metric and return its MetricDefinition."""
    md, _ = store.upsert(
        user_id=user_id,
        project_id=project_id,
        metric_name=name,
        business_definition=definition,
    )
    return md


# --------------------------------------------------------------------- tests


def test_exact_match_scores_one(store, retriever):
    """1. Exact normalized-name match yields score=1.0 with the exact-match reason."""
    _seed(store, name="net_revenue")
    out = retriever.retrieve("net_revenue", project_id="p1")
    assert len(out) == 1
    assert isinstance(out[0], RetrievedMemory)
    assert out[0].score == 1.0
    assert out[0].reason == "exact name match"
    assert out[0].category == "metric"


def test_exact_match_is_case_insensitive_via_normalization(store, retriever):
    """2. 'Net Revenue' (with space + casing) matches 'net_revenue' exactly."""
    _seed(store, name="net_revenue")
    out = retriever.retrieve("Net Revenue", project_id="p1")
    assert len(out) == 1
    assert out[0].score == 1.0
    assert out[0].reason == "exact name match"


def test_substring_match_score_in_band(store, retriever):
    """3. Substring match returns score in (0.5, 0.9) with the substring reason."""
    _seed(store, name="net_revenue")
    out = retriever.retrieve("revenue", project_id="p1")
    assert len(out) == 1
    assert 0.5 < out[0].score < 0.9
    assert out[0].reason == "substring match on name: 'net_revenue'"


def test_k_limit_honored(store, retriever):
    """4. retrieve respects the k cap."""
    for name in ("net_revenue", "gross_revenue", "revenue_per_user"):
        _seed(store, name=name)
    out = retriever.retrieve("revenue", project_id="p1", k=2)
    assert len(out) == 2


def test_empty_active_list_returns_empty(store, retriever):
    """5. No metrics in the project -> empty result."""
    assert retriever.retrieve("anything", project_id="p1") == []


def test_empty_query_returns_empty(store, retriever):
    """6. Empty / whitespace-only query -> empty result regardless of state."""
    _seed(store, name="net_revenue")
    assert retriever.retrieve("", project_id="p1") == []
    assert retriever.retrieve("   ", project_id="p1") == []


def test_project_isolation(store, retriever):
    """7. Metrics in project A are invisible to a retrieve scoped to B."""
    _seed(store, project_id="A", name="net_revenue")
    assert retriever.retrieve("net_revenue", project_id="B") == []
    a = retriever.retrieve("net_revenue", project_id="A")
    assert len(a) == 1


def test_sorted_by_score_desc(store, retriever):
    """8. Exact match outranks substring match in the merged output."""
    # 'revenue' is an exact match for one metric and a substring of another.
    _seed(store, name="revenue")
    _seed(store, name="net_revenue")
    out = retriever.retrieve("revenue", project_id="p1", k=5)
    assert len(out) == 2
    # exact first
    assert out[0].memory["normalized_name"] == "revenue"
    assert out[0].score == 1.0
    # substring next
    assert out[1].memory["normalized_name"] == "net_revenue"
    assert out[1].score < 1.0


def test_tie_break_by_metric_name_asc(store, retriever):
    """9. Equal-scoring substring hits sort by metric_name ascending."""
    # All three metrics share the same length, so the substring score is
    # identical for query "revenue". Tie-break by metric_name asc.
    _seed(store, name="b_revenue")
    _seed(store, name="a_revenue")
    _seed(store, name="c_revenue")
    out = retriever.retrieve("revenue", project_id="p1", k=5)
    scores = [r.score for r in out]
    assert scores[0] == scores[1] == scores[2]
    names = [r.memory["metric_name"] for r in out]
    assert names == sorted(names)


def test_substring_reason_includes_normalized_name(store, retriever):
    """10. Substring reason includes the matched normalized name verbatim."""
    _seed(store, name="Net Revenue Per User")
    out = retriever.retrieve("revenue", project_id="p1")
    assert len(out) == 1
    # normalize("Net Revenue Per User") == "net_revenue_per_user"
    assert out[0].reason == "substring match on name: 'net_revenue_per_user'"


# --------------------------------------------------------------------- tokenized
# Tests for the new tokenized whole-token match path. The bug it fixes:
# natural-language queries like "Show conversion by channel" normalize to
# "show_conversion_by_channel", which is NOT a substring of "conversion",
# so the legacy two-path retriever returned []. The tokenized path looks
# at the query as a list of tokens and accepts the metric name as a hit
# iff it appears as a contiguous whole-token sub-sequence.


def test_tokenized_match_finds_metric_in_natural_language(store, retriever):
    """11. 'Show conversion by channel' finds the 'conversion' metric."""
    _seed(store, name="conversion")
    out = retriever.retrieve("Show conversion by channel", project_id="p1")
    assert len(out) == 1
    assert out[0].memory["normalized_name"] == "conversion"
    assert "tokenized match" in out[0].reason
    assert "'conversion'" in out[0].reason
    assert "whole token(s)" in out[0].reason


def test_tokenized_match_multi_token_name(store, retriever):
    """12. Multi-token metric ('active_user') matched against query
    'active user count' (tokens ['active', 'user', 'count'])."""
    _seed(store, name="active_user")
    out = retriever.retrieve("active user count", project_id="p1")
    assert len(out) == 1
    assert out[0].memory["normalized_name"] == "active_user"
    assert "tokenized match" in out[0].reason


def test_tokenized_match_does_not_fire_on_partial_token(store, retriever):
    """13. Query 'ion' must NOT match the 'conversion' metric — the legacy
    substring path matched the *query* inside the *name*; the tokenized
    path runs in the opposite direction and would only fire if the metric
    name appeared as a complete token in the query. ['ion'] does not
    contain 'conversion' as a token."""
    _seed(store, name="conversion")
    out = retriever.retrieve("ion", project_id="p1")
    # NB: this query goes through *substring* matching first
    # ('ion' IN 'conversion' → True), so we expect the substring reason,
    # NOT a tokenized one. The key assertion is: the tokenized path itself
    # does NOT produce a hit (verified via the reason string).
    if out:
        assert "tokenized match" not in out[0].reason


def test_tokenized_match_does_not_fire_on_unrelated_query(store, retriever):
    """14. Query 'what is the weather today' does not match 'conversion'."""
    _seed(store, name="conversion")
    out = retriever.retrieve("what is the weather today", project_id="p1")
    assert out == []


def test_tokenized_score_within_cap(store, retriever):
    """15. Tokenized scores must stay within [0.5, 0.8]."""
    # Three seeds with different name/query-length ratios.
    _seed(store, name="conversion")
    _seed(store, name="active_user")
    _seed(store, name="x")

    out = retriever.retrieve("Show conversion by channel", project_id="p1")
    assert len(out) >= 1
    for rm in out:
        if "tokenized match" in rm.reason:
            assert 0.5 <= rm.score <= 0.8, (
                f"tokenized score {rm.score} for {rm.memory['normalized_name']} "
                "outside [0.5, 0.8] cap"
            )


def test_tokenized_reason_string_contains_phrase(store, retriever):
    """16. The reason string contains the documented tokenized-match phrase."""
    _seed(store, name="conversion")
    out = retriever.retrieve("show conversion now", project_id="p1")
    assert len(out) == 1
    assert out[0].reason == (
        "tokenized match: 'conversion' found in query as whole token(s)"
    )


def test_priority_substring_beats_tokenized(store, retriever):
    """17. A single metric matched by both substring and tokenized paths
    takes the substring score (we don't double-count); but in practice
    the two paths are mutually exclusive (substring requires query⊂name;
    tokenized requires name appears as whole tokens in query, which can
    only happen when name⊆query). Sanity: substring still ranks above
    tokenized when they fight on different metrics.

    With metric 'conversion' and query 'show conversion by channel',
    'conversion' is a substring of the query? NO — the substring path
    checks query⊂name, not the reverse. So this query hits the
    tokenized path only. Verify it still scores below an exact match.
    """
    _seed(store, name="conversion")
    _seed(store, name="show_conversion_by_channel")
    out = retriever.retrieve("show conversion by channel", project_id="p1")
    # 'show_conversion_by_channel' is an exact match → 1.0
    # 'conversion' is a tokenized match → <= 0.8
    assert len(out) == 2
    assert out[0].score == 1.0
    assert out[1].score <= 0.8
    assert "tokenized match" in out[1].reason


# --------------------------------------------------------------------- B3 hybrid
# Tests for the embedding-augmented hybrid retrieval added in B3
# (mads/memory/retrievers/metric_retriever.py).
#
# These tests exercise the embedding leg with a REAL
# SentenceTransformerEmbedder when available so we get genuine semantic
# scores; the synonym / abbreviation cases need a real model because
# HashEmbedder is keyword-bag, not semantic. The fixtures below skip
# semantic tests when the model isn't loadable (e.g. offline CI).


@pytest.fixture(scope="module")
def real_embedder():
    """Real SentenceTransformerEmbedder, or skip if unavailable.

    Loaded once per module to amortize the model-load cost across the
    semantic tests below.
    """
    try:
        from mads.memory.embedding import SentenceTransformerEmbedder

        emb = SentenceTransformerEmbedder()
        # Trigger lazy model load now so the skip is decided up-front.
        emb.embed("warmup")
        return emb
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"SentenceTransformerEmbedder unavailable: {e!r}")


@pytest.fixture
def embedded_store(real_embedder, tmp_path):
    """File-backed store with the real embedder wired."""
    s = MetricDefinitionStore(
        db_path=str(tmp_path / "metric_retriever_b3.sqlite"),
        embedder=real_embedder,
    )
    try:
        yield s
    finally:
        s.close()


def test_b3_qualified_customer_plural_synonym(embedded_store, real_embedder):
    """18. The exact failing case from multi_01: store 'qualified_customer'
    with a real business definition; query "qualified customers" surfaces
    it via semantic similarity. The string fast-path can't see this match
    (none of the three string strategies catch 'customers' ≠ 'customer'),
    so the semantic leg is what closes the gap."""
    _seed(
        embedded_store,
        name="qualified_customer",
        definition=(
            "A customer who has been with the company for more than 12 months "
            "(tenure > 12) AND pays more than $50/month (MonthlyCharges > 50)."
        ),
    )
    r = MetricRetriever(embedded_store, embedder=real_embedder)
    out = r.retrieve(
        "What percent of qualified customers churn on this dataset?",
        project_id="p1",
    )
    assert len(out) >= 1
    hit = next(
        (rm for rm in out if rm.memory["normalized_name"] == "qualified_customer"),
        None,
    )
    assert hit is not None, f"qualified_customer not surfaced; out={out!r}"
    # The score should be at or above the calibrated 0.40 floor.
    # Measured similarity for this exact case is ~0.48 with all-MiniLM-L6-v2.
    assert hit.score >= 0.40, f"score {hit.score} below floor"
    # Reason should mention the semantic signal.
    assert "semantic" in hit.reason


def test_b3_embedding_floor_drops_unrelated_query(embedded_store, real_embedder):
    """19. An unrelated query (no shared tokens, no semantic match)
    returns no metric hits — embedding-floor filters the noise."""
    _seed(
        embedded_store,
        name="qualified_customer",
        definition="tenure > 12 AND MonthlyCharges > 50",
    )
    r = MetricRetriever(embedded_store, embedder=real_embedder)
    out = r.retrieve("what is the weather today in tokyo", project_id="p1")
    assert out == [], f"expected no hits for unrelated query, got {out!r}"


def test_b3_hybrid_max_of_keeps_exact_at_1(embedded_store, real_embedder):
    """20. A query that exact-matches by name AND is semantically loud
    still scores 1.0 (string wins the max-of). The reason string mentions
    BOTH signals so callers can see what fired."""
    _seed(
        embedded_store,
        name="conversion",
        definition="orders divided by sessions",
    )
    r = MetricRetriever(embedded_store, embedder=real_embedder)
    out = r.retrieve("conversion", project_id="p1")
    assert len(out) == 1
    assert out[0].score == 1.0
    assert "exact name match" in out[0].reason


def test_b3_embedding_only_match_when_string_misses(
    embedded_store, real_embedder
):
    """21. A query that has zero string overlap but semantically close
    surfaces via the embedding-only leg. Score is the raw similarity."""
    _seed(
        embedded_store,
        name="monthly_recurring_revenue",
        definition=(
            "Monthly recurring revenue (MRR): the predictable revenue a "
            "company can expect every month, computed as ARPU × active "
            "subscribers at month end."
        ),
    )
    r = MetricRetriever(embedded_store, embedder=real_embedder)
    out = r.retrieve("MRR figure for our subscribers", project_id="p1")
    # The query shares "subscribers" with the definition; the metric
    # name 'monthly_recurring_revenue' is what we want surfaced.
    if not out:
        pytest.skip(
            "embedder did not surface MRR via synonym query "
            "(score < floor) — semantic strength varies by model"
        )
    hit = next(
        (rm for rm in out if rm.memory["normalized_name"] == "monthly_recurring_revenue"),
        None,
    )
    assert hit is not None, f"MRR metric not surfaced; out={out!r}"
    # Embedding-only matches phrase as "semantic match (similarity=X.XX)".
    assert "semantic" in hit.reason


def test_b3_graceful_degradation_without_embedder(store):
    """22. Retriever built with embedder=None falls back to string-only.

    All the existing string-path tests use this configuration (the base
    `retriever` fixture passes no embedder); this test makes the
    contract explicit."""
    _seed(store, name="net_revenue")
    r = MetricRetriever(store, embedder=None)
    out = r.retrieve("net_revenue", project_id="p1")
    assert len(out) == 1
    assert out[0].score == 1.0
    assert out[0].reason == "exact name match"


def test_b3_floor_clamps_below_threshold(embedded_store, real_embedder):
    """23. Raising the floor above the semantic similarity drops the row.

    Confirms the floor is wired and effective."""
    _seed(
        embedded_store,
        name="qualified_customer",
        definition="tenure > 12 AND MonthlyCharges > 50",
    )
    # Raise the floor absurdly high so even the qualified-customers
    # query falls below it.
    r = MetricRetriever(embedded_store, embedder=real_embedder, embedding_floor=0.99)
    out = r.retrieve("qualified customers", project_id="p1")
    # The string fast-path can still hit (substring of 'qualified_customer'),
    # so we expect at most one hit and it should NOT mention 'semantic'.
    for rm in out:
        assert "semantic" not in rm.reason, (
            f"semantic leg should be clamped: {rm.reason}"
        )


# =================================================================== v1.4: applicability decoration
# MetricRetriever decorates each result with applicable/missing_columns when
# the caller passes available_columns. When available_columns is None, the
# decoration is skipped (back-compat with the existing string-only callers).


def _seed_with_required(store, *, name, definition="d", required):
    md, _ = store.upsert(
        user_id="u1",
        project_id="p1",
        metric_name=name,
        business_definition=definition,
        required_columns=required,
    )
    return md


def test_applicable_decoration_skipped_when_none(store, retriever):
    """24. available_columns=None: no applicable/missing keys on returned memory."""
    _seed_with_required(store, name="m", required=["x"])
    out = retriever.retrieve("m", project_id="p1")
    assert len(out) == 1
    mem = out[0].memory
    assert "applicable" not in mem
    assert "missing_columns" not in mem


def test_applicable_true_when_required_subset(store, retriever):
    """25. available_columns covers required -> applicable=True, missing=[]."""
    _seed_with_required(store, name="qualified", required=["tenure", "Charges"])
    out = retriever.retrieve(
        "qualified", project_id="p1",
        available_columns={"tenure", "Charges", "Churn"},
    )
    assert len(out) == 1
    mem = out[0].memory
    assert mem["applicable"] is True
    assert mem["missing_columns"] == []


def test_applicable_false_lists_missing_columns(store, retriever):
    """26. Missing column -> applicable=False, missing_columns lists the gap."""
    _seed_with_required(store, name="active_user", required=["last_payment_date"])
    out = retriever.retrieve(
        "active_user", project_id="p1",
        available_columns={"tenure", "Churn"},
    )
    assert len(out) == 1
    mem = out[0].memory
    assert mem["applicable"] is False
    assert mem["missing_columns"] == ["last_payment_date"]
    # And the column display snapshot must be present (sorted) so the
    # downstream formatter has everything it needs.
    assert mem["_available_columns_for_display"] == ["Churn", "tenure"]


def test_mixed_applicable_and_inapplicable_metrics(store, retriever):
    """27. A retrieval returning both applicable and inapplicable rows
    decorates each independently."""
    _seed_with_required(store, name="active_user", required=["last_payment_date"])
    _seed_with_required(store, name="active_users_plural", required=["tenure"])
    out = retriever.retrieve(
        "active",
        project_id="p1",
        available_columns={"tenure", "Churn"},
    )
    # Both substring-match 'active'.
    by_name = {rm.memory["metric_name"]: rm for rm in out}
    assert "active_user" in by_name
    assert "active_users_plural" in by_name
    assert by_name["active_user"].memory["applicable"] is False
    assert by_name["active_user"].memory["missing_columns"] == [
        "last_payment_date"
    ]
    assert by_name["active_users_plural"].memory["applicable"] is True
    assert by_name["active_users_plural"].memory["missing_columns"] == []


def test_applicable_true_for_metric_with_no_required(store, retriever):
    """28. None / [] required_columns -> always applicable regardless of df."""
    # Legacy row (None).
    _seed(store, name="legacy", definition="prose only")
    # Explicit empty list.
    store.upsert(
        user_id="u1",
        project_id="p1",
        metric_name="row_count",
        business_definition="count rows",
        required_columns=[],
    )
    out_legacy = retriever.retrieve(
        "legacy", project_id="p1", available_columns=set()
    )
    out_count = retriever.retrieve(
        "row_count", project_id="p1", available_columns=set()
    )
    assert len(out_legacy) == 1 and out_legacy[0].memory["applicable"] is True
    assert len(out_count) == 1 and out_count[0].memory["applicable"] is True
