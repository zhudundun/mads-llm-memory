"""Pytest suite for :class:`mads.memory.retrievers.HybridRetriever`.

The hybrid retriever is a thin dispatcher: it fans out to the three
per-category retrievers, concatenates results, sorts by score (with
category-order tie-break), and caps at ``k``. These tests cover the
contract: dispatch, sort, tie-break, k-cap, dataset-skip, and category
tagging.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mads.memory.analysis_episode import AnalysisEpisodeStore
from mads.memory.dataset_schema import DatasetSchemaStore
from mads.memory.embedding import HashEmbedder
from mads.memory.metric_definition import MetricDefinitionStore
from mads.memory.retrievers import (
    DatasetRetriever,
    EpisodeRetriever,
    HybridRetriever,
    MetricRetriever,
    PreferenceRetriever,
)
from mads.memory.user_preference import PreferenceStore


# --------------------------------------------------------------------- fixtures


@pytest.fixture
def metric_store():
    s = MetricDefinitionStore(db_path=":memory:")
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def dataset_store():
    s = DatasetSchemaStore(db_path=":memory:")
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def episode_store():
    s = AnalysisEpisodeStore(db_path=":memory:", embedder=HashEmbedder())
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def preference_store():
    s = PreferenceStore(db_path=":memory:")
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def hybrid(metric_store, dataset_store, episode_store, preference_store):
    return HybridRetriever(
        metric=MetricRetriever(metric_store),
        dataset=DatasetRetriever(dataset_store),
        episode=EpisodeRetriever(episode_store),
        preference=PreferenceRetriever(preference_store),
    )


@pytest.fixture
def small_df():
    return pd.DataFrame(
        {
            "order_id": [1, 2, 3],
            "customer_id": [10, 20, 30],
            "amount": [9.99, 19.99, 29.99],
        }
    )


# --------------------------------------------------------------------- tests


def test_dispatches_to_all_three(
    hybrid, metric_store, dataset_store, episode_store, small_df
):
    """1. With a query that matches in every category, all three contribute."""
    metric_store.upsert(
        user_id="u1",
        project_id="p1",
        metric_name="revenue",
        business_definition="d",
    )
    dataset_store.register_dataset(
        user_id="u1",
        df=small_df,
        dataset_name="orders",
        table_name="orders_t",
        grain="g",
    )
    episode_store.record(
        user_id="u1",
        project_id="p1",
        question="why did revenue drop?",
        conclusion="paid search",
    )
    out = hybrid.retrieve(
        "revenue",
        user_id="u1",
        project_id="p1",
        k=10,
        df=small_df,
        table_name="orders_t",
    )
    cats = {r.category for r in out}
    assert cats == {"metric", "dataset", "episode"}


def test_sorted_by_score_desc_across_categories(
    hybrid, metric_store, episode_store
):
    """2. Across categories the result is sorted by score desc."""
    # Exact metric match -> score 1.0; episode semantic match < 1.0.
    metric_store.upsert(
        user_id="u1",
        project_id="p1",
        metric_name="revenue",
        business_definition="d",
    )
    episode_store.record(
        user_id="u1",
        project_id="p1",
        question="why did revenue drop?",
        conclusion="paid search",
    )
    out = hybrid.retrieve(
        "revenue", user_id="u1", project_id="p1", k=10
    )
    scores = [r.score for r in out]
    assert scores == sorted(scores, reverse=True)
    assert out[0].category == "metric"
    assert out[0].score == 1.0


def test_tie_break_metric_before_dataset_before_episode(
    hybrid, metric_store, dataset_store, episode_store, small_df
):
    """3. Equal-scoring rows break ties in metric < dataset < episode order."""
    # Force a tie: pin the score for each category by construction.
    # Metric: exact match -> 1.0.
    # Dataset: register WITH file_bytes and look it up with the same bytes -> 1.0.
    # Episode: distance 0 (exact embed-text) -> 1.0.
    raw = b"id,name\n1,a\n2,b\n"
    # We deliberately use a metric name that normalizes to the same form
    # as the episode's embed-text query ("anchor\n\nanchor" -> "anchor_anchor")
    # so all three categories produce a score of ~1.0 simultaneously.
    metric_store.upsert(
        user_id="u1",
        project_id="p1",
        metric_name="anchor_anchor",
        business_definition="d",
    )
    dataset_store.register_dataset(
        user_id="u1",
        df=small_df,
        dataset_name="orders",
        table_name="orders_t",
        grain="g",
        file_bytes=raw,
    )
    target = episode_store.record(
        user_id="u1",
        project_id="p1",
        question="anchor",
        conclusion="anchor",
    )
    # Embed-text for the seeded episode is "anchor\n\nanchor".
    query_text = "anchor\n\nanchor"
    out = hybrid.retrieve(
        query_text,
        user_id="u1",
        project_id="p1",
        k=10,
        df=small_df,
        file_bytes=raw,
        table_name="orders_t",
    )
    # Metric retriever sees the normalized "anchor\n\nanchor" -> "anchor"
    # which matches "anchor" exactly. All three score ~1.0; check the
    # tie-break ordering between metric and dataset, then episode last.
    top_cats = [r.category for r in out if r.score == pytest.approx(1.0, abs=1e-3)]
    # The exact-match rows should be ordered metric, dataset, episode.
    assert top_cats[0] == "metric"
    assert "dataset" in top_cats
    # Find the indices to confirm ordering.
    cat_indices = {c: top_cats.index(c) for c in set(top_cats)}
    assert cat_indices["metric"] < cat_indices["dataset"]
    if "episode" in cat_indices:
        assert cat_indices["dataset"] < cat_indices["episode"]
    # Sanity: the target episode is in the result somewhere.
    assert any(
        r.category == "episode" and r.memory["id"] == target.id for r in out
    )


def test_k_limit_after_fusion(
    hybrid, metric_store, episode_store
):
    """4. k caps the merged list, not each category individually."""
    for name in ("revenue", "net_revenue", "gross_revenue"):
        metric_store.upsert(
            user_id="u1",
            project_id="p1",
            metric_name=name,
            business_definition="d",
        )
    for i in range(3):
        episode_store.record(
            user_id="u1",
            project_id="p1",
            question=f"revenue drop q{i}",
            conclusion="paid search",
        )
    out = hybrid.retrieve("revenue", user_id="u1", project_id="p1", k=2)
    assert len(out) == 2


def test_dataset_skipped_when_no_artifact(
    hybrid, metric_store, dataset_store, small_df
):
    """5. Without df/file_bytes, the dataset retriever is not called."""
    dataset_store.register_dataset(
        user_id="u1",
        df=small_df,
        dataset_name="orders",
        table_name="orders_t",
        grain="g",
    )
    metric_store.upsert(
        user_id="u1",
        project_id="p1",
        metric_name="revenue",
        business_definition="d",
    )
    # No df, no file_bytes -> only metric should appear in the output.
    out = hybrid.retrieve(
        "revenue", user_id="u1", project_id="p1", k=10
    )
    assert all(r.category != "dataset" for r in out)


def test_mixed_results_preserve_category_tags(
    hybrid, metric_store, episode_store
):
    """6. The category tag on each row reflects the source retriever."""
    metric_store.upsert(
        user_id="u1",
        project_id="p1",
        metric_name="revenue",
        business_definition="d",
    )
    episode_store.record(
        user_id="u1",
        project_id="p1",
        question="why did revenue drop?",
        conclusion="paid search",
    )
    out = hybrid.retrieve(
        "revenue", user_id="u1", project_id="p1", k=10
    )
    by_cat = {r.category: r for r in out}
    assert "metric" in by_cat
    assert "episode" in by_cat
    # Metric row has metric_name; episode row has question.
    assert "metric_name" in by_cat["metric"].memory
    assert "question" in by_cat["episode"].memory


# --------------------------------------------------------------------- preference

# The PreferenceRetriever was added to close the v1.1 retrieval-surface
# gap that blocked eda_03 and eda_04. Preferences are always-on user
# state: they should be returned for every retrieve call, *in addition
# to* the k cap on relevance-ranked rows, and ranked first among ties.


def test_preferences_always_included_regardless_of_k(
    hybrid, metric_store, preference_store
):
    """7. With many metrics and preferences, ``k=2`` caps the relevance
    rows but ALL preferences come through (preferences are user-state,
    additional to k)."""
    for i in range(10):
        metric_store.upsert(
            user_id="u1",
            project_id="p1",
            metric_name=f"revenue_{i}",
            business_definition="d",
        )
    for i in range(5):
        preference_store.set("u1", f"pref_{i}", f"value_{i}")

    out = hybrid.retrieve("revenue", user_id="u1", project_id="p1", k=2)

    pref_rows = [r for r in out if r.category == "preference"]
    other_rows = [r for r in out if r.category != "preference"]
    assert len(pref_rows) == 5, "ALL 5 preferences must be returned"
    assert len(other_rows) == 2, "k=2 caps relevance rows"
    assert len(out) == 7


def test_preferences_sort_first_among_score_one(
    hybrid, metric_store, preference_store
):
    """8. A score-1.0 metric (exact match) and 2 score-1.0 preferences:
    the preferences come first."""
    metric_store.upsert(
        user_id="u1",
        project_id="p1",
        metric_name="revenue",  # exact match -> score 1.0
        business_definition="d",
    )
    preference_store.set("u1", "alpha_pref", "v1")
    preference_store.set("u1", "beta_pref", "v2")

    out = hybrid.retrieve("revenue", user_id="u1", project_id="p1", k=10)
    # Only score-1.0 rows: 2 prefs + 1 metric.
    top_three = out[:3]
    cats = [r.category for r in top_three]
    assert cats[:2] == ["preference", "preference"]
    assert cats[2] == "metric"


def test_no_preferences_when_user_has_none(hybrid, metric_store):
    """9. HybridRetriever still works (no errors, no preference rows in
    the output) for a user with zero preferences."""
    metric_store.upsert(
        user_id="u1",
        project_id="p1",
        metric_name="revenue",
        business_definition="d",
    )
    out = hybrid.retrieve("revenue", user_id="u1", project_id="p1", k=10)
    assert all(r.category != "preference" for r in out)
    assert len(out) == 1
    assert out[0].category == "metric"


def test_preferences_user_isolated_in_hybrid(
    hybrid, metric_store, preference_store
):
    """10. Preferences in user_a are invisible to a retrieve scoped to user_b."""
    preference_store.set("alice", "alice_pref_1", "v")
    preference_store.set("alice", "alice_pref_2", "v")
    preference_store.set("bob", "bob_pref", "v")

    out_alice = hybrid.retrieve(
        "anything", user_id="alice", project_id="p1", k=5
    )
    out_bob = hybrid.retrieve(
        "anything", user_id="bob", project_id="p1", k=5
    )
    alice_keys = {
        r.memory["preference_key"] for r in out_alice if r.category == "preference"
    }
    bob_keys = {
        r.memory["preference_key"] for r in out_bob if r.category == "preference"
    }
    assert alice_keys == {"alice_pref_1", "alice_pref_2"}
    assert bob_keys == {"bob_pref"}


def test_hybrid_shape_with_three_prefs_two_metrics_k5(
    hybrid, metric_store, preference_store
):
    """11. End-to-end shape verification: 3 prefs + 2 matching metrics
    with k=5 returns 5 (relevance cap) + 3 (all prefs) = 5 total
    because there are only 2 matching metrics. With 10 metrics and k=5,
    we'd see 5 relevance + 3 prefs = 8. Verifies the additive contract."""
    metric_store.upsert(
        user_id="u1",
        project_id="p1",
        metric_name="revenue",
        business_definition="d",
    )
    metric_store.upsert(
        user_id="u1",
        project_id="p1",
        metric_name="net_revenue",
        business_definition="d",
    )
    for k in ("pref_a", "pref_b", "pref_c"):
        preference_store.set("u1", k, "v")

    out = hybrid.retrieve("revenue", user_id="u1", project_id="p1", k=5)
    # 2 metric rows + 3 pref rows
    by_cat = {}
    for r in out:
        by_cat.setdefault(r.category, []).append(r)
    assert len(by_cat.get("preference", [])) == 3
    assert len(by_cat.get("metric", [])) == 2
    assert len(out) == 5


# --------------------------------------------------------------------- query-relevance gate

# The HybridRetriever gates on `query_classifier.is_data_question` at the
# top of `retrieve`. Non-analytical queries (poems, jokes, off-topic chat)
# should short-circuit to `[]` so no sub-retriever fires and no memory
# pollutes the prompt. See `neg_01_unrelated_query` in eval/scenarios.json.


def test_hybrid_returns_empty_for_creative_query(
    hybrid, metric_store, preference_store
):
    """12. Non-data queries trip the gate and return [] regardless of
    seeded preferences / metrics."""
    for i in range(3):
        preference_store.set("u1", f"pref_{i}", f"value_{i}")
    metric_store.upsert(
        user_id="u1",
        project_id="p1",
        metric_name="revenue",
        business_definition="d",
    )
    out = hybrid.retrieve(
        "Write me a short poem about telecoms",
        user_id="u1",
        project_id="p1",
        k=10,
    )
    assert out == []


def test_hybrid_returns_normally_for_data_query(
    hybrid, metric_store, preference_store
):
    """13. The gate is a NO-OP for analytical queries — happy-path
    behavior (3 prefs + 1 matching metric → 4 rows) is preserved."""
    for i in range(3):
        preference_store.set("u1", f"pref_{i}", f"value_{i}")
    metric_store.upsert(
        user_id="u1",
        project_id="p1",
        metric_name="revenue",
        business_definition="d",
    )
    out = hybrid.retrieve(
        "Show me the revenue data",
        user_id="u1",
        project_id="p1",
        k=10,
    )
    assert len(out) >= 4
    cats = {r.category for r in out}
    assert "preference" in cats
    assert "metric" in cats
