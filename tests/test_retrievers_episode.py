"""Pytest suite for :class:`mads.memory.retrievers.EpisodeRetriever`.

All tests use the deterministic :class:`HashEmbedder` so we avoid the
sentence-transformers model load. The retriever wraps
:meth:`AnalysisEpisodeStore.find_similar` and applies post-filter
metadata, so these tests cover both the vector path and the filter
combinations.
"""

from __future__ import annotations

import time

import pytest

from mads.memory.analysis_episode import AnalysisEpisodeStore
from mads.memory.embedding import HashEmbedder
from mads.memory.retrievers import EpisodeRetriever, RetrievedMemory


# --------------------------------------------------------------------- fixtures


@pytest.fixture
def embedder():
    return HashEmbedder()


@pytest.fixture
def store(embedder):
    s = AnalysisEpisodeStore(db_path=":memory:", embedder=embedder)
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def retriever(store):
    return EpisodeRetriever(store)


def _seed(
    store,
    *,
    user_id="u1",
    project_id="p1",
    question="Why did revenue drop?",
    conclusion="Paid search shift.",
    analysis_type=None,
    datasets_used=None,
    assumptions=None,
):
    return store.record(
        user_id=user_id,
        project_id=project_id,
        question=question,
        conclusion=conclusion,
        analysis_type=analysis_type,
        datasets_used=datasets_used,
        assumptions=assumptions,
    )


# --------------------------------------------------------------------- tests


def test_returns_results_sorted_by_score_desc(store, retriever):
    """1. Vector hits come back sorted by score descending (distance ascending)."""
    _seed(store, question="revenue drop", conclusion="paid search shift")
    time.sleep(0.002)
    _seed(store, question="churn model", conclusion="add tenure feature")
    out = retriever.retrieve("revenue drop", project_id="p1", k=5)
    assert len(out) == 2
    assert all(isinstance(r, RetrievedMemory) for r in out)
    scores = [r.score for r in out]
    assert scores == sorted(scores, reverse=True)


def test_score_formula_matches_spec(store, retriever):
    """2. score = 1.0 - clamp(distance, 0, 2)/2.0, sanity-checked on an exact hit."""
    target = _seed(
        store,
        question="Why did revenue drop on 2026-05-15?",
        conclusion="Paid-search mix shift.",
    )
    _seed(store, question="How is churn?", conclusion="Adding tenure.")
    exact_text = "Why did revenue drop on 2026-05-15?\n\nPaid-search mix shift."
    out = retriever.retrieve(exact_text, project_id="p1", k=2)
    assert out[0].memory["id"] == target.id
    # Distance ~0 for an identical embed -> score ~1.0.
    assert out[0].score == pytest.approx(1.0, abs=1e-3)
    # Generic invariant for all returned rows.
    for r in out:
        assert 0.0 <= r.score <= 1.0


def test_analysis_type_filter_excludes_mismatched(store, retriever):
    """3. analysis_type filter drops episodes that don't match exactly."""
    _seed(
        store,
        question="rev drop",
        conclusion="paid search",
        analysis_type="root_cause",
    )
    _seed(
        store,
        question="rev drop",
        conclusion="paid search again",
        analysis_type="forecast",
    )
    out = retriever.retrieve(
        "rev drop", project_id="p1", k=5, analysis_type="root_cause"
    )
    assert len(out) == 1
    assert out[0].memory["analysis_type"] == "root_cause"


def test_datasets_used_filter_is_case_insensitive_substring(store, retriever):
    """4. datasets_used_contains is a case-insensitive substring filter."""
    _seed(
        store,
        question="q",
        conclusion="c",
        datasets_used="payments, traffic_attribution",
    )
    _seed(
        store,
        question="q",
        conclusion="c",
        datasets_used="orders",
    )
    out = retriever.retrieve(
        "q", project_id="p1", k=5, datasets_used_contains="PAYMENTS"
    )
    assert len(out) == 1
    assert "payments" in out[0].memory["datasets_used"]


def test_created_after_filter_excludes_older(store, retriever):
    """5. created_after drops episodes whose created_at < the cutoff."""
    e1 = _seed(store, question="older q", conclusion="older c")
    # Pick a cutoff strictly between e1 and e2.
    time.sleep(0.01)
    cutoff = e1.created_at  # >= e1.created_at keeps e1
    # Build cutoff that excludes e1 by appending a high suffix.
    cutoff_exclusive = e1.created_at + "z"  # "z" sorts after digits/T/etc.
    time.sleep(0.01)
    _seed(store, question="newer q", conclusion="newer c")
    out_all = retriever.retrieve(
        "q", project_id="p1", k=5, created_after=cutoff
    )
    # Both kept when cutoff equals the older row's created_at.
    assert len(out_all) == 2
    out_newer = retriever.retrieve(
        "q", project_id="p1", k=5, created_after=cutoff_exclusive
    )
    # Only the newer episode survives.
    assert len(out_newer) == 1
    assert out_newer[0].memory["question"] == "newer q"


def test_multiple_filters_combine(store, retriever):
    """6. Multiple filters compose with AND semantics."""
    _seed(
        store,
        question="q",
        conclusion="c",
        analysis_type="root_cause",
        datasets_used="payments",
    )
    _seed(
        store,
        question="q",
        conclusion="c",
        analysis_type="root_cause",
        datasets_used="orders",
    )
    _seed(
        store,
        question="q",
        conclusion="c",
        analysis_type="forecast",
        datasets_used="payments",
    )
    out = retriever.retrieve(
        "q",
        project_id="p1",
        k=5,
        analysis_type="root_cause",
        datasets_used_contains="payments",
    )
    assert len(out) == 1
    assert out[0].memory["analysis_type"] == "root_cause"
    assert "payments" in out[0].memory["datasets_used"]


def test_k_limit_honored_after_filter(store, retriever):
    """7. k caps the result AFTER post-filter, not before."""
    for i in range(6):
        _seed(
            store,
            question=f"q{i}",
            conclusion="paid search",
            analysis_type="root_cause",
        )
    out = retriever.retrieve(
        "paid search", project_id="p1", k=2, analysis_type="root_cause"
    )
    assert len(out) == 2


def test_reason_includes_distance_and_filter_notes(store, retriever):
    """8. The reason string carries the distance and any active filter notes."""
    _seed(
        store,
        question="rev drop",
        conclusion="paid search",
        analysis_type="root_cause",
        datasets_used="payments",
    )
    out = retriever.retrieve(
        "rev drop",
        project_id="p1",
        k=5,
        analysis_type="root_cause",
        datasets_used_contains="payments",
    )
    assert len(out) == 1
    reason = out[0].reason
    assert reason.startswith("semantic match (distance=")
    assert "filtered:" in reason
    assert "analysis_type=root_cause" in reason
    assert "payments" in reason


def test_project_isolation(store, retriever):
    """9. Episodes in project A are invisible to project B."""
    _seed(store, project_id="A", question="rev drop", conclusion="c")
    out = retriever.retrieve("rev drop", project_id="B", k=5)
    assert out == []


def test_empty_query_returns_empty(store, retriever):
    """10. Empty / whitespace-only query returns []."""
    _seed(store, question="q", conclusion="c")
    assert retriever.retrieve("", project_id="p1") == []
    assert retriever.retrieve("   ", project_id="p1") == []


def test_no_filters_means_no_filter_notes(store, retriever):
    """11. Without filters, the reason has no 'filtered:' suffix."""
    _seed(store, question="rev drop", conclusion="c")
    out = retriever.retrieve("rev drop", project_id="p1", k=5)
    assert len(out) == 1
    assert "filtered:" not in out[0].reason
    assert out[0].reason.startswith("semantic match (distance=")


def test_category_tag_is_episode(store, retriever):
    """12. Every returned RetrievedMemory has category='episode'."""
    for i in range(3):
        _seed(store, question=f"q{i}", conclusion=f"c{i}")
    out = retriever.retrieve("q", project_id="p1", k=5)
    assert out
    assert all(r.category == "episode" for r in out)
