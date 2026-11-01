"""Pytest suite for :class:`mads.memory.retrievers.PreferenceRetriever`.

Closes the v1.1 retrieval-surface gap: preferences are now surfaced via
the proactive retriever (in addition to being writable via the tools).

Contract under test:

* ALL active preferences for the user are returned (no query filter).
* Every result has ``score=1.0`` (preferences are user-state, not
  relevance-ranked context).
* Reason string is exactly ``f"active user preference: '{key}'"``.
* Category is ``"preference"`` on every row.
* Tenancy is enforced: user A never sees user B's preferences.
"""

from __future__ import annotations

import pytest

from mads.memory.retrievers import PreferenceRetriever, RetrievedMemory
from mads.memory.user_preference import PreferenceStore


# --------------------------------------------------------------------- fixtures


@pytest.fixture
def store():
    s = PreferenceStore(db_path=":memory:")
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def retriever(store):
    return PreferenceRetriever(store)


# --------------------------------------------------------------------- tests


def test_returns_all_active_preferences(store, retriever):
    """1. ALL active preferences for the user are surfaced (no query filter)."""
    store.set("alice", "report_style", "exec summary first")
    store.set("alice", "tenure_buckets", "0-12, 13-24, 25-48, 49+")
    store.set("alice", "color_scheme", "dark")

    out = retriever.retrieve(user_id="alice")
    assert len(out) == 3
    keys = sorted(r.memory["preference_key"] for r in out)
    assert keys == ["color_scheme", "report_style", "tenure_buckets"]
    for rm in out:
        assert rm.score == 1.0
        assert isinstance(rm, RetrievedMemory)


def test_empty_when_no_preferences(store, retriever):
    """2. Fresh user with no preferences -> empty result."""
    assert retriever.retrieve(user_id="nobody") == []


def test_excludes_inactive_preferences(store, retriever):
    """3. ``forget`` closes the active row; it must not be returned."""
    store.set("alice", "k1", "v1")
    store.set("alice", "k2", "v2")
    store.forget("alice", "k1")

    out = retriever.retrieve(user_id="alice")
    assert len(out) == 1
    assert out[0].memory["preference_key"] == "k2"


def test_score_is_uniformly_one(store, retriever):
    """4. Every returned preference scores exactly 1.0, regardless of key."""
    for key, value in [
        ("a", "1"),
        ("bb_long", "2"),
        ("c_with_underscores", "3"),
        ("DDDD", "4"),
        ("e", "5"),
    ]:
        store.set("alice", key, value)

    out = retriever.retrieve(user_id="alice")
    assert len(out) == 5
    assert all(rm.score == 1.0 for rm in out)


def test_reason_format(store, retriever):
    """5 (v1.3). The reason string includes the effective scope label."""
    store.set("alice", "tenure_buckets", "0-12, 13-24, 25-48, 49+")
    out = retriever.retrieve(user_id="alice")
    assert len(out) == 1
    assert out[0].reason == (
        "active user preference: 'tenure_buckets' [scope: global]"
    )


def test_category_is_preference(store, retriever):
    """6. Category tag on every row is exactly ``"preference"``."""
    for k, v in [("a", "1"), ("b", "2"), ("c", "3")]:
        store.set("alice", k, v)

    out = retriever.retrieve(user_id="alice")
    assert all(rm.category == "preference" for rm in out)


def test_user_isolation(store, retriever):
    """7. Retrieve scoped to user_a never sees user_b's preferences."""
    store.set("alice", "a1", "v")
    store.set("alice", "a2", "v")
    store.set("bob", "b1", "v")

    out_alice = retriever.retrieve(user_id="alice")
    out_bob = retriever.retrieve(user_id="bob")
    assert {r.memory["preference_key"] for r in out_alice} == {"a1", "a2"}
    assert {r.memory["preference_key"] for r in out_bob} == {"b1"}


def test_rejects_empty_user_id(store, retriever):
    """8. Empty / non-string user_id returns []."""
    store.set("alice", "k", "v")
    assert retriever.retrieve(user_id="") == []
    assert retriever.retrieve(user_id=None) == []  # type: ignore[arg-type]


def test_k_cap_optional_truncates_deterministically(store, retriever):
    """9. When ``k`` is a positive int, the result is truncated *after*
    listing all preferences, preserving the deterministic preference_key-asc
    order from ``list_active``."""
    for k in ("a", "b", "c", "d", "e"):
        store.set("alice", k, "v")
    out = retriever.retrieve(user_id="alice", k=3)
    assert len(out) == 3
    assert [r.memory["preference_key"] for r in out] == ["a", "b", "c"]


def test_k_none_returns_all(store, retriever):
    """10. The default (``k=None``) returns ALL preferences without capping."""
    for k in ("a", "b", "c", "d", "e", "f"):
        store.set("alice", k, "v")
    out = retriever.retrieve(user_id="alice")
    assert len(out) == 6


def test_memory_dict_carries_value(store, retriever):
    """11. The returned ``memory`` dict includes preference_value so the LLM
    can act on the preference, not just see its key."""
    store.set("alice", "report_style", "3-bullet executive summary first")
    out = retriever.retrieve(user_id="alice")
    assert len(out) == 1
    assert out[0].memory["preference_value"] == (
        "3-bullet executive summary first"
    )


# ----------------------------------------------------------- axis-model project filter


def test_default_project_id_none_returns_only_global(store, retriever):
    """12. ``project_id=None`` returns ONLY global prefs (project_id IS NULL)."""
    store.set("alice", "global_k", "g")
    store.set("alice", "project_k", "p", project_id="proj1")
    out = retriever.retrieve(user_id="alice")
    keys = [r.memory["preference_key"] for r in out]
    assert keys == ["global_k"]


def test_project_id_set_returns_global_and_scoped(store, retriever):
    """13. ``project_id=X`` returns global prefs UNION project-scoped prefs
    for X (IDENTITY_OPTIONAL fan-in)."""
    store.set("alice", "global_k", "g")
    store.set("alice", "p1_k", "p1", project_id="proj1")
    store.set("alice", "p2_k", "p2", project_id="proj2")
    out = retriever.retrieve(user_id="alice", project_id="proj1")
    keys = sorted(r.memory["preference_key"] for r in out)
    assert keys == ["global_k", "p1_k"]


def test_project_id_isolation(store, retriever):
    """14. Two projects' scoped prefs don't bleed into each other."""
    store.set("alice", "rule", "v_proj1", project_id="proj1")
    store.set("alice", "rule", "v_proj2", project_id="proj2")

    out_proj1 = retriever.retrieve(user_id="alice", project_id="proj1")
    out_proj2 = retriever.retrieve(user_id="alice", project_id="proj2")

    p1_values = {r.memory["preference_value"] for r in out_proj1}
    p2_values = {r.memory["preference_value"] for r in out_proj2}

    assert "v_proj1" in p1_values
    assert "v_proj2" not in p1_values
    assert "v_proj2" in p2_values
    assert "v_proj1" not in p2_values


# ----------------------------------------------------------- v1.3 dataset axis


def test_dataset_fingerprint_none_excludes_dataset_rows(store, retriever):
    """15 (v1.3). With dataset_fingerprint=None, dataset-scoped rows are
    excluded; only global rows (assuming project_id also None) surface."""
    store.set("alice", "global_k", "g")
    store.set("alice", "dataset_k", "d", dataset_fingerprint="fp_churn")
    out = retriever.retrieve(user_id="alice")
    keys = [r.memory["preference_key"] for r in out]
    assert keys == ["global_k"]


def test_dataset_fingerprint_set_returns_global_and_dataset(store, retriever):
    """16 (v1.3). dataset_fingerprint=fp returns global UNION dataset-scoped
    prefs for fp, excluding rows scoped to a different dataset."""
    store.set("alice", "global_k", "g")
    store.set("alice", "d1_k", "v1", dataset_fingerprint="fp_a")
    store.set("alice", "d2_k", "v2", dataset_fingerprint="fp_b")
    out = retriever.retrieve(
        user_id="alice", dataset_fingerprint="fp_a"
    )
    keys = sorted(r.memory["preference_key"] for r in out)
    assert keys == ["d1_k", "global_k"]


def test_most_specific_wins_in_retrieval(store, retriever):
    """17 (v1.3). When global, project, and dataset rows exist for the SAME
    key, the retriever surfaces only the most-specific row — no duplicates."""
    store.set("alice", "k", "global_v")
    store.set("alice", "k", "project_v", project_id="proj1")
    store.set("alice", "k", "dataset_v", dataset_fingerprint="fp1")
    store.set(
        "alice", "k", "project_dataset_v",
        project_id="proj1", dataset_fingerprint="fp1",
    )
    out = retriever.retrieve(
        user_id="alice",
        project_id="proj1",
        dataset_fingerprint="fp1",
    )
    # ONE row for the key — the most specific.
    pref_rows = [r for r in out if r.memory["preference_key"] == "k"]
    assert len(pref_rows) == 1
    assert pref_rows[0].memory["preference_value"] == "project_dataset_v"


def test_scope_label_in_reason_global(store, retriever):
    """18 (v1.3). Reason label is 'global' for a global row."""
    store.set("alice", "k", "v")
    out = retriever.retrieve(user_id="alice")
    assert out[0].reason == "active user preference: 'k' [scope: global]"


def test_scope_label_in_reason_project(store, retriever):
    """19 (v1.3). Reason label is 'project' for a project-only row."""
    store.set("alice", "k", "v", project_id="proj1")
    out = retriever.retrieve(user_id="alice", project_id="proj1")
    assert out[0].reason == "active user preference: 'k' [scope: project]"


def test_scope_label_in_reason_dataset(store, retriever):
    """20 (v1.3). Reason label is 'dataset' for a dataset-only row."""
    store.set("alice", "k", "v", dataset_fingerprint="fp1")
    out = retriever.retrieve(user_id="alice", dataset_fingerprint="fp1")
    assert out[0].reason == "active user preference: 'k' [scope: dataset]"


def test_scope_label_in_reason_project_dataset(store, retriever):
    """21 (v1.3). Reason label is 'project+dataset' for a fully-scoped row."""
    store.set(
        "alice", "k", "v",
        project_id="proj1", dataset_fingerprint="fp1",
    )
    out = retriever.retrieve(
        user_id="alice",
        project_id="proj1",
        dataset_fingerprint="fp1",
    )
    assert out[0].reason == (
        "active user preference: 'k' [scope: project+dataset]"
    )


def test_dataset_isolation_other_dataset_excluded(store, retriever):
    """22 (v1.3). A pref scoped to fp_a does NOT surface for a session
    bound to fp_b — the cross_01 fix at the retriever level."""
    store.set(
        "alice", "rename_rule", "customerID → customer_id",
        dataset_fingerprint="fp_churn",
    )
    out_other = retriever.retrieve(
        user_id="alice", dataset_fingerprint="fp_synthetic"
    )
    assert out_other == []  # No rows surface for the other dataset.
