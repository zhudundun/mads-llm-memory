"""Pytest suite for the ``analysis_episode`` memory module.

All tests run against an in-memory SQLite store with :class:`HashEmbedder`
(no sentence-transformers model load). The store-level tests exercise
:class:`AnalysisEpisodeStore` directly; the tool-level tests exercise the
LangChain ``@tool`` callables returned by
:func:`make_analysis_episode_factory(...).tools()`.
"""

from __future__ import annotations

import math
import re
import time

import pytest

from mads.memory.analysis_episode import (
    AnalysisEpisode,
    AnalysisEpisodeStore,
)
from mads.memory.analysis_episode_tools import (
    AnalysisEpisodeToolFactory,
    make_analysis_episode_factory,
)
from mads.memory.embedding import (
    Embedder,
    HashEmbedder,
    SentenceTransformerEmbedder,
)


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


_UUID_HEX_RE = re.compile(r"^[0-9a-f]{32}$")


def _tools_by_name(tools):
    return {t.name: t for t in tools}


def _seed(
    store,
    *,
    user_id="u1",
    project_id="p1",
    question="Why did revenue drop on 2026-05-15?",
    conclusion="Revenue drop was caused by paid-search mix shift.",
    analysis_type=None,
    datasets_used=None,
    metrics_used=None,
    methods_used=None,
    code_refs=None,
    chart_refs=None,
    assumptions=None,
    user_feedback=None,
):
    return store.record(
        user_id=user_id,
        project_id=project_id,
        question=question,
        conclusion=conclusion,
        analysis_type=analysis_type,
        datasets_used=datasets_used,
        metrics_used=metrics_used,
        methods_used=methods_used,
        code_refs=code_refs,
        chart_refs=chart_refs,
        assumptions=assumptions,
        user_feedback=user_feedback,
    )


# =================================================================== EMBEDDER


def test_hash_embedder_is_deterministic():
    """1. Same input -> same vector across two HashEmbedder instances."""
    a = HashEmbedder().embed("revenue drop on tuesday")
    b = HashEmbedder().embed("revenue drop on tuesday")
    assert a == b


def test_hash_embedder_dim_is_384():
    """2. HashEmbedder default dim is 384 (matches the vec0 schema)."""
    he = HashEmbedder()
    assert he.dim == 384
    assert len(he.embed("hello")) == 384


def test_hash_embedder_different_inputs_different_outputs():
    """3. Distinct inputs produce distinct vectors."""
    he = HashEmbedder()
    v1 = he.embed("revenue drop")
    v2 = he.embed("churn model improvement")
    assert v1 != v2


def test_hash_embedder_shared_tokens_are_closer():
    """4. 'revenue drop' is closer to 'revenue rose' than to 'churn model'."""
    he = HashEmbedder()
    a = he.embed("revenue drop")
    b = he.embed("revenue rose")
    c = he.embed("churn model")

    def cos(x, y):
        return sum(xi * yi for xi, yi in zip(x, y))

    sim_ab = cos(a, b)
    sim_ac = cos(a, c)
    assert sim_ab > sim_ac


def test_hash_embedder_outputs_unit_norm():
    """5. L2 norm of every HashEmbedder output is ~1.0."""
    he = HashEmbedder()
    for text in ("revenue", "churn model", "Why did X happen on 2026-05-15?"):
        v = he.embed(text)
        norm = math.sqrt(sum(x * x for x in v))
        assert norm == pytest.approx(1.0, rel=1e-6, abs=1e-6)


def test_hash_embedder_empty_text_returns_sentinel_unit_vector():
    """6. Empty / punctuation-only text returns a unit vector (no NaN)."""
    he = HashEmbedder()
    v = he.embed("")
    norm = math.sqrt(sum(x * x for x in v))
    assert norm == pytest.approx(1.0)


def test_embedder_protocol_satisfied():
    """7. HashEmbedder and SentenceTransformerEmbedder satisfy the Protocol."""
    assert isinstance(HashEmbedder(), Embedder)
    # Constructing the sentence-transformer embedder is cheap (lazy load).
    assert isinstance(SentenceTransformerEmbedder(), Embedder)


# =================================================================== RECORD / GET


def test_record_returns_episode_with_uuid(store):
    """8. record returns an AnalysisEpisode with a 32-char hex UUID."""
    ep = _seed(store)
    assert isinstance(ep, AnalysisEpisode)
    assert _UUID_HEX_RE.match(ep.id)


def test_record_sets_timestamps(store):
    """9. created_at and updated_at are populated and equal on insert."""
    ep = _seed(store)
    assert ep.created_at
    assert ep.updated_at
    assert ep.created_at == ep.updated_at


def test_get_returns_recorded_episode(store):
    """10. get fetches the row we just inserted."""
    ep = _seed(store, question="Q1", conclusion="C1")
    out = store.get(project_id="p1", episode_id=ep.id)
    assert out is not None
    assert out.id == ep.id
    assert out.question == "Q1"
    assert out.conclusion == "C1"


def test_get_unknown_returns_none(store):
    """11. get for an unknown id returns None."""
    assert store.get(project_id="p1", episode_id="nope") is None


def test_record_empty_question_raises(store):
    """12. record with empty question raises ValueError."""
    with pytest.raises(ValueError):
        _seed(store, question="")
    with pytest.raises(ValueError):
        _seed(store, question="   ")


def test_record_empty_conclusion_raises(store):
    """13. record with empty conclusion raises ValueError."""
    with pytest.raises(ValueError):
        _seed(store, conclusion="")
    with pytest.raises(ValueError):
        _seed(store, conclusion="   ")


def test_record_inserts_into_both_tables(store):
    """14. record writes to BOTH main and vec tables (one row each)."""
    ep = _seed(store)
    main_count = store._conn.execute(
        "SELECT COUNT(*) AS c FROM analysis_episode_memory WHERE id = ?",
        (ep.id,),
    ).fetchone()["c"]
    vec_count = store._conn.execute(
        "SELECT COUNT(*) AS c FROM analysis_episode_vec WHERE episode_id = ?",
        (ep.id,),
    ).fetchone()["c"]
    assert main_count == 1
    assert vec_count == 1


def test_record_is_atomic_on_embed_failure(store):
    """15. If the embedder raises, NEITHER table has a row for the would-be episode."""

    class _Boom:
        @property
        def dim(self):
            return 384

        def embed(self, text):
            raise RuntimeError("embed exploded")

    # Swap embedder for this single record call.
    original = store._embedder
    store._embedder = _Boom()
    try:
        with pytest.raises(RuntimeError):
            _seed(store, question="qX", conclusion="cX")
    finally:
        store._embedder = original

    main_count = store._conn.execute(
        "SELECT COUNT(*) AS c FROM analysis_episode_memory"
    ).fetchone()["c"]
    vec_count = store._conn.execute(
        "SELECT COUNT(*) AS c FROM analysis_episode_vec"
    ).fetchone()["c"]
    assert main_count == 0
    assert vec_count == 0


# =================================================================== LIST / SEARCH


def test_list_recent_newest_first(store):
    """16. list_recent returns episodes ordered by created_at DESC."""
    e1 = _seed(store, question="Q1", conclusion="C1")
    time.sleep(0.005)
    e2 = _seed(store, question="Q2", conclusion="C2")
    time.sleep(0.005)
    e3 = _seed(store, question="Q3", conclusion="C3")
    rows = store.list_recent(project_id="p1", limit=10)
    assert [r.id for r in rows] == [e3.id, e2.id, e1.id]


def test_list_recent_respects_limit(store):
    """17. list_recent caps at `limit`."""
    for i in range(5):
        _seed(store, question=f"Q{i}", conclusion=f"C{i}")
        time.sleep(0.002)
    rows = store.list_recent(project_id="p1", limit=3)
    assert len(rows) == 3


def test_list_recent_empty_project(store):
    """18. list_recent for a never-touched project returns []."""
    assert store.list_recent(project_id="ghost") == []


def test_search_matches_question(store):
    """19. search hits against the question field."""
    _seed(store, question="Why did revenue drop?", conclusion="C")
    rows = store.search(project_id="p1", query="revenue")
    assert len(rows) == 1


def test_search_matches_conclusion(store):
    """20. search hits against the conclusion field."""
    _seed(store, question="Q", conclusion="The retention curve flattened.")
    rows = store.search(project_id="p1", query="retention")
    assert len(rows) == 1


def test_search_matches_assumptions(store):
    """21. search hits against the assumptions field."""
    _seed(
        store,
        question="Q",
        conclusion="C",
        assumptions="we excluded test_users via is_test_user=false",
    )
    rows = store.search(project_id="p1", query="is_test_user")
    assert len(rows) == 1


def test_search_matches_user_feedback(store):
    """22. search hits against user_feedback."""
    ep = _seed(store, question="Q", conclusion="C")
    store.update_feedback(
        user_id="u1",
        project_id="p1",
        episode_id=ep.id,
        feedback="follow-up: not a structural change",
    )
    rows = store.search(project_id="p1", query="structural")
    assert len(rows) == 1


def test_search_is_case_insensitive(store):
    """23. search matches regardless of case."""
    _seed(store, question="Why did REVENUE drop?", conclusion="C")
    rows_low = store.search(project_id="p1", query="revenue")
    rows_up = store.search(project_id="p1", query="REVENUE")
    rows_mixed = store.search(project_id="p1", query="Revenue")
    assert len(rows_low) == 1
    assert len(rows_up) == 1
    assert len(rows_mixed) == 1


def test_search_does_not_match_structured_fields(store):
    """24. search does NOT hit against analysis_type / datasets_used / etc."""
    _seed(
        store,
        question="Q",
        conclusion="C",
        analysis_type="root_cause",
        datasets_used="payments, traffic_attribution",
        metrics_used="net_revenue",
        methods_used="cohort_analysis",
        code_refs="notebooks/rev_drop.ipynb",
        chart_refs="charts/rev_drop.png",
    )
    assert store.search(project_id="p1", query="root_cause") == []
    assert store.search(project_id="p1", query="payments") == []
    assert store.search(project_id="p1", query="cohort_analysis") == []
    assert store.search(project_id="p1", query="rev_drop.ipynb") == []


def test_search_empty_project(store):
    """25. search for a never-touched project returns []."""
    assert store.search(project_id="ghost", query="revenue") == []


def test_search_empty_query_returns_empty(store):
    """26. search with an empty query returns []."""
    _seed(store)
    assert store.search(project_id="p1", query="") == []


# =================================================================== SEMANTIC


def test_find_similar_returns_tuples_ordered_by_distance(store):
    """27. find_similar returns (episode, distance) pairs ordered by distance asc."""
    _seed(store, question="revenue drop", conclusion="paid search shift")
    time.sleep(0.002)
    _seed(store, question="churn model", conclusion="add tenure feature")
    hits = store.find_similar(project_id="p1", query="revenue drop", k=5)
    assert len(hits) == 2
    assert all(isinstance(t, tuple) and len(t) == 2 for t in hits)
    assert all(isinstance(t[0], AnalysisEpisode) for t in hits)
    assert all(isinstance(t[1], float) for t in hits)
    # ordering
    dists = [d for _, d in hits]
    assert dists == sorted(dists)


def test_find_similar_respects_k(store):
    """28. find_similar caps at k results."""
    for i in range(4):
        _seed(store, question=f"Q{i}", conclusion=f"C{i}")
    hits = store.find_similar(project_id="p1", query="Q0", k=2)
    assert len(hits) == 2


def test_find_similar_empty_project(store):
    """29. find_similar for an empty project returns []."""
    assert store.find_similar(project_id="ghost", query="anything", k=3) == []


def test_find_similar_identical_text_is_closest(store):
    """30. The episode whose embed-text equals the query ranks first."""
    e_target = _seed(
        store,
        question="Why did revenue drop on 2026-05-15?",
        conclusion="Paid-search mix shift.",
    )
    _seed(store, question="How is the churn model?", conclusion="Adding tenure.")
    # Build the exact embed-text the store would build.
    query = "Why did revenue drop on 2026-05-15?\n\nPaid-search mix shift."
    hits = store.find_similar(project_id="p1", query=query, k=3)
    assert hits[0][0].id == e_target.id
    # Distance to identical text should be ~0.
    assert hits[0][1] < 0.01


def test_find_similar_respects_project_id(store):
    """31. find_similar does NOT bleed across projects."""
    _seed(store, project_id="A", question="revenue drop", conclusion="C")
    hits = store.find_similar(
        project_id="B", query="revenue drop", k=5
    )
    assert hits == []


# =================================================================== MUTABILITY


def test_update_feedback_bumps_updated_at(store):
    """32. update_feedback bumps updated_at strictly above the original."""
    ep = _seed(store)
    original_updated = ep.updated_at
    time.sleep(0.01)
    updated = store.update_feedback(
        user_id="u1",
        project_id="p1",
        episode_id=ep.id,
        feedback="Looked correct to me.",
    )
    assert updated is not None
    assert updated.updated_at > original_updated
    # created_at is NOT touched
    assert updated.created_at == ep.created_at


def test_update_feedback_changes_embedded_text(store):
    """33. After update_feedback, find_similar with the new text ranks the episode first."""
    target = _seed(
        store,
        question="Why did revenue drop?",
        conclusion="Paid search mix shift.",
    )
    _seed(
        store,
        question="How is churn?",
        conclusion="Adding tenure feature improved AUC.",
    )
    # "zorklewump" is a low-collision sentinel token under the HashEmbedder
    # (verified to land in a bucket disjoint from both seeded episodes), so a
    # query for it should rank the re-embedded episode strictly first.
    store.update_feedback(
        user_id="u1",
        project_id="p1",
        episode_id=target.id,
        feedback="Follow-up: zorklewump observed the next week.",
    )
    hits = store.find_similar(
        project_id="p1", query="zorklewump", k=2
    )
    assert hits[0][0].id == target.id


def test_update_feedback_unknown_id_returns_none(store):
    """34. update_feedback on a never-seen id returns None."""
    out = store.update_feedback(
        user_id="u1",
        project_id="p1",
        episode_id="ghost",
        feedback="anything",
    )
    assert out is None


def test_update_feedback_empty_raises(store):
    """35. update_feedback with empty feedback raises ValueError."""
    ep = _seed(store)
    with pytest.raises(ValueError):
        store.update_feedback(
            user_id="u1",
            project_id="p1",
            episode_id=ep.id,
            feedback="",
        )
    with pytest.raises(ValueError):
        store.update_feedback(
            user_id="u1",
            project_id="p1",
            episode_id=ep.id,
            feedback="   ",
        )


def test_other_fields_are_immutable_by_design(store):
    """36. The store exposes NO methods for editing fields other than user_feedback."""
    public = {
        name
        for name in dir(AnalysisEpisodeStore)
        if not name.startswith("_")
    }
    # The only mutator MUST be update_feedback.
    assert "update_feedback" in public
    # Sanity: none of these forbidden mutators exist.
    forbidden = {
        "update_question",
        "update_conclusion",
        "update_analysis_type",
        "update_assumptions",
        "update_datasets_used",
        "update_metrics_used",
        "delete",
        "delete_episode",
    }
    leaks = forbidden & public
    assert leaks == set(), f"forbidden mutators exposed: {leaks}"


def test_update_feedback_wrong_project_returns_none(store):
    """37. update_feedback with a mismatched project_id returns None."""
    ep = _seed(store, project_id="A")
    out = store.update_feedback(
        user_id="u1",
        project_id="B",
        episode_id=ep.id,
        feedback="anything",
    )
    assert out is None


# =================================================================== TENANCY


def test_project_isolation_get(store):
    """38. get for project A's episode from project B returns None."""
    ep = _seed(store, project_id="A")
    assert store.get(project_id="B", episode_id=ep.id) is None
    assert store.get(project_id="A", episode_id=ep.id) is not None


def test_project_isolation_list(store):
    """39. list_recent is project-scoped."""
    _seed(store, project_id="A")
    _seed(store, project_id="A")
    _seed(store, project_id="B")
    assert len(store.list_recent(project_id="A")) == 2
    assert len(store.list_recent(project_id="B")) == 1


def test_project_isolation_search(store):
    """40. search is project-scoped."""
    _seed(
        store,
        project_id="A",
        question="Why did revenue drop?",
        conclusion="x",
    )
    assert store.search(project_id="B", query="revenue") == []


def test_two_users_share_project_library(store):
    """41. Two users in the same project see each other's episodes."""
    ep_alice = _seed(store, user_id="alice", project_id="growth")
    rows = store.list_recent(project_id="growth")
    assert any(r.id == ep_alice.id for r in rows)
    # Bob can fetch it.
    fetched = store.get(project_id="growth", episode_id=ep_alice.id)
    assert fetched is not None
    assert fetched.user_id == "alice"


# =================================================================== TOOL LAYER


def test_factory_returns_exactly_six_tools(store):
    """42. Factory yields exactly 6 tools with the contracted names."""
    f = make_analysis_episode_factory(store, "u1", "p1")
    assert isinstance(f, AnalysisEpisodeToolFactory)
    tools = f.tools()
    assert len(tools) == 6
    names = sorted(t.name for t in tools)
    assert names == sorted(
        [
            "record_analysis_episode",
            "get_analysis_episode",
            "list_recent_analyses",
            "find_similar_analyses",
            "search_analyses",
            "update_user_feedback",
        ]
    )


def test_tool_signatures_hide_tenancy(store):
    """43. None of the 6 tools accept user_id or project_id as arguments."""
    tools = make_analysis_episode_factory(store, "u1", "p1").tools()
    for t in tools:
        schema = t.args_schema.model_json_schema()
        props = set(schema.get("properties", {}).keys())
        assert "user_id" not in props, (
            f"{t.name} leaks user_id in its arg schema: {props}"
        )
        assert "project_id" not in props, (
            f"{t.name} leaks project_id in its arg schema: {props}"
        )


def test_record_tool_returns_human_readable_string(store):
    """44. record_analysis_episode returns a readable string with the id."""
    tools = _tools_by_name(make_analysis_episode_factory(store, "u1", "p1").tools())
    out = tools["record_analysis_episode"].invoke(
        {
            "question": "Why did revenue drop on 2026-05-15?",
            "conclusion": "Paid-search mix shift.",
            "analysis_type": "root_cause",
        }
    )
    assert isinstance(out, str)
    assert "Recorded analysis episode" in out
    # 32-char hex UUID appears somewhere in the string.
    assert re.search(r"[0-9a-f]{32}", out)


def test_get_tool_empty_then_populated(store):
    """45. get_analysis_episode: empty-state, then formatted full view."""
    tools = _tools_by_name(make_analysis_episode_factory(store, "u1", "p1").tools())
    empty = tools["get_analysis_episode"].invoke({"episode_id": "nope"})
    assert "No analysis episode" in empty

    record_out = tools["record_analysis_episode"].invoke(
        {
            "question": "Q",
            "conclusion": "C",
        }
    )
    ep_id = re.search(r"[0-9a-f]{32}", record_out).group(0)
    populated = tools["get_analysis_episode"].invoke({"episode_id": ep_id})
    assert "question" in populated
    assert "conclusion" in populated
    assert ep_id in populated


def test_list_recent_tool_empty_then_populated(store):
    """46. list_recent_analyses: empty-state, then compact table."""
    tools = _tools_by_name(make_analysis_episode_factory(store, "u1", "p1").tools())
    out_empty = tools["list_recent_analyses"].invoke({"limit": 5})
    assert "No analysis episodes" in out_empty

    tools["record_analysis_episode"].invoke(
        {"question": "rev drop question", "conclusion": "rev drop conclusion"}
    )
    out = tools["list_recent_analyses"].invoke({"limit": 5})
    assert "Most recent" in out
    assert "rev drop" in out


def test_find_similar_tool_returns_ranked_hits(store):
    """47. find_similar_analyses returns hits ordered with distances."""
    tools = _tools_by_name(make_analysis_episode_factory(store, "u1", "p1").tools())
    tools["record_analysis_episode"].invoke(
        {"question": "revenue drop", "conclusion": "paid search shift"}
    )
    tools["record_analysis_episode"].invoke(
        {"question": "churn model", "conclusion": "add tenure feature"}
    )
    out = tools["find_similar_analyses"].invoke(
        {"query": "revenue drop", "k": 2}
    )
    assert "Semantic matches" in out
    assert "d=" in out  # distance label appears
    # Both episodes referenced.
    assert "revenue drop" in out
    assert "churn model" in out


def test_find_similar_tool_empty_state(store):
    """48. find_similar_analyses returns the empty-state message when none."""
    tools = _tools_by_name(make_analysis_episode_factory(store, "u1", "p1").tools())
    out = tools["find_similar_analyses"].invoke(
        {"query": "anything", "k": 3}
    )
    assert "No analysis episodes matched" in out


def test_search_tool_matches_and_empty(store):
    """49. search_analyses returns hits and an empty-state message."""
    tools = _tools_by_name(make_analysis_episode_factory(store, "u1", "p1").tools())
    tools["record_analysis_episode"].invoke(
        {"question": "Why did revenue drop?", "conclusion": "shift"}
    )
    hit = tools["search_analyses"].invoke({"query": "revenue"})
    assert "Substring matches" in hit
    assert "revenue" in hit.lower()
    miss = tools["search_analyses"].invoke({"query": "xyzqq"})
    assert "No analysis episodes matched" in miss
    err = tools["search_analyses"].invoke({"query": ""})
    assert "Error" in err


def test_update_feedback_tool_changes_and_re_embeds(store):
    """50. update_user_feedback tool bumps updated_at and re-embeds for retrieval."""
    tools = _tools_by_name(make_analysis_episode_factory(store, "u1", "p1").tools())
    record_out = tools["record_analysis_episode"].invoke(
        {"question": "Q", "conclusion": "C"}
    )
    ep_id = re.search(r"[0-9a-f]{32}", record_out).group(0)
    time.sleep(0.01)

    out = tools["update_user_feedback"].invoke(
        {
            "episode_id": ep_id,
            "feedback": "zorklewump recovery the next week",
        }
    )
    assert "Updated user_feedback" in out
    assert ep_id in out

    # Re-embedded: a semantic search for the new feedback text finds it.
    sim_out = tools["find_similar_analyses"].invoke(
        {"query": "zorklewump", "k": 1}
    )
    assert ep_id[:8] in sim_out

    # Unknown id -> error string.
    err = tools["update_user_feedback"].invoke(
        {"episode_id": "ghost", "feedback": "anything"}
    )
    assert "Error" in err

    # Empty feedback -> error string.
    err2 = tools["update_user_feedback"].invoke(
        {"episode_id": ep_id, "feedback": ""}
    )
    assert "Error" in err2


def test_two_factories_different_projects_isolated(store):
    """51. Factories with different project_id are isolated."""
    f_a = make_analysis_episode_factory(store, "u1", "A")
    f_b = make_analysis_episode_factory(store, "u1", "B")
    a = _tools_by_name(f_a.tools())
    b = _tools_by_name(f_b.tools())

    a["record_analysis_episode"].invoke(
        {"question": "A-only Q", "conclusion": "A-only C"}
    )
    out_a = a["list_recent_analyses"].invoke({"limit": 5})
    out_b = b["list_recent_analyses"].invoke({"limit": 5})
    assert "A-only" in out_a
    assert "No analysis episodes" in out_b


def test_factory_rejects_empty_tenancy(store):
    """52. make_analysis_episode_factory rejects empty user_id / project_id."""
    with pytest.raises(ValueError):
        make_analysis_episode_factory(store, "", "p1")
    with pytest.raises(ValueError):
        make_analysis_episode_factory(store, "u1", "")
