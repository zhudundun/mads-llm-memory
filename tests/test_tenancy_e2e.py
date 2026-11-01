"""Cross-layer tenancy tests.

These tests confirm the tenancy contract that V1_DESIGN_BRIEF and CLAUDE.md
both stake out: ``user_id`` / ``project_id`` are NEVER exposed in any tool
signature visible to the LLM. They are bound at the factory layer, and
isolation is enforced at the store layer.

We test:

* Two ``MemoryManager`` instances on the same db with different ``project_id``
  cannot see each other's metrics or episodes.
* Same ``project_id`` but different ``user_id`` can still read each other's
  metrics (a metric is a project-level definition).
* Episodes recorded under one user are returned by ``list_recent`` for any
  user in the same project (project-scoped).
* No memory tool's ``args_schema`` exposes ``user_id`` or ``project_id``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mads.manager import MemoryManager
from mads.memory import HashEmbedder


@pytest.fixture
def shared_db(tmp_path) -> str:
    return str(tmp_path / "shared_tenancy.sqlite")


# --------------------------------------------------------------------- helpers


def _find(tools, name):
    for t in tools:
        if t.name == name:
            return t
    raise KeyError(name)


# --------------------------------------------------------------------- metric isolation


def test_two_projects_metrics_isolated(shared_db):
    m1 = MemoryManager(
        user_id="alice", project_id="p1", db_path=shared_db, embedder=HashEmbedder()
    )
    m2 = MemoryManager(
        user_id="alice", project_id="p2", db_path=shared_db, embedder=HashEmbedder()
    )
    try:
        add1 = _find(m1.metric.tools(), "add_metric_definition")
        out = add1.invoke(
            {"metric_name": "revenue", "business_definition": "sum(amount)"}
        )
        assert "Metric 'revenue'" in out

        # p2 cannot see p1's metric.
        lst2 = _find(m2.metric.tools(), "list_metric_definitions")
        assert "revenue" not in lst2.invoke({})

        # And vice versa, p1 doesn't see p2's metric.
        add2 = _find(m2.metric.tools(), "add_metric_definition")
        add2.invoke(
            {"metric_name": "ARPU", "business_definition": "revenue / users"}
        )
        lst1 = _find(m1.metric.tools(), "list_metric_definitions")
        assert "ARPU" not in lst1.invoke({})
    finally:
        m1.close()
        m2.close()


# --------------------------------------------------------------------- episode isolation


def test_two_projects_episodes_isolated(shared_db):
    m1 = MemoryManager(
        user_id="alice", project_id="p1", db_path=shared_db, embedder=HashEmbedder()
    )
    m2 = MemoryManager(
        user_id="alice", project_id="p2", db_path=shared_db, embedder=HashEmbedder()
    )
    try:
        rec1 = _find(m1.episode.tools(), "record_analysis_episode")
        rec1.invoke({"question": "Q in p1", "conclusion": "C in p1"})

        lst2 = _find(m2.episode.tools(), "list_recent_analyses")
        out2 = lst2.invoke({"limit": 10})
        assert "Q in p1" not in out2

        # p1 sees its own.
        lst1 = _find(m1.episode.tools(), "list_recent_analyses")
        assert "Q in p1" in lst1.invoke({"limit": 10})
    finally:
        m1.close()
        m2.close()


# --------------------------------------------------------------------- same-project cross-user


def test_same_project_metrics_visible_across_users(shared_db):
    m1 = MemoryManager(
        user_id="alice", project_id="team", db_path=shared_db, embedder=HashEmbedder()
    )
    m2 = MemoryManager(
        user_id="bob", project_id="team", db_path=shared_db, embedder=HashEmbedder()
    )
    try:
        add1 = _find(m1.metric.tools(), "add_metric_definition")
        add1.invoke(
            {"metric_name": "active_users", "business_definition": "distinct(user_id)"}
        )
        lst2 = _find(m2.metric.tools(), "list_metric_definitions")
        assert "active_users" in lst2.invoke({})
    finally:
        m1.close()
        m2.close()


def test_same_project_episodes_visible_across_users(shared_db):
    m1 = MemoryManager(
        user_id="alice", project_id="team", db_path=shared_db, embedder=HashEmbedder()
    )
    m2 = MemoryManager(
        user_id="bob", project_id="team", db_path=shared_db, embedder=HashEmbedder()
    )
    try:
        rec = _find(m1.episode.tools(), "record_analysis_episode")
        rec.invoke({"question": "alice_q", "conclusion": "alice_c"})
        lst = _find(m2.episode.tools(), "list_recent_analyses")
        assert "alice_q" in lst.invoke({"limit": 10})
    finally:
        m1.close()
        m2.close()


# --------------------------------------------------------------------- args_schema tenancy


def _schema_fields(tool) -> set:
    """Return the set of field names in the tool's args_schema (if any)."""
    schema = getattr(tool, "args_schema", None)
    if schema is None:
        return set()
    # Pydantic v2: model_fields
    fields = getattr(schema, "model_fields", None)
    if fields is not None:
        return set(fields.keys())
    # Pydantic v1 fallback
    return set(getattr(schema, "__fields__", {}).keys())


def test_metric_tools_args_schema_excludes_tenancy(shared_db):
    m = MemoryManager(
        user_id="u", project_id="p", db_path=shared_db, embedder=HashEmbedder()
    )
    try:
        for tool in m.metric.tools():
            fields = _schema_fields(tool)
            assert "user_id" not in fields, f"{tool.name} leaks user_id"
            assert "project_id" not in fields, f"{tool.name} leaks project_id"
    finally:
        m.close()


def test_episode_tools_args_schema_excludes_tenancy(shared_db):
    m = MemoryManager(
        user_id="u", project_id="p", db_path=shared_db, embedder=HashEmbedder()
    )
    try:
        for tool in m.episode.tools():
            fields = _schema_fields(tool)
            assert "user_id" not in fields, f"{tool.name} leaks user_id"
            assert "project_id" not in fields, f"{tool.name} leaks project_id"
    finally:
        m.close()


def test_dataset_tools_args_schema_excludes_tenancy(shared_db):
    m = MemoryManager(
        user_id="u", project_id="p", db_path=shared_db, embedder=HashEmbedder()
    )
    try:
        m.bind_dataframe(
            pd.DataFrame({"a": [1]}), dataset_name="d", table_name="t"
        )
        for tool in m.dataset_tools():
            fields = _schema_fields(tool)
            assert "user_id" not in fields, f"{tool.name} leaks user_id"
            assert "project_id" not in fields, f"{tool.name} leaks project_id"
    finally:
        m.close()
