"""Tests for the pure-templating :func:`mads.composer.compose`."""

from __future__ import annotations

import pytest

from mads.adapters.ads import AgentResult
from mads.agents.types import TurnArtifacts
from mads.composer import compose


def _turn(**kwargs) -> TurnArtifacts:
    defaults = dict(
        question="q",
        agent_name="eda",
        agent_result=AgentResult(
            response="hi",
            tool_calls=[],
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        ),
        memory_reads=[],
        memory_writes=[],
    )
    defaults.update(kwargs)
    return TurnArtifacts(**defaults)


def test_header_includes_agent_name():
    out = compose(_turn())
    assert out.startswith("# Answer (eda)")


def test_response_appears():
    out = compose(_turn())
    assert "hi" in out


def test_none_response_renders_placeholder():
    out = compose(
        _turn(
            agent_result=AgentResult(
                response=None,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            )
        )
    )
    assert "_(no response)_" in out


def test_viz_code_section_appears_only_if_set():
    no_viz = compose(_turn())
    assert "Generated code" not in no_viz

    with_viz = compose(
        _turn(
            agent_result=AgentResult(
                response="x",
                viz_code="def plot(df): ...",
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            )
        )
    )
    assert "Generated code" in with_viz
    assert "def plot(df): ..." in with_viz


def test_memory_consulted_section_conditional():
    no_reads = compose(_turn())
    assert "Memory consulted" not in no_reads
    with_reads = compose(
        _turn(
            memory_reads=[
                {"name": "list_metric_definitions", "args": {}},
            ]
        )
    )
    assert "Memory consulted (1)" in with_reads
    assert "list_metric_definitions" in with_reads


def test_memory_written_section_conditional():
    no_writes = compose(_turn())
    assert "Memory written" not in no_writes
    with_writes = compose(
        _turn(
            memory_writes=[
                {"name": "add_metric_definition", "args": {"metric_name": "x"}},
            ]
        )
    )
    assert "Memory written (1)" in with_writes
    assert "metric_name=x" in with_writes


def test_tokens_line():
    out = compose(_turn())
    assert "_tokens: 30 (in=10, out=20)_" in out


def test_idempotent_pure():
    t = _turn()
    a = compose(t)
    b = compose(t)
    assert a == b


def test_arg_truncation():
    long = "x" * 500
    out = compose(
        _turn(
            memory_writes=[
                {"name": "add_metric_definition", "args": {"k": long}},
            ]
        )
    )
    # The arg should not appear in full
    assert long not in out
    assert "…" in out


def test_agent_name_loader():
    out = compose(_turn(agent_name="loader"))
    assert "# Answer (loader)" in out


# --------------------------------------------------------------------- new
# Tests for the proactive ``## Memories used (N)`` provenance section.
# Driven off ``TurnArtifacts.memories_injected``; absent when empty so
# legacy reactive wrappers (which never populate this field) keep their
# current output unchanged.

from mads.memory.retrievers import RetrievedMemory  # noqa: E402


def _metric_rm(name="conversion", definition="signups / sessions"):
    return RetrievedMemory(
        memory={
            "metric_name": name,
            "normalized_name": name,
            "business_definition": definition,
        },
        score=0.7,
        reason="tokenized match",
        category="metric",
    )


def _dataset_rm(table="orders", grain=None):
    return RetrievedMemory(
        memory={"table_name": table, "dataset_name": table, "grain": grain},
        score=0.7,
        reason="semantic_fingerprint match",
        category="dataset",
    )


def _episode_rm(question="Q3 conversion analysis", created_at="2026-05-20T10:33:14+00:00"):
    return RetrievedMemory(
        memory={
            "id": "ep123",
            "question": question,
            "conclusion": "concluded",
            "created_at": created_at,
        },
        score=0.6,
        reason="semantic match (distance=0.20)",
        category="episode",
    )


def test_memories_used_section_appears_when_nonempty():
    out = compose(_turn(memories_injected=[_metric_rm()]))
    assert "## Memories used (1)" in out
    assert "1. MetricMemory: conversion = signups / sessions" in out


def test_memories_used_section_absent_when_empty():
    out = compose(_turn())
    assert "Memories used" not in out


def test_memories_used_uses_numbered_list_not_bullets():
    rms = [_metric_rm(), _dataset_rm(table="orders"), _episode_rm()]
    out = compose(_turn(memories_injected=rms))
    assert "## Memories used (3)" in out
    # numbered, not "- "
    assert "1. MetricMemory:" in out
    assert "2. DatasetMemory:" in out
    assert "3. EpisodeMemory:" in out


def test_memories_used_section_ordering_before_consulted():
    """Memories used must come BEFORE Memory consulted in the output."""
    out = compose(
        _turn(
            memories_injected=[_metric_rm()],
            memory_reads=[{"name": "list_metric_definitions", "args": {}}],
        )
    )
    idx_used = out.index("## Memories used")
    idx_consulted = out.index("## Memory consulted")
    assert idx_used < idx_consulted


def test_memories_used_metric_summary_format():
    out = compose(
        _turn(
            memories_injected=[
                _metric_rm(name="revenue", definition="sum of paid orders")
            ]
        )
    )
    assert "MetricMemory: revenue = sum of paid orders" in out


def test_memories_used_dataset_summary_with_grain():
    out = compose(
        _turn(
            memories_injected=[_dataset_rm(table="orders", grain="one row per order")]
        )
    )
    assert "DatasetMemory: orders grain = one row per order" in out


def test_memories_used_dataset_summary_without_grain():
    out = compose(_turn(memories_injected=[_dataset_rm(table="orders", grain=None)]))
    assert "DatasetMemory: orders" in out
    assert "grain" not in out


def test_memories_used_episode_summary_with_date():
    out = compose(
        _turn(
            memories_injected=[
                _episode_rm(
                    question="Q3 conversion analysis",
                    created_at="2026-05-20T10:33:14+00:00",
                )
            ]
        )
    )
    assert 'EpisodeMemory: "Q3 conversion analysis" (2026-05-20)' in out


def test_memories_used_truncates_long_metric_definition():
    long_def = "x" * 500
    out = compose(
        _turn(memories_injected=[_metric_rm(name="m", definition=long_def)])
    )
    # The full long def shouldn't appear; ellipsis should.
    assert long_def not in out
    assert "…" in out


def test_memories_used_truncates_long_episode_question():
    long_q = "y" * 500
    out = compose(
        _turn(memories_injected=[_episode_rm(question=long_q)])
    )
    assert long_q not in out
    assert "…" in out


def test_memories_used_falls_back_to_sql_when_business_missing():
    rm = RetrievedMemory(
        memory={
            "metric_name": "x",
            "normalized_name": "x",
            "business_definition": "",
            "sql_definition": "SELECT 1",
        },
        score=0.5,
        reason="exact name match",
        category="metric",
    )
    out = compose(_turn(memories_injected=[rm]))
    assert "MetricMemory: x = SELECT 1" in out
