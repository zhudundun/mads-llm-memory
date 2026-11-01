"""Tests for :class:`mads.wrappers.memory_aware.MemoryAwareEDA`.

The wrapper is mostly orchestration; these tests verify:

* ``ctx.profile_summary`` lands inside the prompt sent to the ADS handle.
* The hybrid retriever is called with the df-derived semantic fingerprint.
* :meth:`MemoryManager.format_context` renders citation-tagged blocks.
* The validators are called post-invoke (warn-only path).
* Validator warnings surface on :class:`TurnArtifacts.warnings`.
* Retrieved memories surface on :class:`TurnArtifacts.memories_injected`.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from mads.adapters.ads import AgentResult
from mads.agents.types import TurnArtifacts
from mads.manager import MemoryManager
from mads.memory import HashEmbedder
from mads.memory.retrievers import RetrievedMemory
from mads.wrappers.memory_aware import MemoryAwareEDA


# --------------------------------------------------------------------- fixtures


@pytest.fixture
def manager(tmp_path) -> MemoryManager:
    db = str(tmp_path / "ma.sqlite")
    m = MemoryManager(
        user_id="u", project_id="p", db_path=db, embedder=HashEmbedder()
    )
    yield m
    m.close()


@pytest.fixture
def df() -> pd.DataFrame:
    # Distinct columns so the semantic_fingerprint is non-degenerate.
    return pd.DataFrame(
        {
            "channel": ["organic", "paid", "referral"] * 3 + ["organic"],
            "sessions": list(range(10)),
            "signups": [0, 1, 0, 2, 0, 0, 1, 1, 0, 1],
        }
    )


class _FakeHandle:
    """Captures the prompt passed to invoke()."""

    def __init__(self, response: str = "ok", tool_calls=None):
        self._response = response
        self._tool_calls = tool_calls or []
        self.last_prompt: str | None = None

    def invoke(self, prompt: str) -> AgentResult:
        self.last_prompt = prompt
        return AgentResult(
            response=self._response,
            tool_calls=list(self._tool_calls),
            prompt_tokens=1,
            completion_tokens=2,
            total_tokens=3,
        )


# --------------------------------------------------------------------- tests


def test_profile_summary_in_enriched_prompt(manager, df, monkeypatch):
    """The ctx.profile_summary lines must appear in the prompt the LLM sees."""
    fake = _FakeHandle()

    def fake_make_eda(d, *, extra_tools=None, model=None):
        return fake

    monkeypatch.setattr("mads.wrappers.memory_aware.make_eda", fake_make_eda)

    wrapper = MemoryAwareEDA(manager)
    wrapper.run(
        "Show conversion by channel",
        df,
        user_id="u",
        project_id="p",
        dataset_name="channels",
        table_name="channels",
    )

    assert fake.last_prompt is not None
    # The deterministic first line of profile_summary is "Dataset: <name>".
    assert "Dataset: channels" in fake.last_prompt
    # And the row-count line includes the literal "10 rows".
    assert "10 rows" in fake.last_prompt
    # User question is present too.
    assert "Show conversion by channel" in fake.last_prompt


def test_hybrid_retrieval_called_with_fingerprint(manager, df, monkeypatch):
    """``retrieve`` must receive the df's semantic_fingerprint as the filter."""
    fake = _FakeHandle()
    monkeypatch.setattr(
        "mads.wrappers.memory_aware.make_eda",
        lambda d, *, extra_tools=None, model=None: fake,
    )

    captured: dict[str, Any] = {}
    real_retrieve = manager.retrieve

    def spy_retrieve(**kwargs):
        captured.update(kwargs)
        return real_retrieve(**kwargs)

    monkeypatch.setattr(manager, "retrieve", spy_retrieve)

    wrapper = MemoryAwareEDA(manager)
    wrapper.run(
        "anything",
        df,
        user_id="u",
        project_id="p",
        dataset_name="channels",
        table_name="channels",
    )

    # The wrapper computes the fingerprint via the profiler and passes it
    # through as ``dataset_fingerprint``; we don't recompute the hash here,
    # just assert it was threaded through as a non-empty hex string.
    assert "dataset_fingerprint" in captured
    fp = captured["dataset_fingerprint"]
    assert isinstance(fp, str) and len(fp) >= 16
    assert captured["user_id"] == "u"
    assert captured["project_id"] == "p"
    assert captured["query"] == "anything"


def test_memory_block_format_citation_tags():
    """``format_context`` must emit ``[memory:CATEGORY:ID]`` headers."""
    memories = [
        RetrievedMemory(
            memory={
                "normalized_name": "conversion",
                "metric_name": "conversion",
                "business_definition": "signups / sessions",
                "sql_definition": "SELECT SUM(signups)/SUM(sessions) FROM t",
            },
            score=1.0,
            reason="exact name match",
            category="metric",
        ),
        RetrievedMemory(
            memory={
                "id": "ep-123",
                "question": "Why did conv drop?",
                "conclusion": "Paid traffic regressed.",
                "assumptions": "first-touch attribution",
            },
            score=0.78,
            reason="semantic match (distance=0.44)",
            category="episode",
        ),
    ]
    out = MemoryManager.format_context(memories)
    assert out.startswith("[memory:metric:conversion] (score=1.00)")
    assert "business: signups / sessions" in out
    assert "[memory:episode:ep-123] (score=0.78)" in out
    assert "question: Why did conv drop?" in out

    # Empty list returns sentinel.
    assert MemoryManager.format_context([]) == "(no relevant memories found)"


def test_validate_runs_post_invoke(manager, df, monkeypatch):
    """The manager's ``validate`` must be called with the AgentResult."""
    fake = _FakeHandle()
    monkeypatch.setattr(
        "mads.wrappers.memory_aware.make_eda",
        lambda d, *, extra_tools=None, model=None: fake,
    )

    seen: dict[str, Any] = {}

    def spy_validate(agent_result, *, memories_used):
        seen["agent_result"] = agent_result
        seen["memories_used"] = memories_used
        return []

    monkeypatch.setattr(manager, "validate", spy_validate)

    wrapper = MemoryAwareEDA(manager)
    wrapper.run(
        "anything",
        df,
        user_id="u",
        project_id="p",
        dataset_name="channels",
        table_name="channels",
    )

    assert "agent_result" in seen
    assert isinstance(seen["agent_result"], AgentResult)
    # And it was called *after* the fake handle was invoked (the result
    # we got handed has the fake's response).
    assert seen["agent_result"].response == "ok"
    # memories_used was threaded through (list, possibly empty).
    assert isinstance(seen["memories_used"], list)


def test_warnings_surface_on_turn_artifacts(manager, df, monkeypatch):
    """Validator warnings must end up on ``TurnArtifacts.warnings``."""
    fake = _FakeHandle()
    monkeypatch.setattr(
        "mads.wrappers.memory_aware.make_eda",
        lambda d, *, extra_tools=None, model=None: fake,
    )
    monkeypatch.setattr(
        manager,
        "validate",
        lambda ar, *, memories_used: ["fake warn A", "fake warn B"],
    )

    wrapper = MemoryAwareEDA(manager)
    ta = wrapper.run(
        "q",
        df,
        user_id="u",
        project_id="p",
        dataset_name="channels",
        table_name="channels",
    )
    assert isinstance(ta, TurnArtifacts)
    assert ta.warnings == ["fake warn A", "fake warn B"]


def test_memories_injected_returned_in_trace(manager, df, monkeypatch):
    """Retrieved memories must surface on ``TurnArtifacts.memories_injected``."""
    fake = _FakeHandle()
    monkeypatch.setattr(
        "mads.wrappers.memory_aware.make_eda",
        lambda d, *, extra_tools=None, model=None: fake,
    )

    fake_memory = RetrievedMemory(
        memory={
            "normalized_name": "conversion",
            "metric_name": "conversion",
            "business_definition": "signups / sessions",
        },
        score=1.0,
        reason="exact name match",
        category="metric",
    )
    monkeypatch.setattr(
        manager,
        "retrieve",
        lambda **kw: [fake_memory],
    )

    wrapper = MemoryAwareEDA(manager)
    ta = wrapper.run(
        "Show conversion by channel",
        df,
        user_id="u",
        project_id="p",
        dataset_name="channels",
        table_name="channels",
    )
    assert len(ta.memories_injected) == 1
    assert ta.memories_injected[0] is fake_memory
    # And the memory block was injected into the prompt.
    assert "[memory:metric:conversion]" in (fake.last_prompt or "")
