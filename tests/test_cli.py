"""Tests for :mod:`mads.cli`.

We exercise the argparse surface, env-var precedence, and the memory
subcommands directly against a temporary SQLite db. ``ask`` is exercised with
a monkeypatched ``wrap_*`` so no LLM is invoked.
"""

from __future__ import annotations

import os
import sys
from unittest import mock

import pandas as pd
import pytest

from mads import cli
from mads.adapters.ads import AgentResult
from mads.agents.types import TurnArtifacts


# --------------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    """Strip MADS_* env vars and ANTHROPIC_API_KEY so each test starts clean."""
    for k in ("MADS_USER", "MADS_PROJECT", "MADS_DB_PATH", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("USER", "tester")
    # Direct stores at a per-test sqlite.
    db = str(tmp_path / "cli.sqlite")
    monkeypatch.setenv("MADS_DB_PATH", db)
    yield db


# --------------------------------------------------------------------- helpers


def _run(argv, capsys, exit_code=None):
    rc = cli.main(argv)
    out, err = capsys.readouterr()
    if exit_code is not None:
        assert rc == exit_code, f"rc={rc}; stderr={err!r}"
    return rc, out, err


# --------------------------------------------------------------------- shape / help


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])
    assert excinfo.value.code == 0


def test_no_subcommand_prints_help_and_exits_2(capsys):
    rc, out, err = _run([], capsys)
    assert rc == 2


def test_memory_no_subcommand_exits_2(capsys):
    rc, out, err = _run(["memory"], capsys)
    assert rc == 2


# --------------------------------------------------------------------- env defaults


def test_missing_user_for_memory_subcommand_exits_2(capsys, monkeypatch):
    monkeypatch.delenv("USER", raising=False)
    rc, out, err = _run(["memory", "list", "metric"], capsys)
    assert rc == 2
    assert "user" in err.lower()


def test_mads_user_env_var_used(capsys, monkeypatch):
    monkeypatch.setenv("MADS_USER", "envuser")
    rc, out, err = _run(["memory", "list", "metric"], capsys)
    assert rc == 0
    # Empty-state message OK.


def test_env_db_path_resolves_when_no_arg():
    # _resolve_db should return the env var when no arg is given.
    assert cli._resolve_db(None) == os.environ["MADS_DB_PATH"]


def test_arg_db_path_wins_over_env(monkeypatch):
    monkeypatch.setenv("MADS_DB_PATH", "/tmp/env.sqlite")
    assert cli._resolve_db("/tmp/arg.sqlite") == "/tmp/arg.sqlite"


def test_user_arg_wins_over_env(monkeypatch):
    monkeypatch.setenv("MADS_USER", "envuser")
    monkeypatch.setenv("USER", "shelluser")
    assert cli._resolve_user("argv-user") == "argv-user"


def test_project_default_is_default(monkeypatch):
    monkeypatch.delenv("MADS_PROJECT", raising=False)
    assert cli._resolve_project(None) == "default"


# --------------------------------------------------------------------- memory list (happy paths)


def test_memory_list_metric_empty_state(capsys):
    rc, out, err = _run(["memory", "list", "metric", "--user", "u"], capsys)
    assert rc == 0
    assert "No metrics" in out


def test_memory_list_episode_empty_state(capsys):
    rc, out, err = _run(["memory", "list", "episode", "--user", "u"], capsys)
    assert rc == 0
    # The episode-store empty state message contains "No analyses" or "no recent".
    assert "No " in out or "no " in out


def test_memory_list_dataset_empty_state(capsys):
    rc, out, err = _run(["memory", "list", "dataset", "--user", "u"], capsys)
    assert rc == 0


# --------------------------------------------------------------------- memory show


def test_memory_show_metric_unknown_emits_friendly(capsys):
    rc, out, err = _run(
        ["memory", "show", "metric", "nonexistent", "--user", "u"], capsys
    )
    assert rc == 0
    assert "No metric" in out


def test_memory_history_metric_unknown(capsys):
    rc, out, err = _run(
        ["memory", "history", "metric", "nope", "--user", "u"], capsys
    )
    assert rc == 0
    assert "No history" in out


# --------------------------------------------------------------------- ask (mocked)


def _fake_turn(agent_name="eda") -> TurnArtifacts:
    return TurnArtifacts(
        question="q",
        agent_name=agent_name,
        agent_result=AgentResult(
            response="mocked",
            tool_calls=[],
            prompt_tokens=1,
            completion_tokens=2,
            total_tokens=3,
        ),
        memory_reads=[],
        memory_writes=[],
    )


class _FakeWrapped:
    def __init__(self, agent_name="eda"):
        self.agent_name = agent_name

    def invoke(self, prompt):
        return _fake_turn(self.agent_name)


def test_ask_without_api_key_exits_2(capsys, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    rc, out, err = _run(["ask", "load /tmp/x.csv", "--user", "u"], capsys)
    assert rc == 2
    assert "ANTHROPIC_API_KEY" in err


def test_ask_loader_routes_and_invokes(capsys, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")

    def fake_wrap_loader(manager, **kw):
        return _FakeWrapped(agent_name="loader")

    monkeypatch.setattr("mads.agents.wrapper.wrap_loader", fake_wrap_loader)
    rc, out, err = _run(["ask", "load the file", "--user", "u"], capsys)
    assert rc == 0
    assert "Answer (loader)" in out
    assert "mocked" in out


def test_ask_eda_requires_data(capsys, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    rc, out, err = _run(
        ["ask", "describe the dataset", "--user", "u"], capsys
    )
    assert rc == 2
    assert "--data" in err


def test_ask_viz_with_data(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    csv = tmp_path / "x.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(csv, index=False)

    def fake_wrap_viz(manager, df, *, dataset_name, table_name, **kw):
        return _FakeWrapped(agent_name="viz")

    monkeypatch.setattr("mads.agents.wrapper.wrap_viz", fake_wrap_viz)
    rc, out, err = _run(
        ["ask", "plot a vs b", "--user", "u", "--data", str(csv)], capsys
    )
    assert rc == 0
    assert "Answer (viz)" in out


# --------------------------------------------------------------------- similar


def test_memory_similar_episode_empty_state(capsys):
    rc, out, err = _run(
        ["memory", "similar", "episode", "anything", "--user", "u", "--k", "3"],
        capsys,
    )
    assert rc == 0
