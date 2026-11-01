"""Tests for :mod:`eval.compare`.

Use synthetic JSONL fixtures (no real eval output) to verify the diff table
logic, label semantics, and exit-code contract.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make `eval/` importable as a top-level module.
REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import compare  # noqa: E402  — eval/compare.py


# --------------------------------------------------------------------- fixtures


def _write_runs(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


@pytest.fixture
def baseline_dir(tmp_path) -> Path:
    p = tmp_path / "results" / "loader"
    _write_runs(
        p / "runs.jsonl",
        [
            {"scenario_id": "loader_01_snake_case", "b_referenced_a": False,
             "b_referenced_a_reason": "preview retains original casing"},
            {"scenario_id": "loader_02_no_phone_service_null", "b_referenced_a": False,
             "b_referenced_a_reason": "no null-coercion rule"},
            {"scenario_id": "loader_03_drop_pii", "b_referenced_a": True,
             "b_referenced_a_reason": "customerID absent"},
        ],
    )
    return p


@pytest.fixture
def v1_dir(tmp_path) -> Path:
    p = tmp_path / "results" / "loader__with_memory"
    _write_runs(
        p / "runs.jsonl",
        [
            {"scenario_id": "loader_01_snake_case", "b_referenced_a": True,
             "b_referenced_a_reason": "preview shows snake_case columns"},
            {"scenario_id": "loader_02_no_phone_service_null", "b_referenced_a": False,
             "b_referenced_a_reason": "still no coercion"},
            {"scenario_id": "loader_03_drop_pii", "b_referenced_a": True,
             "b_referenced_a_reason": "customerID absent"},
        ],
    )
    return p


# --------------------------------------------------------------------- load_runs


def test_load_runs_returns_empty_for_missing_file(tmp_path):
    assert compare.load_runs(tmp_path / "nope.jsonl") == {}


def test_load_runs_skips_malformed_lines(tmp_path):
    p = tmp_path / "runs.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps({"scenario_id": "ok", "b_referenced_a": True}),
                "{not valid json",
                "",
                json.dumps({"scenario_id": "ok2", "b_referenced_a": False}),
            ]
        )
    )
    rows = compare.load_runs(p)
    assert set(rows) == {"ok", "ok2"}


# --------------------------------------------------------------------- classify


def test_classify_labels():
    assert compare.classify({"b_referenced_a": False}, {"b_referenced_a": True}) == "gap-closed"
    assert compare.classify({"b_referenced_a": False}, {"b_referenced_a": False}) == "unchanged"
    assert compare.classify({"b_referenced_a": True}, {"b_referenced_a": False}) == "regressed"
    assert compare.classify({"b_referenced_a": True}, {"b_referenced_a": True}) == "still-helping"
    assert compare.classify(None, {"b_referenced_a": True}) == "baseline-missing"
    assert compare.classify({"b_referenced_a": True}, None) == "v1-missing"
    assert compare.classify(None, None) == "both-missing"


# --------------------------------------------------------------------- compare


def test_compare_counts_gap_closed(baseline_dir, v1_dir):
    comp = compare.compare(
        "loader",
        baseline_dir=baseline_dir,
        with_memory_dir=v1_dir,
    )
    assert comp["n_total"] == 3
    assert comp["n_closed"] == 1
    labels = {r["scenario_id"]: r["label"] for r in comp["rows"]}
    assert labels["loader_01_snake_case"] == "gap-closed"
    assert labels["loader_02_no_phone_service_null"] == "unchanged"
    assert labels["loader_03_drop_pii"] == "still-helping"


# --------------------------------------------------------------------- render


def test_render_markdown_contains_table(baseline_dir, v1_dir):
    comp = compare.compare(
        "loader",
        baseline_dir=baseline_dir,
        with_memory_dir=v1_dir,
    )
    md = compare.render_markdown([comp])
    assert "# Baseline vs with-memory comparison" in md
    assert "Gap-closed:** 1/3" in md
    assert "loader_01_snake_case" in md
    assert "gap-closed" in md


def test_render_markdown_byte_stable(baseline_dir, v1_dir):
    comp = compare.compare(
        "loader",
        baseline_dir=baseline_dir,
        with_memory_dir=v1_dir,
    )
    a = compare.render_markdown([comp])
    b = compare.render_markdown([comp])
    assert a == b


# --------------------------------------------------------------------- main


def test_main_exit_0_when_gap_closed(tmp_path, baseline_dir, v1_dir, capsys):
    out = tmp_path / "COMPARE.md"
    rc = compare.main(
        [
            "--agent",
            "loader",
            "--baseline-root",
            str(baseline_dir.parent),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert out.exists()
    assert "gap-closed" in out.read_text()


def test_main_exit_1_when_no_gap_closed(tmp_path, capsys):
    # Both empty -> 0 closed -> exit 1.
    rb = tmp_path / "results" / "loader"
    rv = tmp_path / "results" / "loader__with_memory"
    rb.mkdir(parents=True)
    rv.mkdir(parents=True)
    (rb / "runs.jsonl").write_text("")
    (rv / "runs.jsonl").write_text("")
    out = tmp_path / "COMPARE.md"
    rc = compare.main(
        [
            "--agent",
            "loader",
            "--baseline-root",
            str(rb.parent),
            "--out",
            str(out),
        ]
    )
    assert rc == 1
