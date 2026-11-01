"""Unit tests for eval.judge — JSON parsing + retry behavior.

We never call Anthropic in these tests; the judge accepts a stub ``llm=``
that returns canned text per call.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from judge import agreement_with_rubric, judge_pair  # noqa: E402


class _StubLLM:
    """Stub langchain ChatModel: returns the next canned response on .invoke()."""

    def __init__(self, *responses: str) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def invoke(self, prompt: str):
        self.calls.append(prompt)
        if not self._responses:
            raise AssertionError("StubLLM ran out of canned responses")
        return self._responses.pop(0)


def _kwargs():
    return dict(
        session_a_prompt="A asked X",
        session_b_prompt="B asked Y",
        baseline_response="baseline reply",
        mem_aware_response="memory reply",
    )


def test_judge_parses_clean_json_first_try():
    llm = _StubLLM(
        '{"winner": "memory-aware", "baseline_score": 2, "mem_aware_score": 4, "reason": "honors A"}'
    )
    out = judge_pair(**_kwargs(), llm=llm)
    assert out["winner"] == "memory-aware"
    assert out["baseline_score"] == 2
    assert out["mem_aware_score"] == 4
    assert out["reason"] == "honors A"
    assert len(llm.calls) == 1


def test_judge_strips_markdown_fence():
    llm = _StubLLM(
        '```json\n{"winner": "tie", "baseline_score": 3, "mem_aware_score": 3, "reason": "equal"}\n```'
    )
    out = judge_pair(**_kwargs(), llm=llm)
    assert out["winner"] == "tie"
    assert out["baseline_score"] == 3


def test_judge_retries_once_then_succeeds():
    llm = _StubLLM(
        "not json at all",
        '{"winner": "baseline", "baseline_score": 4, "mem_aware_score": 2, "reason": "baseline cleaner"}',
    )
    out = judge_pair(**_kwargs(), llm=llm)
    assert out["winner"] == "baseline"
    assert len(llm.calls) == 2


def test_judge_persistent_failure_records_raw_attempts():
    llm = _StubLLM("not json", "also not json")
    out = judge_pair(**_kwargs(), llm=llm)
    assert out["winner"] == "parse_error"
    assert out["baseline_score"] is None
    assert out["mem_aware_score"] is None
    assert "raw_attempts" in out
    assert out["raw_attempts"] == ["not json", "also not json"]


def test_judge_rejects_out_of_range_scores():
    # 6 is invalid (>5); judge falls through to parse_error after 2 attempts.
    llm = _StubLLM(
        '{"winner": "memory-aware", "baseline_score": 6, "mem_aware_score": 4, "reason": "bad"}',
        '{"winner": "memory-aware", "baseline_score": 0, "mem_aware_score": 4, "reason": "bad"}',
    )
    out = judge_pair(**_kwargs(), llm=llm)
    assert out["winner"] == "parse_error"


def test_agreement_with_rubric_truth_table():
    # Pass + winner=memory-aware -> agree
    assert agreement_with_rubric(gt_passed=True, judge_winner="memory-aware") is True
    # Pass + winner=baseline -> DISAGREE
    assert agreement_with_rubric(gt_passed=True, judge_winner="baseline") is False
    # Pass + winner=tie -> DISAGREE
    assert agreement_with_rubric(gt_passed=True, judge_winner="tie") is False
    # Fail + winner=memory-aware -> DISAGREE
    assert agreement_with_rubric(gt_passed=False, judge_winner="memory-aware") is False
    # Fail + winner=baseline -> agree
    assert agreement_with_rubric(gt_passed=False, judge_winner="baseline") is True
    # Fail + winner=tie -> agree
    assert agreement_with_rubric(gt_passed=False, judge_winner="tie") is True
