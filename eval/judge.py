"""Single-LLM judge for paired (baseline vs memory-aware) responses.

Calls Anthropic Claude Sonnet 4.6 (same model the agent uses — no
model-mixing) with a strict JSON-output prompt. Parses with ``json.loads``,
retries once on parse failure, records raw output on persistent failure.

Public surface:

    judge_pair(*, session_a_prompt, session_b_prompt,
               baseline_response, mem_aware_response,
               model="claude-sonnet-4-6",
               llm=None) -> dict

The returned dict shape:

    {
        "winner": "baseline" | "memory-aware" | "tie" | "parse_error",
        "baseline_score": int (1..5) or None,
        "mem_aware_score": int (1..5) or None,
        "reason": str,
        "raw_attempts": list[str]  # only set on persistent parse failure
    }

The ``llm`` parameter is optional dependency-injection — production callers
leave it None and we build a fresh ChatAnthropic on each call; tests pass a
stub. We don't cache the LLM at module level because (a) one call per
scenario, not per token; and (b) it lets the harness pick up env changes
between scenarios.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

JUDGE_MODEL: str = "claude-sonnet-4-6"

_JUDGE_PROMPT_TEMPLATE = """You are an evaluator comparing two AI assistant responses.

Session A taught the assistant: {session_a_prompt}

Session B asked (new session, no in-context memory of A): {session_b_prompt}

Baseline response (no memory layer): {baseline_response}

Memory-aware response: {mem_aware_response}

Output ONLY valid JSON:
{{
  "winner": "baseline" | "memory-aware" | "tie",
  "baseline_score": 1-5,
  "mem_aware_score": 1-5,
  "reason": "one sentence on why"
}}

Judge rubric:
- Did the response honor A's stated rule/preference/correction?
- Is it factually consistent with the underlying dataset?
- Is the structure/format what A asked for?"""


def _build_prompt(
    *,
    session_a_prompt: str,
    session_b_prompt: str,
    baseline_response: str,
    mem_aware_response: str,
) -> str:
    return _JUDGE_PROMPT_TEMPLATE.format(
        session_a_prompt=session_a_prompt,
        session_b_prompt=session_b_prompt,
        baseline_response=baseline_response,
        mem_aware_response=mem_aware_response,
    )


def _build_llm(model: str):
    """Construct a fresh ChatAnthropic for one judge call. Late import so the
    judge module is testable without langchain installed."""
    from langchain_anthropic import ChatAnthropic  # noqa: PLC0415

    return ChatAnthropic(model=model, temperature=0)


def _extract_text(raw: Any) -> str:
    """Pull a text payload off either a langchain AIMessage or a plain str."""
    if isinstance(raw, str):
        return raw
    content = getattr(raw, "content", raw)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # langchain may return content as a list of dicts with 'text' keys.
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _try_parse_judge_json(text: str) -> Optional[dict]:
    """Try to parse a judge JSON payload out of ``text``.

    Returns the validated dict (with int scores) on success, None on failure.
    The judge sometimes wraps JSON in markdown fences (```json ...```), so we
    first try strict parse, then a fallback "first {...} blob" extraction.
    """
    candidates: list[str] = [text.strip()]
    # Strip markdown fences if present.
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        candidates.append(fence_match.group(1))
    # First {...} blob anywhere.
    blob_match = _JSON_BLOCK_RE.search(text)
    if blob_match:
        candidates.append(blob_match.group(0))

    for cand in candidates:
        try:
            parsed = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(parsed, dict):
            continue
        # Validate shape.
        winner = parsed.get("winner")
        if winner not in ("baseline", "memory-aware", "tie"):
            continue
        try:
            bs = int(parsed.get("baseline_score"))
            ms = int(parsed.get("mem_aware_score"))
        except (TypeError, ValueError):
            continue
        if not (1 <= bs <= 5 and 1 <= ms <= 5):
            continue
        reason = str(parsed.get("reason") or "")
        return {
            "winner": winner,
            "baseline_score": bs,
            "mem_aware_score": ms,
            "reason": reason,
        }
    return None


def judge_pair(
    *,
    session_a_prompt: str,
    session_b_prompt: str,
    baseline_response: str,
    mem_aware_response: str,
    model: str = JUDGE_MODEL,
    llm: Any = None,
) -> dict:
    """Score a (baseline, memory-aware) pair with the judge LLM.

    One Anthropic call per scenario. Retries once on JSON parse failure;
    on persistent failure returns ``winner='parse_error'`` and includes
    the raw attempt(s) under ``raw_attempts``.

    Pass ``llm`` to inject a stub (tests do this); production leaves it None
    and a fresh ChatAnthropic is built per call.
    """
    prompt = _build_prompt(
        session_a_prompt=session_a_prompt,
        session_b_prompt=session_b_prompt,
        baseline_response=baseline_response,
        mem_aware_response=mem_aware_response,
    )
    if llm is None:
        llm = _build_llm(model)

    raw_attempts: list[str] = []
    for attempt in range(2):
        raw = llm.invoke(prompt)
        text = _extract_text(raw)
        raw_attempts.append(text)
        parsed = _try_parse_judge_json(text)
        if parsed is not None:
            return parsed

    return {
        "winner": "parse_error",
        "baseline_score": None,
        "mem_aware_score": None,
        "reason": "judge produced no parseable JSON after 2 attempts",
        "raw_attempts": raw_attempts,
    }


def agreement_with_rubric(*, gt_passed: bool, judge_winner: str) -> bool:
    """Define the rubric-vs-judge "agree" predicate per the spec.

    agree = (gt_passed implies judge_winner == "memory-aware") AND
            (not gt_passed implies judge_winner != "memory-aware")
    """
    if gt_passed:
        return judge_winner == "memory-aware"
    return judge_winner != "memory-aware"


__all__ = [
    "JUDGE_MODEL",
    "judge_pair",
    "agreement_with_rubric",
]
