"""Agent-agnostic baseline harness for paired-session scenarios.

Public surface:
    run_scenario_pair(scenario, agent_factory, result_dir, timeout_s=90) -> dict

Design contract:
    * The harness imports NOTHING from ai_data_science_team. The caller wires
      that in via a closure (agent_factory) and an `invoke` callable on the
      scenario dispatcher side. See eval/agent_factories.py for the project's
      concrete ADS wiring.
    * Two sessions are run on TWO separate fresh agent instances (one
      agent_factory() call per session). This is the whole point of the
      cross-session-only design: a baseline agent has no memory bridge from A
      to B, so B's behavior should be independent of A.
    * Each session is wallclock-bounded by `timeout_s` (default 90 s). On
      timeout we save what we have and mark the session error="timeout".
    * Token + cost accounting: the per-agent dispatcher (eval/agent_factories.py)
      is responsible for returning prompt/completion/total tokens in its payload.
      It does this by attaching a provider-agnostic BaseCallbackHandler at LLM
      construction time and reading the totals after the agent finishes. The
      harness applies pricing constants (PRICE_INPUT_USD_PER_M /
      PRICE_OUTPUT_USD_PER_M) — currently Claude Sonnet 4.6 list pricing.

Per-session artifact:
    <result_dir>/<scenario_id>__session_<a|b>.json   (machine-readable)
    <result_dir>/<scenario_id>__session_<a|b>.md     (human-readable)
"""
from __future__ import annotations

import json
import threading
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# Claude Sonnet 4.6 list pricing (USD per 1M tokens).
# Kept as module-level constants so they're trivially auditable / overridable.
# If you swap models, also update ANTHROPIC_MODEL in eval/agent_factories.py.
PRICE_INPUT_USD_PER_M = 3.00
PRICE_OUTPUT_USD_PER_M = 15.00


def _compute_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens * PRICE_INPUT_USD_PER_M / 1_000_000.0
        + completion_tokens * PRICE_OUTPUT_USD_PER_M / 1_000_000.0
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# -----------------------------------------------------------------------------
# Session record
# -----------------------------------------------------------------------------
@dataclass
class SessionRecord:
    scenario_id: str
    session: str  # "a" or "b"
    agent_name: str
    prompt: str
    intent: str
    timestamp: str = field(default_factory=_now_iso)
    response: Optional[str] = None
    tool_calls: list = field(default_factory=list)  # list of {"name": str}
    tool_call_count: int = 0
    tool_call_names: list = field(default_factory=list)
    latency_ms: Optional[int] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    # viz-only extras (None for loader/eda)
    viz_code: Optional[str] = None
    error: Optional[str] = None
    error_traceback: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------------------------------------------------------------
# Timeout machinery
# -----------------------------------------------------------------------------
class _TimeoutBox:
    """Holds the result/exception from a worker thread."""

    def __init__(self) -> None:
        self.result: Any = None
        self.exc: Optional[BaseException] = None
        self.tb: Optional[str] = None


def _run_with_timeout(
    fn: Callable[[], Any], timeout_s: int
) -> tuple[Any, Optional[str], Optional[str], bool]:
    """
    Returns (result, exc_repr, traceback_str, timed_out).
    The worker thread is daemonic; if it times out we leave it running and
    return. The langgraph runtime has no co-op cancellation hook, so this is
    the cleanest we can do without subprocesses.
    """
    box = _TimeoutBox()

    def target() -> None:
        try:
            box.result = fn()
        except BaseException as e:  # noqa: BLE001 — capture everything
            box.exc = e
            box.tb = traceback.format_exc()

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        return None, None, None, True
    if box.exc is not None:
        return None, f"{type(box.exc).__name__}: {box.exc}", box.tb, False
    return box.result, None, None, False


# -----------------------------------------------------------------------------
# Invocation primitive
# -----------------------------------------------------------------------------
def _invoke_once(
    agent: Any,
    agent_name: str,
    prompt: str,
    invoke_fn: Callable[[str, Any], Dict[str, Any]],
    timeout_s: int,
) -> Dict[str, Any]:
    """
    Drives one session. `invoke_fn(prompt, agent)` is the per-agent
    dispatcher (provided by caller, lives in agent_factories.py) and must
    return a dict like:
        {
            "response": str,                  # the AI message
            "tool_calls": list[dict|str],     # raw tool-call records
            "viz_code": str | None,           # only set for viz
            "prompt_tokens": int,             # populated by dispatcher
            "completion_tokens": int,
            "total_tokens": int,
        }
    Token accounting is the dispatcher's responsibility (see _read_usage in
    eval/agent_factories.py).
    """
    out: Dict[str, Any] = {
        "response": None,
        "tool_calls": [],
        "viz_code": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    t0 = time.monotonic()

    def do_invoke() -> Dict[str, Any]:
        return invoke_fn(prompt, agent)

    result, exc_repr, tb_str, timed_out = _run_with_timeout(do_invoke, timeout_s)
    latency_ms = int((time.monotonic() - t0) * 1000)

    if timed_out:
        return {
            **out,
            "latency_ms": latency_ms,
            "error": "timeout",
            "error_traceback": None,
        }
    if exc_repr is not None:
        return {
            **out,
            "latency_ms": latency_ms,
            "error": exc_repr,
            "error_traceback": tb_str,
        }

    # Successful invocation
    payload = result or {}
    return {
        "response": payload.get("response"),
        "tool_calls": payload.get("tool_calls") or [],
        "viz_code": payload.get("viz_code"),
        "prompt_tokens": int(payload.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(payload.get("completion_tokens", 0) or 0),
        "total_tokens": int(payload.get("total_tokens", 0) or 0),
        "latency_ms": latency_ms,
        "error": None,
        "error_traceback": None,
    }


# -----------------------------------------------------------------------------
# Tool-call normalization
# -----------------------------------------------------------------------------
def _normalize_tool_calls(raw: Any) -> tuple[list, list, int]:
    """
    Returns (raw_records, names, count).
    ADS get_tool_calls() can return a list of strings, a list of dicts, or
    something dataframe-shaped. We coerce to a simple [{"name": ...}, ...].
    """
    records: list = []
    names: list = []

    if raw is None:
        return records, names, 0

    try:
        iterable = list(raw)
    except Exception:
        return [{"repr": repr(raw)}], [], 1

    for item in iterable:
        if isinstance(item, str):
            records.append({"name": item})
            names.append(item)
        elif isinstance(item, dict):
            fn_field = item.get("function")
            fn_name = fn_field.get("name") if isinstance(fn_field, dict) else None
            name = (
                item.get("name")
                or item.get("tool")
                or item.get("tool_name")
                or fn_name
            )
            records.append(item)
            names.append(str(name) if name else "<unknown>")
        else:
            r = repr(item)
            if len(r) > 500:
                r = r[:500] + "...<truncated>"
            records.append({"repr": r})
            # Best-effort name extraction
            n = getattr(item, "name", None) or getattr(item, "tool", None)
            names.append(str(n) if n else "<unknown>")

    return records, names, len(records)


# -----------------------------------------------------------------------------
# Markdown emitter
# -----------------------------------------------------------------------------
def _session_to_markdown(rec: SessionRecord) -> str:
    lines = [
        f"# {rec.scenario_id} — session {rec.session.upper()}",
        "",
        f"- **agent**: `{rec.agent_name}`",
        f"- **timestamp**: {rec.timestamp}",
        f"- **latency_ms**: {rec.latency_ms}",
        f"- **prompt_tokens**: {rec.prompt_tokens}",
        f"- **completion_tokens**: {rec.completion_tokens}",
        f"- **total_tokens**: {rec.total_tokens}",
        f"- **cost_usd**: ${rec.cost_usd:.6f}",
        f"- **tool_call_count**: {rec.tool_call_count}",
        f"- **tool_call_names**: {rec.tool_call_names}",
    ]
    if rec.error:
        lines += ["", f"- **ERROR**: `{rec.error}`"]
    lines += [
        "",
        "## Intent",
        "",
        rec.intent,
        "",
        "## Prompt",
        "",
        "```",
        rec.prompt,
        "```",
        "",
        "## Response",
        "",
    ]
    if rec.response is None:
        lines.append("_(no response)_")
    else:
        lines.append(rec.response)

    if rec.viz_code:
        lines += [
            "",
            "## Generated visualization code",
            "",
            "```python",
            rec.viz_code,
            "```",
        ]

    if rec.error_traceback:
        lines += [
            "",
            "## Traceback",
            "",
            "```",
            rec.error_traceback,
            "```",
        ]
    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------------
# Public entrypoint
# -----------------------------------------------------------------------------
def run_scenario_pair(
    scenario: Dict[str, Any],
    agent_factory: Callable[[], Any],
    result_dir: Path,
    timeout_s: int = 90,
    invoke_fn: Optional[Callable[[str, Any], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Run session A on a fresh agent, then session B on a SECOND fresh agent.
    Persists per-session JSON + MD into result_dir.

    Args:
        scenario: dict matching the scenarios.json schema.
        agent_factory: zero-arg callable producing a fresh agent for ONE session.
        result_dir: where to write artifacts (will be created).
        timeout_s: wallclock budget per session.
        invoke_fn: dispatcher (prompt, agent) -> payload dict. Required.
                   See eval/agent_factories.py::invoke_for_agent for the canonical impl.

    Returns:
        {"scenario_id", "agent", "a": {...}, "b": {...}, "ok": bool, "error": str|None}
    """
    if invoke_fn is None:
        raise ValueError(
            "run_scenario_pair: invoke_fn is required. Pass eval.agent_factories.invoke_for_agent "
            "(curried for the right agent name) or any (prompt, agent) -> payload callable."
        )

    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    scenario_id = scenario["id"]
    agent_name = scenario["agent"]

    def run_one(session_label: str) -> SessionRecord:
        sess = scenario[f"session_{session_label}"]
        rec = SessionRecord(
            scenario_id=scenario_id,
            session=session_label,
            agent_name=agent_name,
            prompt=sess["prompt"],
            intent=sess["intent"],
        )
        try:
            agent = agent_factory()
        except BaseException as e:  # factory failure
            rec.error = f"factory_error: {type(e).__name__}: {e}"
            rec.error_traceback = traceback.format_exc()
            _persist(rec, result_dir)
            return rec

        payload = _invoke_once(
            agent=agent,
            agent_name=agent_name,
            prompt=sess["prompt"],
            invoke_fn=invoke_fn,
            timeout_s=timeout_s,
        )

        rec.response = payload.get("response")
        rec.latency_ms = payload.get("latency_ms")
        rec.prompt_tokens = payload.get("prompt_tokens", 0)
        rec.completion_tokens = payload.get("completion_tokens", 0)
        rec.total_tokens = payload.get("total_tokens", 0)
        rec.cost_usd = round(
            _compute_cost_usd(rec.prompt_tokens, rec.completion_tokens), 6
        )
        rec.viz_code = payload.get("viz_code")
        rec.error = payload.get("error")
        rec.error_traceback = payload.get("error_traceback")

        records, names, count = _normalize_tool_calls(payload.get("tool_calls"))
        rec.tool_calls = records
        rec.tool_call_names = names
        rec.tool_call_count = count

        _persist(rec, result_dir)
        return rec

    rec_a = run_one("a")
    rec_b = run_one("b")

    ok = rec_a.error is None and rec_b.error is None
    combined_err = rec_a.error or rec_b.error
    return {
        "scenario_id": scenario_id,
        "agent": agent_name,
        "a": rec_a.to_dict(),
        "b": rec_b.to_dict(),
        "ok": ok,
        "error": combined_err,
    }


def _persist(rec: SessionRecord, result_dir: Path) -> None:
    base = result_dir / f"{rec.scenario_id}__session_{rec.session}"
    json_path = base.with_suffix(".json")
    md_path = base.with_suffix(".md")
    json_path.write_text(json.dumps(rec.to_dict(), indent=2, default=str))
    md_path.write_text(_session_to_markdown(rec))


__all__ = [
    "run_scenario_pair",
    "PRICE_INPUT_USD_PER_M",
    "PRICE_OUTPUT_USD_PER_M",
]
