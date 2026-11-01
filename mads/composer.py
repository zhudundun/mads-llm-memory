"""Pure-templating final-answer composer.

Takes a :class:`mads.agents.types.TurnArtifacts` and emits a CLI-ready string.
No LLM call. See V1_DESIGN_BRIEF §6.
"""

from __future__ import annotations

from typing import Any

from mads.agents.types import TurnArtifacts


def compose(turn: TurnArtifacts) -> str:
    """Format the wrapped agent's output + memory citations into a string."""
    parts: list[str] = []
    parts.append(f"# Answer ({turn.agent_name})")
    parts.append("")
    parts.append(turn.agent_result.response or "_(no response)_")

    if turn.agent_result.viz_code:
        parts.append("")
        parts.append("## Generated code")
        parts.append("```python")
        parts.append(turn.agent_result.viz_code)
        parts.append("```")

    # NEW: user-facing provenance — the actual memories proactively injected
    # into the prompt (from the proactive wrapper's hybrid retrieve). This
    # is intentionally separate from "Memory consulted" (which records the
    # LLM's *tool calls*); these are memories surfaced WITHOUT the LLM
    # asking. Only emitted when the proactive wrapper actually injected
    # something — legacy reactive wrappers leave ``memories_injected``
    # empty, so the header simply doesn't appear.
    injected = getattr(turn, "memories_injected", None) or []
    if injected:
        parts.append("")
        parts.append(f"## Memories used ({len(injected)})")
        for i, rm in enumerate(injected, start=1):
            parts.append(f"{i}. {_memory_line(rm)}")

    if turn.memory_reads:
        parts.append("")
        parts.append(f"## Memory consulted ({len(turn.memory_reads)})")
        for rec in turn.memory_reads:
            parts.append(f"- {rec.get('name', '?')}({_short_args(rec)})")

    if turn.memory_writes:
        parts.append("")
        parts.append(f"## Memory written ({len(turn.memory_writes)})")
        for rec in turn.memory_writes:
            parts.append(f"- {rec.get('name', '?')}({_short_args(rec)})")

    if getattr(turn, "warnings", None):
        parts.append("")
        parts.append(f"## Warnings ({len(turn.warnings)})")
        for w in turn.warnings:
            parts.append(f"- {w}")

    parts.append("")
    r = turn.agent_result
    parts.append(
        f"_tokens: {r.total_tokens} "
        f"(in={r.prompt_tokens}, out={r.completion_tokens})_"
    )
    return "\n".join(parts)


# --------------------------------------------------------------------- helpers
# (memory-line formatting)

# Maximum length for a single memory summary's content snippet. Picked at
# 60 chars so a full line (counting "{N}. {Category}Memory: ") sits well
# under ~100 chars in a typical terminal.
_SUMMARY_WIDTH = 60


def _memory_line(rm: Any) -> str:
    """Return ``"{Category}Memory: {summary}"`` for one RetrievedMemory."""
    category = getattr(rm, "category", "?") or "?"
    label = category.capitalize() if isinstance(category, str) else "?"
    summary = _summary_for(rm)
    return f"{label}Memory: {summary}"


def _summary_for(rm: Any) -> str:
    """Dispatch to a per-category one-line summary.

    Each branch reads the underlying ``rm.memory`` dict (the source row
    coerced to plain dict by the retrievers) and renders a single line
    suitable for display in the composer's ``## Memories used`` section.
    """
    cat = getattr(rm, "category", None)
    mem = getattr(rm, "memory", None) or {}
    if cat == "metric":
        return _metric_summary(mem)
    if cat == "dataset":
        return _dataset_summary(mem)
    if cat == "episode":
        return _episode_summary(mem)
    if cat == "preference":
        return _preference_summary(mem)
    return "(unknown memory category)"


def _metric_summary(mem: dict) -> str:
    """``{metric_name} = {business_definition}`` (def truncated to 60ch).

    Falls back to ``sql_definition`` if ``business_definition`` is empty.
    """
    name = mem.get("metric_name") or mem.get("normalized_name") or "?"
    definition = mem.get("business_definition") or mem.get("sql_definition") or ""
    if not definition:
        return f"{name}"
    return f"{name} = {_truncate(str(definition), _SUMMARY_WIDTH)}"


def _dataset_summary(mem: dict) -> str:
    """``{table_name} grain = {grain}`` if grain is set, else
    ``{table_name or dataset_name}`` with any high-signal fallback.

    The ``dataset_memory`` row does not carry row/column counts (those
    live in the separate ``dataset_context_memory`` table and aren't on
    this row), so we fall back to whichever name/grain fields exist.
    """
    name = mem.get("table_name") or mem.get("dataset_name") or "?"
    grain = mem.get("grain")
    if grain:
        return f"{name} grain = {_truncate(str(grain), _SUMMARY_WIDTH)}"
    return str(name)


def _preference_summary(mem: dict) -> str:
    """``{preference_key} = {preference_value}`` (value truncated to 60ch).

    Falls back to ``"?"`` when the key is missing (defensive — every
    PreferenceStore row carries one).
    """
    key = mem.get("preference_key") or "?"
    value = mem.get("preference_value") or ""
    if not value:
        return f"{key}"
    return f"{key} = {_truncate(str(value), _SUMMARY_WIDTH)}"


def _episode_summary(mem: dict) -> str:
    """``"{question}" ({YYYY-MM-DD})`` — question truncated to 60 chars,
    ``created_at`` parsed to ISO date.
    """
    question = mem.get("question") or ""
    created_at = mem.get("created_at") or ""
    date = _date_from_timestamp(created_at)
    q = _truncate(str(question), _SUMMARY_WIDTH)
    if date:
        return f'"{q}" ({date})'
    return f'"{q}"'


def _truncate(s: str, width: int) -> str:
    """Truncate ``s`` to ``width`` chars (inclusive of an ellipsis if cut)."""
    if len(s) <= width:
        return s
    return s[: max(0, width - 1)] + "…"


def _date_from_timestamp(ts: str) -> str:
    """Extract a ``YYYY-MM-DD`` date from a stored timestamp string.

    Accepts both ISO-8601 (``2026-05-20T10:33:14+00:00``) and SQLite's
    legacy ``YYYY-MM-DD HH:MM:SS`` form. Returns ``""`` on unrecognized
    input rather than raising — display code must never crash.
    """
    if not isinstance(ts, str) or not ts:
        return ""
    # Both forms start with a 10-char ``YYYY-MM-DD`` if well-formed.
    head = ts[:10]
    if len(head) == 10 and head[4] == "-" and head[7] == "-":
        return head
    return ""


def _short_args(rec: dict, *, width: int = 80) -> str:
    """Render tool-call args as a one-liner, truncated.

    ``rec`` is expected to be ``{"name": ..., "args": dict}``. Returns the
    args formatted as ``k=v, k=v, ...``; if the result exceeds ``width``
    characters it is truncated with an ellipsis.
    """
    args = rec.get("args") if isinstance(rec, dict) else None
    if not args:
        return ""
    if not isinstance(args, dict):
        s = str(args)
    else:
        s = ", ".join(f"{k}={_short(v)}" for k, v in args.items())
    if len(s) > width:
        return s[: width - 1] + "…"
    return s


def _short(v: Any, *, max_len: int = 40) -> str:
    s = repr(v) if not isinstance(v, str) else v
    if len(s) > max_len:
        return s[: max_len - 1] + "…"
    return s


__all__ = ["compose"]

