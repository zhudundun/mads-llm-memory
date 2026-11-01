"""policies — v1.1 architectural scaffold.

Design intent: config-driven write/retention policies. Today the rules
("close prior active row on upsert", "embed only free-text on episode
write", "never embed PII") are hard-coded in each store. A ``Policies``
object lets ops toggle behavior per deployment (e.g. "redact PII before
embed in prod but skip the regex in dev for speed"; "retain only the latest
N versions of each metric definition"; "auto-decay episodes older than 1y").

v1.1 trigger: first time a downstream user asks for a retention knob, or the
first time PII scrubbing actually ships (the brief flags this as
aspirational).

Status: STUB — interface only, no behavior. See docs/architecture.md.
"""
from __future__ import annotations

from typing import Any


class Policies:
    """Config-driven memory policies (write rules, retention, redaction)."""

    def apply(self, category: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Apply registered policies to a write payload before it hits the store.

        Inputs: the category name (``"metric_definition"``, ...) and the raw
        payload dict the caller wants to write. Output: the (possibly
        redacted, possibly rejected) payload to actually persist.
        """
        raise NotImplementedError(
            "policies is a v1.1 scaffold; not yet implemented."
        )


__all__ = ["Policies"]
