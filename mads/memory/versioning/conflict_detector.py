"""conflict_detector — v1.1 architectural scaffold.

Design intent: detects when two concurrent writes to the same identity tuple
would collide. Today SQLite's single-writer model gives us serializability
for free, but as the system grows toward multi-process eval runs or a
write-ahead-log style queue, we'll need to recognize the case where two
``upsert``s with overlapping ``valid_from`` windows arrive for the same
identity tuple and decide whether to merge, reject, or branch.

v1.1 trigger: when the eval harness or a downstream UI starts issuing
parallel writes — at which point the bare ``valid_to = now()`` close
becomes lossy.

Status: STUB — interface only, no behavior. See docs/architecture.md.
"""
from __future__ import annotations

from typing import Any


class ConflictDetector:
    """Detects concurrent-write conflicts on a versioned-category identity tuple."""

    def detect(
        self,
        identity: tuple[Any, ...],
        *,
        incoming: dict[str, Any],
        existing: list[dict[str, Any]],
    ) -> str:
        """Return one of ``"ok"``, ``"merge"``, ``"reject"``, ``"branch"``.

        Inputs: the identity tuple, the incoming payload, and the currently
        active row(s) for that identity. Output: a verdict the version
        manager will act on.
        """
        raise NotImplementedError(
            "conflict_detector is a v1.1 scaffold; not yet implemented."
        )


__all__ = ["ConflictDetector"]
