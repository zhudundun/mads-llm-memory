"""version_manager — v1.1 architectural scaffold.

Design intent: centralizes the close-and-insert logic that currently lives
in each cat 1-4 store's ``upsert`` method. The pattern is uniform — find
active rows for the identity tuple, set their ``valid_to = now()``, insert
the new row with ``valid_from = now()`` and ``valid_to = NULL``, merge
unchanged fields from the closed row into the new row — but it's duplicated
across four files today.

v1.1 trigger: lands with :class:`BaseSqliteStore` (same observed need:
factor out boilerplate when a 6th category arrives).

Status: STUB — interface only, no behavior. See docs/architecture.md.
"""
from __future__ import annotations

from typing import Any


class VersionManager:
    """Centralizes the close-and-insert versioning pattern for cats 1-4."""

    def close_and_insert(
        self,
        store: Any,
        identity: tuple[Any, ...],
        payload: dict[str, Any],
    ) -> str:
        """Close active rows for ``identity`` and insert ``payload``; return new id.

        Inputs: the concrete store (for its connection + table name), the
        identity tuple to close on, and the new payload. Output: the id of
        the inserted row. Handles the merge of unchanged fields from the
        closed row into the new row so callers can pass partial payloads.
        """
        raise NotImplementedError(
            "version_manager is a v1.1 scaffold; not yet implemented."
        )


__all__ = ["VersionManager"]
