"""sqlite_store — v1.1 architectural scaffold.

Design intent: abstract base for the per-category SQLite stores. Today every
category re-implements the same boilerplate: ``__init__`` opens a connection,
runs the inline migration SQL, exposes ``conn``; ``_load_json``/``_dump_json``
helpers; ``valid_to`` close-and-insert for the versioned categories. A
``BaseSqliteStore`` lets each category file contain only its identity tuple,
its schema-specific columns, and its category-specific logic.

v1.1 trigger: when adding category 6 (workflow_recipe) or 7
(pipeline_decision) — two more categories of essentially the same SQLite
boilerplate is the point where extracting the base class pays off.

Status: STUB — interface only, no behavior. See docs/architecture.md.
"""
from __future__ import annotations

from typing import Any


class BaseSqliteStore:
    """Abstract base for per-category SQLite-backed stores."""

    def __init__(self, db_path: str | None = None) -> None:
        raise NotImplementedError(
            "sqlite_store is a v1.1 scaffold; not yet implemented."
        )

    def upsert(self, identity: tuple[Any, ...], payload: dict[str, Any]) -> str:
        """Close the active row for ``identity`` (if any) and insert ``payload``.

        Returns the new row id. The versioned categories (cats 1-4) will
        delegate their upsert through this method; append-only categories
        (cat 5) will use a separate :meth:`append` instead.
        """
        raise NotImplementedError(
            "sqlite_store is a v1.1 scaffold; not yet implemented."
        )


__all__ = ["BaseSqliteStore"]
