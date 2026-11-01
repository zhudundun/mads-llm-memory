"""SQLite-backed store for the `user_preference` memory category.

Scoping (per the v1.3 axis model — see :mod:`mads.memory.scope`):

* ``user_id`` is IDENTITY_REQUIRED (every retrieval filters by user).
* ``project_id`` is IDENTITY_OPTIONAL — NULL means a preference that is
  not scoped to any project; a non-NULL value means scoped to that
  specific project.
* ``dataset_fingerprint`` is IDENTITY_OPTIONAL (v1.3) — NULL means the
  preference applies to all datasets; a non-NULL value (a semantic
  fingerprint) scopes it to the specific dataset bound to the session.

The four scope combinations (global / project-only / dataset-only /
project+dataset) all coexist for the same (user, key) tuple.

Versioning rule: on
``set(user_id, key, value, project_id=..., dataset_fingerprint=...)``
any active row for the SAME identity tuple
``(user_id, project_id, dataset_fingerprint, preference_key)`` is closed
(``valid_to = now``) and a brand-new row is inserted. Rows for OTHER
scope combinations are untouched.

Rows are otherwise immutable; history is reconstructed by ordering rows
for a given identity tuple by ``valid_from`` ASC.

Retrieval — most-specific wins:

* ``get(user_id, key, project_id=..., dataset_fingerprint=...)`` — when
  multiple rows are eligible (e.g. a global pref and a project-scoped
  pref both exist for the same key), the most specific match wins
  (dataset > project > global). If only some axes are bound, the lookup
  returns the most-specific match among the eligible rows.
* ``list_active(user_id, project_id=..., dataset_fingerprint=...)`` —
  returns the effective view: one row per ``preference_key``, picking
  the most-specific match per key (same precedence as ``get``).
"""

from __future__ import annotations

import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Optional

# ----------------------------------------------------------------- migration

# Inline migration SQL kept in sync with ``infra/migrations/0006_axis_model.sql``
# (project_id column + new index) and ``0007_dataset_axis_for_preferences.sql``
# (dataset_fingerprint column + four-axis index).
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS user_preference_memory (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT,
    dataset_fingerprint TEXT,
    preference_key TEXT NOT NULL,
    preference_value TEXT NOT NULL,
    source TEXT,
    confidence REAL,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
"""

# Four-axis active-row partial unique index. NULL collapses to '' inside
# the index expression so the four scope variants coexist on the same
# (user, key) without colliding, but no two GLOBAL active rows are
# allowed (a versioning bug guard).
_INDEX_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS ix_active_user_pref
    ON user_preference_memory (
        user_id,
        COALESCE(project_id, ''),
        COALESCE(dataset_fingerprint, ''),
        preference_key
    )
    WHERE valid_to IS NULL;
"""

# Covering index for the retrieval filter pattern.
_LOOKUP_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS ix_pref_scope_lookup
    ON user_preference_memory (
        user_id, project_id, dataset_fingerprint, valid_to
    );
"""


_COLUMNS = (
    "id",
    "user_id",
    "project_id",
    "dataset_fingerprint",
    "preference_key",
    "preference_value",
    "source",
    "confidence",
    "valid_from",
    "valid_to",
    "created_at",
    "updated_at",
)


def _utcnow_iso() -> str:
    # Microsecond precision so back-to-back writes (e.g. set("A"), set("B")
    # on the same key) produce strictly increasing ``valid_from`` strings
    # — ``ORDER BY valid_from ASC`` then yields the correct historical
    # order even at sub-second cadence.
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {col: row[col] for col in _COLUMNS}


def _specificity_rank(row: dict) -> int:
    """Return a rank for the row's scope: dataset+project=3, dataset=2,
    project=1, global=0. Higher is more specific."""
    has_proj = row.get("project_id") is not None
    has_ds = row.get("dataset_fingerprint") is not None
    if has_proj and has_ds:
        return 3
    if has_ds:
        return 2
    if has_proj:
        return 1
    return 0


class PreferenceStore:
    """SQLite-backed store for per-user (optionally per-project,
    optionally per-dataset) preferences."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        *,
        conn: Optional[sqlite3.Connection] = None,
        lock: Optional[threading.RLock] = None,
    ) -> None:
        if conn is not None:
            self._owns_conn = False
            self._db_path = None
            self._conn = conn
            self._conn.row_factory = sqlite3.Row
        else:
            if db_path is None:
                db_path = os.environ.get("MADS_DB_PATH") or "./.mads/memory.sqlite"
            self._db_path = db_path
            self._owns_conn = True

            if db_path != ":memory:":
                parent = os.path.dirname(os.path.abspath(db_path))
                if parent:
                    os.makedirs(parent, exist_ok=True)

            self._conn = sqlite3.connect(
                db_path,
                isolation_level=None,  # autocommit; we manage txns explicitly
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON;")
            if db_path != ":memory:":
                self._conn.execute("PRAGMA journal_mode = WAL;")

        self._lock = lock if lock is not None else threading.RLock()

        self._init_schema()

    # ------------------------------------------------------------------ schema
    def _init_schema(self) -> None:
        """Idempotent schema init + axis-model migrations (0006 + 0007).

        Creates the table (fresh DBs get the new shape directly) and runs
        ALTER ... ADD COLUMN for older DBs that pre-date the project_id
        (v1.2) or dataset_fingerprint (v1.3) columns. Then drops any
        legacy index variants and recreates the four-axis active-row
        index plus the lookup covering index.
        """
        self._conn.executescript(_SCHEMA_SQL)

        # Detect older-schema DBs and ALTER in place.
        cur = self._conn.execute("PRAGMA table_info(user_preference_memory)")
        cols = {row[1] for row in cur.fetchall()}
        if "project_id" not in cols:
            self._conn.execute(
                "ALTER TABLE user_preference_memory ADD COLUMN project_id TEXT"
            )
        if "dataset_fingerprint" not in cols:
            self._conn.execute(
                "ALTER TABLE user_preference_memory "
                "ADD COLUMN dataset_fingerprint TEXT"
            )

        # Drop any prior index variant so the new four-axis one wins.
        self._conn.execute("DROP INDEX IF EXISTS ix_active_user_pref")
        self._conn.execute(_INDEX_SQL)
        self._conn.execute(_LOOKUP_INDEX_SQL)

    # ------------------------------------------------------------------ writes

    def set(
        self,
        user_id: str,
        key: str,
        value: str,
        *,
        project_id: Optional[str] = None,
        dataset_fingerprint: Optional[str] = None,
        source: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> str:
        """Set a preference, superseding any active row for the identity tuple.

        The identity tuple is
        ``(user_id, project_id, dataset_fingerprint, key)`` where any of
        ``project_id`` / ``dataset_fingerprint`` can be None ("not
        scoped on that axis"). Per IDENTITY_OPTIONAL semantics, the four
        scope variants coexist; this method only closes the row that
        matches the supplied scope tuple exactly.
        """
        if confidence is not None and not (0.0 <= float(confidence) <= 1.0):
            raise ValueError(
                f"confidence must be in [0, 1] if provided, got {confidence!r}"
            )

        new_id = uuid.uuid4().hex
        now = _utcnow_iso()

        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute("BEGIN IMMEDIATE;")
                cur.execute(
                    """
                    UPDATE user_preference_memory
                       SET valid_to = ?, updated_at = ?
                     WHERE user_id = ?
                       AND COALESCE(project_id, '') = COALESCE(?, '')
                       AND COALESCE(dataset_fingerprint, '') = COALESCE(?, '')
                       AND preference_key = ?
                       AND valid_to IS NULL
                    """,
                    (
                        now,
                        now,
                        user_id,
                        project_id,
                        dataset_fingerprint,
                        key,
                    ),
                )
                cur.execute(
                    """
                    INSERT INTO user_preference_memory (
                        id, user_id, project_id, dataset_fingerprint,
                        preference_key, preference_value,
                        source, confidence,
                        valid_from, valid_to, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
                    """,
                    (
                        new_id,
                        user_id,
                        project_id,
                        dataset_fingerprint,
                        key,
                        value,
                        source,
                        confidence,
                        now,
                        now,
                        now,
                    ),
                )
                cur.execute("COMMIT;")
            except Exception:
                try:
                    cur.execute("ROLLBACK;")
                except sqlite3.OperationalError:
                    pass
                raise
            finally:
                cur.close()

        return new_id

    def forget(
        self,
        user_id: str,
        key: str,
        *,
        project_id: Optional[str] = None,
        dataset_fingerprint: Optional[str] = None,
    ) -> bool:
        """Close the active row for the identity tuple. No replacement inserted.

        Closes only the row matching
        ``(user_id, project_id, dataset_fingerprint, key)`` exactly. The
        four scope variants coexist; ``forget`` operates on one of them
        at a time. Forgetting the global pref does NOT touch a
        project- or dataset-scoped row with the same key, and vice
        versa.
        """
        now = _utcnow_iso()
        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute("BEGIN IMMEDIATE;")
                cur.execute(
                    """
                    UPDATE user_preference_memory
                       SET valid_to = ?, updated_at = ?
                     WHERE user_id = ?
                       AND COALESCE(project_id, '') = COALESCE(?, '')
                       AND COALESCE(dataset_fingerprint, '') = COALESCE(?, '')
                       AND preference_key = ?
                       AND valid_to IS NULL
                    """,
                    (
                        now,
                        now,
                        user_id,
                        project_id,
                        dataset_fingerprint,
                        key,
                    ),
                )
                changed = cur.rowcount
                cur.execute("COMMIT;")
            except Exception:
                try:
                    cur.execute("ROLLBACK;")
                except sqlite3.OperationalError:
                    pass
                raise
            finally:
                cur.close()

        return bool(changed)

    # ------------------------------------------------------------------- reads

    def get(
        self,
        user_id: str,
        key: str,
        *,
        project_id: Optional[str] = None,
        dataset_fingerprint: Optional[str] = None,
    ) -> Optional[dict]:
        """Return the most-specific active row, or None.

        Most-specific wins: among the rows whose ``project_id`` is NULL
        or equals ``project_id`` AND whose ``dataset_fingerprint`` is
        NULL or equals ``dataset_fingerprint``, the row with the highest
        specificity rank wins (dataset+project > dataset > project >
        global). Within the same rank, ties are broken by the
        SQL ``LIMIT 1`` (deterministic in practice for one active row
        per scope tuple).

        Behaviour edge cases:

        * Both ``project_id`` and ``dataset_fingerprint`` are None →
          return ONLY the global row (matches v1.2 behaviour and
          prevents leaking project- or dataset-scoped rows into
          unscoped lookups).
        """
        with self._lock:
            if project_id is None and dataset_fingerprint is None:
                row = self._conn.execute(
                    """
                    SELECT * FROM user_preference_memory
                     WHERE user_id = ?
                       AND preference_key = ?
                       AND project_id IS NULL
                       AND dataset_fingerprint IS NULL
                       AND valid_to IS NULL
                     LIMIT 1
                    """,
                    (user_id, key),
                ).fetchone()
                return _row_to_dict(row) if row is not None else None

            # Fetch every eligible row, then rank by specificity in Python.
            rows = self._conn.execute(
                """
                SELECT * FROM user_preference_memory
                 WHERE user_id = ?
                   AND preference_key = ?
                   AND (project_id IS NULL OR project_id = ?)
                   AND (dataset_fingerprint IS NULL OR dataset_fingerprint = ?)
                   AND valid_to IS NULL
                """,
                (user_id, key, project_id, dataset_fingerprint),
            ).fetchall()

        if not rows:
            return None
        ranked = sorted(
            (_row_to_dict(r) for r in rows),
            key=_specificity_rank,
            reverse=True,
        )
        return ranked[0]

    def list_active(
        self,
        user_id: str,
        *,
        project_id: Optional[str] = None,
        dataset_fingerprint: Optional[str] = None,
    ) -> list[dict]:
        """Return the EFFECTIVE active view: one row per preference_key.

        For each ``preference_key`` that has at least one row matching
        the supplied scope (using IDENTITY_OPTIONAL semantics on each
        axis), return only the MOST-SPECIFIC eligible row (precedence:
        dataset+project > dataset > project > global). Ordered by
        ``preference_key`` ASC.

        Special case: when BOTH ``project_id`` and
        ``dataset_fingerprint`` are None, return ONLY rows with both
        axes NULL (= global only). This matches ``get`` and prevents
        unscoped sessions from leaking project- or dataset-scoped
        prefs.
        """
        with self._lock:
            if project_id is None and dataset_fingerprint is None:
                rows = self._conn.execute(
                    """
                    SELECT * FROM user_preference_memory
                     WHERE user_id = ?
                       AND project_id IS NULL
                       AND dataset_fingerprint IS NULL
                       AND valid_to IS NULL
                     ORDER BY preference_key ASC
                    """,
                    (user_id,),
                ).fetchall()
                return [_row_to_dict(r) for r in rows]

            rows = self._conn.execute(
                """
                SELECT * FROM user_preference_memory
                 WHERE user_id = ?
                   AND (project_id IS NULL OR project_id = ?)
                   AND (dataset_fingerprint IS NULL OR dataset_fingerprint = ?)
                   AND valid_to IS NULL
                 ORDER BY preference_key ASC
                """,
                (user_id, project_id, dataset_fingerprint),
            ).fetchall()

        # Collapse to most-specific per key.
        per_key: dict[str, dict] = {}
        for r in rows:
            d = _row_to_dict(r)
            k = d["preference_key"]
            prior = per_key.get(k)
            if prior is None or _specificity_rank(d) > _specificity_rank(prior):
                per_key[k] = d
        # Stable order by key.
        return [per_key[k] for k in sorted(per_key.keys())]

    def list_history(
        self,
        user_id: str,
        key: str,
        *,
        project_id: Optional[str] = None,
        dataset_fingerprint: Optional[str] = None,
    ) -> list[dict]:
        """Return all rows (active+closed) for the identity tuple, oldest first.

        The identity tuple is
        ``(user_id, project_id, dataset_fingerprint, key)``. Filtering
        is exact-match on each scope axis (NULL maps to ''), so the
        history chain you get back is for ONE specific scope variant,
        not the union across variants.
        """
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM user_preference_memory
                 WHERE user_id = ?
                   AND COALESCE(project_id, '') = COALESCE(?, '')
                   AND COALESCE(dataset_fingerprint, '') = COALESCE(?, '')
                   AND preference_key = ?
                 ORDER BY valid_from ASC
                """,
                (user_id, project_id, dataset_fingerprint, key),
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    # ---------------------------------------------------------------- lifecycle

    def close(self) -> None:
        if self._conn is not None and getattr(self, "_owns_conn", True):
            self._conn.close()
            self._conn = None  # type: ignore[assignment]
