"""SQLite-backed store for the `data_quality_issue` memory category.

Scoping (per the v1.2 axis model — see :mod:`mads.memory.scope`):

* ``dataset_fingerprint`` is IDENTITY_REQUIRED — issues belong to a
  specific dataset (identified by semantic_fingerprint). The same column
  on two physically-different datasets gets two independent issue
  records.
* ``user_id`` is AUDIT — recorded on every row but does NOT participate
  in identity or retrieval filtering. Multiple users sharing a dataset
  see the same issue library.
* ``project_id`` is AUDIT — recorded for accountability but does NOT
  participate in identity. The same dataset (by fingerprint) seen in
  two projects yields ONE shared issue library, because data-quality
  facts are properties of the data, not of the project organizing the
  analysis.

Identity tuple: ``(dataset_fingerprint, normalized_table,
normalized_column, normalized_issue_type)``. ``normalized_column`` is
NULLABLE — table-level issues (e.g. "exclude test users") have
``column_name = NULL``.

Versioning rule: any change to one of the *definitional* fields
(``issue_description``, ``recommended_handling``) closes the prior
active row (``valid_to = now()``) and inserts a brand-new row, both
inside one ``BEGIN IMMEDIATE`` transaction. Metadata changes
(``severity``, ``confidence``) update the active row in place. Identical
inputs are a no-op.

Resolution: ``resolve(...)`` sets ``valid_to = now`` on the active row
with NO replacement row. The issue exits the active list but remains in
``history``. Re-adding the same identity after a resolve starts a fresh
active row — it is NOT a continuation of the old chain.

Rows are immutable once closed; history is reconstructed by ordering
rows on ``(dataset_fingerprint, normalized_table, normalized_column,
normalized_issue_type, valid_from ASC)``.

Migration note (v1.2): pre-v1.2 stored issues by ``project_id`` not
``dataset_fingerprint``. The migration drops all existing rows (safe
because v1.1 had no production users) and reseats identity on the new
column. See ``infra/migrations/0006_axis_model.sql``.
"""

from __future__ import annotations

import os
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field

# ----------------------------------------------------------------- migration

# Inline migration SQL. The store handles fresh DBs (CREATE TABLE IF NOT
# EXISTS) and upgrades pre-axis-model DBs in :meth:`init_schema` by
# detecting the legacy schema shape via PRAGMA and rebuilding from
# scratch (drops existing rows — safe because v1.1 had no production
# users). The corresponding migration file is
# ``infra/migrations/0006_axis_model.sql``.
_INLINE_MIGRATION_SQL = """
CREATE TABLE IF NOT EXISTS data_quality_memory (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    dataset_fingerprint TEXT NOT NULL,
    table_name TEXT NOT NULL,
    column_name TEXT,
    issue_type TEXT NOT NULL,
    normalized_table TEXT NOT NULL,
    normalized_column TEXT,
    normalized_issue_type TEXT NOT NULL,
    issue_description TEXT NOT NULL,
    recommended_handling TEXT,
    severity TEXT NOT NULL DEFAULT 'warning'
        CHECK (severity IN ('info', 'warning', 'critical')),
    confidence REAL NOT NULL DEFAULT 1.0
        CHECK (confidence BETWEEN 0.0 AND 1.0),
    valid_from TIMESTAMP NOT NULL DEFAULT (CURRENT_TIMESTAMP),
    valid_to TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT (CURRENT_TIMESTAMP),
    updated_at TIMESTAMP NOT NULL DEFAULT (CURRENT_TIMESTAMP)
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_dq_active
    ON data_quality_memory(dataset_fingerprint, normalized_table,
                           COALESCE(normalized_column, ''),
                           normalized_issue_type)
    WHERE valid_to IS NULL;

CREATE INDEX IF NOT EXISTS idx_dq_history
    ON data_quality_memory(dataset_fingerprint, normalized_table,
                           COALESCE(normalized_column, ''),
                           normalized_issue_type, valid_from);

CREATE INDEX IF NOT EXISTS idx_dq_table_lookup
    ON data_quality_memory(dataset_fingerprint, normalized_table, valid_to);
"""


_COLUMNS = (
    "id",
    "user_id",
    "project_id",
    "dataset_fingerprint",
    "table_name",
    "column_name",
    "issue_type",
    "normalized_table",
    "normalized_column",
    "normalized_issue_type",
    "issue_description",
    "recommended_handling",
    "severity",
    "confidence",
    "valid_from",
    "valid_to",
    "created_at",
    "updated_at",
)

# Fields that, when changed, trigger a new version (close + insert).
_DEFINITIONAL_FIELDS = (
    "issue_description",
    "recommended_handling",
)

# Fields that, when changed alone, update the active row in place.
# Note: ``dataset_fingerprint`` is now identity, not metadata, so it
# leaves the metadata fields. Changing the fingerprint changes identity
# and produces a different row entirely.
_METADATA_FIELDS = ("severity", "confidence")

Severity = Literal["info", "warning", "critical"]
_VALID_SEVERITIES = ("info", "warning", "critical")

UpsertAction = Literal["inserted", "versioned", "updated_in_place", "noop"]


# --------------------------------------------------------------------- helpers


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_NORMALIZE_RE = re.compile(r"[^0-9a-z]+")


def _normalize_table(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError("table_name must be a non-empty string")
    cleaned = _NORMALIZE_RE.sub("_", name.lower()).strip("_")
    if not cleaned:
        raise ValueError("table_name must be non-empty after normalization")
    return cleaned


def _normalize_column(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    if not isinstance(name, str):
        raise ValueError("column_name must be a string or None")
    cleaned = _NORMALIZE_RE.sub("_", name.lower()).strip("_")
    if not cleaned:
        raise ValueError("column_name must be non-empty after normalization")
    return cleaned


def _normalize_issue_type(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError("issue_type must be a non-empty string")
    cleaned = _NORMALIZE_RE.sub("_", name.lower()).strip("_")
    if not cleaned:
        raise ValueError("issue_type must be non-empty after normalization")
    return cleaned


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {col: row[col] for col in _COLUMNS}


# --------------------------------------------------------------------- model


class DataQualityIssue(BaseModel):
    """Typed view of one row in ``data_quality_memory``.

    ``dataset_fingerprint`` is now REQUIRED (post-axis-model). It carries
    the dataset's semantic_fingerprint (sha256 of sorted columns +
    dtypes) so issues stay anchored to the logical dataset they were
    observed on.
    """

    id: str
    user_id: str
    project_id: str
    dataset_fingerprint: str
    table_name: str
    column_name: Optional[str] = None
    issue_type: str
    normalized_table: str
    normalized_column: Optional[str] = None
    normalized_issue_type: str
    issue_description: str
    recommended_handling: Optional[str] = None
    severity: Severity = "warning"
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    valid_from: str
    valid_to: Optional[str] = None
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row | dict) -> "DataQualityIssue":
        if isinstance(row, sqlite3.Row):
            data = _row_to_dict(row)
        else:
            data = dict(row)
        return cls.model_validate(data)


# --------------------------------------------------------------------- store


class DataQualityStore:
    """SQLite-backed store for dataset-scoped data-quality issues."""

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
                db_path = (
                    os.environ.get("MADS_DB_PATH") or "./.mads/memory.sqlite"
                )
            self._db_path = db_path
            self._owns_conn = True

            if db_path != ":memory:":
                parent = os.path.dirname(os.path.abspath(db_path))
                if parent:
                    os.makedirs(parent, exist_ok=True)

            self._conn = sqlite3.connect(
                db_path,
                isolation_level=None,
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON;")
            if db_path != ":memory:":
                self._conn.execute("PRAGMA journal_mode = WAL;")

        self._lock = lock if lock is not None else threading.RLock()
        self.init_schema()

    # ------------------------------------------------------------------ schema

    def init_schema(self) -> None:
        """Apply the migration. Idempotent and handles axis-model upgrades.

        For a fresh DB: creates the new-shape table directly.

        For a pre-axis-model DB (project-id-as-identity, nullable
        ``dataset_fingerprint``): drops existing rows, recreates the
        table with the new constraints, recreates the indexes. The drop
        is acceptable because v1.1 had no production users; eval DBs
        are per-scenario.
        """
        # Determine whether we need the rebuild path.
        cur = self._conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='data_quality_memory'"
        )
        table_exists = cur.fetchone() is not None

        if not table_exists:
            # Fresh DB → create the new shape directly.
            self._conn.executescript(_INLINE_MIGRATION_SQL)
            return

        # Existing table: check whether dataset_fingerprint is NOT NULL.
        info = self._conn.execute(
            "PRAGMA table_info(data_quality_memory)"
        ).fetchall()
        # PRAGMA table_info returns rows: (cid, name, type, notnull, dflt, pk)
        cols = {row[1]: row for row in info}
        dq_col = cols.get("dataset_fingerprint")
        needs_rebuild = dq_col is None or dq_col[3] == 0  # notnull == 0
        if not needs_rebuild:
            # Already on the new shape; just re-assert indexes (idempotent).
            self._conn.executescript(_INLINE_MIGRATION_SQL)
            return

        # Rebuild path: drop existing rows, drop indexes, rebuild table.
        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute("BEGIN IMMEDIATE;")
                cur.execute("DROP INDEX IF EXISTS uq_dq_active")
                cur.execute("DROP INDEX IF EXISTS idx_dq_history")
                cur.execute("DROP INDEX IF EXISTS idx_dq_table_lookup")
                cur.execute("DROP TABLE data_quality_memory")
                cur.execute("COMMIT;")
            except Exception:
                try:
                    cur.execute("ROLLBACK;")
                except sqlite3.OperationalError:
                    pass
                raise
            finally:
                cur.close()

        self._conn.executescript(_INLINE_MIGRATION_SQL)

    # ------------------------------------------------------------------ writes

    def upsert(
        self,
        *,
        user_id: str,
        project_id: str,
        dataset_fingerprint: str,
        table_name: str,
        column_name: Optional[str],
        issue_type: str,
        issue_description: str,
        recommended_handling: Optional[str] = None,
        severity: Severity = "warning",
        confidence: float = 1.0,
    ) -> tuple[DataQualityIssue, UpsertAction]:
        """Register or update a data-quality issue.

        Identity is now ``(dataset_fingerprint, normalized_table,
        normalized_column, normalized_issue_type)``. ``dataset_fingerprint``
        is required; it carries the dataset's semantic fingerprint.

        Returns ``(issue, action)`` where ``action`` is one of
        ``"inserted"``, ``"versioned"``, ``"updated_in_place"``, ``"noop"``.
        """
        # ----- input validation
        if not isinstance(user_id, str) or not user_id:
            raise ValueError("user_id must be a non-empty string")
        if not isinstance(project_id, str) or not project_id:
            raise ValueError("project_id must be a non-empty string")
        if (
            not isinstance(dataset_fingerprint, str)
            or not dataset_fingerprint
        ):
            raise ValueError(
                "dataset_fingerprint must be a non-empty string"
            )
        if not isinstance(table_name, str) or not table_name.strip():
            raise ValueError("table_name must be a non-empty string")
        if not isinstance(issue_type, str) or not issue_type.strip():
            raise ValueError("issue_type must be a non-empty string")
        if (
            not isinstance(issue_description, str)
            or not issue_description.strip()
        ):
            raise ValueError("issue_description must be a non-empty string")
        if severity not in _VALID_SEVERITIES:
            raise ValueError(
                f"severity must be one of {_VALID_SEVERITIES}, got {severity!r}"
            )
        if not (0.0 <= float(confidence) <= 1.0):
            raise ValueError(
                f"confidence must be in [0, 1], got {confidence!r}"
            )

        normalized_table = _normalize_table(table_name)
        normalized_column = _normalize_column(column_name)
        normalized_issue_type = _normalize_issue_type(issue_type)

        new_values = {
            "issue_description": issue_description,
            "recommended_handling": recommended_handling,
        }

        now = _utcnow_iso()

        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute("BEGIN IMMEDIATE;")
                prior_row = cur.execute(
                    """
                    SELECT * FROM data_quality_memory
                     WHERE dataset_fingerprint = ?
                       AND normalized_table = ?
                       AND COALESCE(normalized_column, '') = COALESCE(?, '')
                       AND normalized_issue_type = ?
                       AND valid_to IS NULL
                     LIMIT 1
                    """,
                    (
                        dataset_fingerprint,
                        normalized_table,
                        normalized_column,
                        normalized_issue_type,
                    ),
                ).fetchone()

                if prior_row is None:
                    new_id = uuid.uuid4().hex
                    cur.execute(
                        """
                        INSERT INTO data_quality_memory (
                            id, user_id, project_id, dataset_fingerprint,
                            table_name, column_name, issue_type,
                            normalized_table, normalized_column,
                            normalized_issue_type,
                            issue_description, recommended_handling,
                            severity, confidence,
                            valid_from, valid_to,
                            created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                  NULL, ?, ?)
                        """,
                        (
                            new_id,
                            user_id,
                            project_id,
                            dataset_fingerprint,
                            table_name,
                            column_name,
                            issue_type,
                            normalized_table,
                            normalized_column,
                            normalized_issue_type,
                            issue_description,
                            recommended_handling,
                            severity,
                            float(confidence),
                            now,
                            now,
                            now,
                        ),
                    )
                    cur.execute("COMMIT;")
                    target_id = new_id
                    action: UpsertAction = "inserted"

                else:
                    prior = _row_to_dict(prior_row)
                    defs_changed = any(
                        prior[k] != new_values[k] for k in _DEFINITIONAL_FIELDS
                    )

                    if not defs_changed:
                        meta_changed = (
                            prior["severity"] != severity
                            or float(prior["confidence"]) != float(confidence)
                        )
                        if not meta_changed:
                            cur.execute("COMMIT;")
                            return (
                                DataQualityIssue.from_row(prior_row),
                                "noop",
                            )

                        cur.execute(
                            """
                            UPDATE data_quality_memory
                               SET severity = ?,
                                   confidence = ?,
                                   updated_at = ?
                             WHERE id = ?
                            """,
                            (
                                severity,
                                float(confidence),
                                now,
                                prior["id"],
                            ),
                        )
                        cur.execute("COMMIT;")
                        target_id = prior["id"]
                        action = "updated_in_place"

                    else:
                        new_id = uuid.uuid4().hex
                        cur.execute(
                            """
                            UPDATE data_quality_memory
                               SET valid_to = ?, updated_at = ?
                             WHERE id = ?
                            """,
                            (now, now, prior["id"]),
                        )
                        cur.execute(
                            """
                            INSERT INTO data_quality_memory (
                                id, user_id, project_id, dataset_fingerprint,
                                table_name, column_name, issue_type,
                                normalized_table, normalized_column,
                                normalized_issue_type,
                                issue_description, recommended_handling,
                                severity, confidence,
                                valid_from, valid_to,
                                created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                      NULL, ?, ?)
                            """,
                            (
                                new_id,
                                user_id,
                                project_id,
                                dataset_fingerprint,
                                table_name,
                                column_name,
                                issue_type,
                                normalized_table,
                                normalized_column,
                                normalized_issue_type,
                                issue_description,
                                recommended_handling,
                                severity,
                                float(confidence),
                                now,
                                now,
                                now,
                            ),
                        )
                        cur.execute("COMMIT;")
                        target_id = new_id
                        action = "versioned"
            except Exception:
                try:
                    cur.execute("ROLLBACK;")
                except sqlite3.OperationalError:
                    pass
                raise
            finally:
                cur.close()

            out_row = self._conn.execute(
                "SELECT * FROM data_quality_memory WHERE id = ?", (target_id,)
            ).fetchone()
            return DataQualityIssue.from_row(out_row), action

    def resolve(
        self,
        *,
        user_id: str,
        project_id: str,
        dataset_fingerprint: str,
        table_name: str,
        column_name: Optional[str],
        issue_type: str,
    ) -> Optional[DataQualityIssue]:
        """Mark the active issue for the identity tuple as resolved."""
        if not isinstance(user_id, str) or not user_id:
            raise ValueError("user_id must be a non-empty string")
        if not isinstance(project_id, str) or not project_id:
            raise ValueError("project_id must be a non-empty string")
        if (
            not isinstance(dataset_fingerprint, str)
            or not dataset_fingerprint
        ):
            raise ValueError(
                "dataset_fingerprint must be a non-empty string"
            )

        try:
            normalized_table = _normalize_table(table_name)
            normalized_column = _normalize_column(column_name)
            normalized_issue_type = _normalize_issue_type(issue_type)
        except ValueError:
            return None

        now = _utcnow_iso()

        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute("BEGIN IMMEDIATE;")
                prior_row = cur.execute(
                    """
                    SELECT * FROM data_quality_memory
                     WHERE dataset_fingerprint = ?
                       AND normalized_table = ?
                       AND COALESCE(normalized_column, '') = COALESCE(?, '')
                       AND normalized_issue_type = ?
                       AND valid_to IS NULL
                     LIMIT 1
                    """,
                    (
                        dataset_fingerprint,
                        normalized_table,
                        normalized_column,
                        normalized_issue_type,
                    ),
                ).fetchone()

                if prior_row is None:
                    cur.execute("COMMIT;")
                    return None

                prior_id = prior_row["id"]
                cur.execute(
                    """
                    UPDATE data_quality_memory
                       SET valid_to = ?, updated_at = ?
                     WHERE id = ?
                    """,
                    (now, now, prior_id),
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

            closed = self._conn.execute(
                "SELECT * FROM data_quality_memory WHERE id = ?", (prior_id,)
            ).fetchone()
            return DataQualityIssue.from_row(closed)

    # ------------------------------------------------------------------- reads

    def get_active_for_column(
        self,
        *,
        dataset_fingerprint: str,
        table_name: str,
        column_name: str,
    ) -> list[DataQualityIssue]:
        """All currently-active issues anchored to ``(table_name, column_name)``
        for this dataset (by fingerprint)."""
        try:
            normalized_table = _normalize_table(table_name)
            normalized_column = _normalize_column(column_name)
        except ValueError:
            return []
        if normalized_column is None:
            return []
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM data_quality_memory
                 WHERE dataset_fingerprint = ?
                   AND normalized_table = ?
                   AND normalized_column = ?
                   AND valid_to IS NULL
                 ORDER BY normalized_issue_type ASC
                """,
                (dataset_fingerprint, normalized_table, normalized_column),
            ).fetchall()
        return [DataQualityIssue.from_row(r) for r in rows]

    def list_active_for_table(
        self, *, dataset_fingerprint: str, table_name: str
    ) -> list[DataQualityIssue]:
        """All active issues anchored to this (table, dataset)."""
        try:
            normalized_table = _normalize_table(table_name)
        except ValueError:
            return []
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM data_quality_memory
                 WHERE dataset_fingerprint = ?
                   AND normalized_table = ?
                   AND valid_to IS NULL
                 ORDER BY
                    CASE WHEN normalized_column IS NULL THEN 0 ELSE 1 END,
                    normalized_column ASC,
                    normalized_issue_type ASC
                """,
                (dataset_fingerprint, normalized_table),
            ).fetchall()
        return [DataQualityIssue.from_row(r) for r in rows]

    def search_active(
        self, *, dataset_fingerprint: str, query: str
    ) -> list[DataQualityIssue]:
        """Case-insensitive substring search over active issues for the
        dataset."""
        if not isinstance(query, str) or not query:
            return []
        like = f"%{query.lower()}%"
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM data_quality_memory
                 WHERE dataset_fingerprint = ?
                   AND valid_to IS NULL
                   AND (
                        LOWER(issue_description) LIKE ?
                     OR LOWER(COALESCE(recommended_handling, '')) LIKE ?
                     OR LOWER(table_name) LIKE ?
                     OR LOWER(COALESCE(column_name, '')) LIKE ?
                     OR LOWER(issue_type) LIKE ?
                   )
                 ORDER BY normalized_table ASC,
                          CASE WHEN normalized_column IS NULL THEN 0 ELSE 1 END,
                          normalized_column ASC,
                          normalized_issue_type ASC
                """,
                (dataset_fingerprint, like, like, like, like, like),
            ).fetchall()
        return [DataQualityIssue.from_row(r) for r in rows]

    def history(
        self,
        *,
        dataset_fingerprint: str,
        table_name: str,
        column_name: Optional[str],
        issue_type: str,
    ) -> list[DataQualityIssue]:
        """Full version history for the identity tuple, oldest first."""
        try:
            normalized_table = _normalize_table(table_name)
            normalized_column = _normalize_column(column_name)
            normalized_issue_type = _normalize_issue_type(issue_type)
        except ValueError:
            return []
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM data_quality_memory
                 WHERE dataset_fingerprint = ?
                   AND normalized_table = ?
                   AND COALESCE(normalized_column, '') = COALESCE(?, '')
                   AND normalized_issue_type = ?
                 ORDER BY valid_from ASC, created_at ASC
                """,
                (
                    dataset_fingerprint,
                    normalized_table,
                    normalized_column,
                    normalized_issue_type,
                ),
            ).fetchall()
        return [DataQualityIssue.from_row(r) for r in rows]

    # ---------------------------------------------------------------- lifecycle

    def close(self) -> None:
        if self._conn is not None and self._owns_conn:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]
