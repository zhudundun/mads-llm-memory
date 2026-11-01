"""SQLite-backed store for the `dataset_schema` memory category.

Covers two related tables:

* ``dataset_memory`` — dataset-level facts (grain, PK, joins, time/target
  columns, known issues). Scoped by ``(user_id, dataset_fingerprint,
  table_name)``.
* ``column_memory`` — column-level facts (semantic type, business meaning,
  PII level, example values, null behavior). Scoped by ``(dataset_fingerprint,
  table_name, column_name)`` — *no* ``user_id``: column facts are a property
  of the data, not the user.

Versioning rule mirrors `user_preference`: on every register-call, any active
row for the identity tuple is closed (``valid_to = now()``) and a brand-new
row is inserted carrying the *merged* state (existing fields kept unless the
caller supplied a non-None override). Rows are immutable; history is
reconstructed by ordering rows on the identity tuple by ``valid_from ASC``.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

_DATASET_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS dataset_memory (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    project_id TEXT,
    dataset_fingerprint TEXT,
    dataset_name TEXT,
    table_name TEXT,
    grain TEXT,
    primary_key TEXT,
    join_keys TEXT,
    time_columns TEXT,
    target_columns TEXT,
    known_issues TEXT,
    exact_fingerprint TEXT,
    semantic_fingerprint TEXT,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
"""

_DATASET_CONTEXT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS dataset_context_memory (
    semantic_fingerprint TEXT PRIMARY KEY,
    dataset_name TEXT NOT NULL,
    exact_fingerprint TEXT,
    columns TEXT NOT NULL,
    dtypes TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    primary_key_candidates TEXT NOT NULL,
    join_key_candidates TEXT NOT NULL,
    time_columns TEXT NOT NULL,
    numeric_columns TEXT NOT NULL,
    categorical_columns TEXT NOT NULL,
    profile_summary TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT (CURRENT_TIMESTAMP),
    updated_at TIMESTAMP NOT NULL DEFAULT (CURRENT_TIMESTAMP)
);
"""

_COLUMN_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS column_memory (
    id TEXT PRIMARY KEY,
    dataset_fingerprint TEXT,
    table_name TEXT,
    column_name TEXT,
    semantic_type TEXT,
    business_meaning TEXT,
    pii_level TEXT,
    example_values TEXT,
    null_behavior TEXT,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
"""

_DATASET_INDEX_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS ix_active_dataset
    ON dataset_memory (user_id, dataset_fingerprint, table_name)
    WHERE valid_to IS NULL;
"""

_DATASET_SEMANTIC_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS ix_dataset_semantic_fp
    ON dataset_memory (user_id, semantic_fingerprint, table_name)
    WHERE valid_to IS NULL;
"""

_DATASET_EXACT_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS ix_dataset_exact_fp
    ON dataset_memory (user_id, exact_fingerprint)
    WHERE valid_to IS NULL;
"""

_COLUMN_INDEX_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS ix_active_column
    ON column_memory (dataset_fingerprint, table_name, column_name)
    WHERE valid_to IS NULL;
"""

_DATASET_COLUMNS = (
    "id",
    "user_id",
    "project_id",
    "dataset_fingerprint",
    "dataset_name",
    "table_name",
    "grain",
    "primary_key",
    "join_keys",
    "time_columns",
    "target_columns",
    "known_issues",
    "exact_fingerprint",
    "semantic_fingerprint",
    "valid_from",
    "valid_to",
    "created_at",
    "updated_at",
)

_DATASET_CONTEXT_COLUMNS = (
    "semantic_fingerprint",
    "dataset_name",
    "exact_fingerprint",
    "columns",
    "dtypes",
    "row_count",
    "primary_key_candidates",
    "join_key_candidates",
    "time_columns",
    "numeric_columns",
    "categorical_columns",
    "profile_summary",
    "created_at",
    "updated_at",
)

_DATASET_CONTEXT_JSON_COLS = {
    "columns",
    "dtypes",
    "primary_key_candidates",
    "join_key_candidates",
    "time_columns",
    "numeric_columns",
    "categorical_columns",
}

_COLUMN_COLUMNS = (
    "id",
    "dataset_fingerprint",
    "table_name",
    "column_name",
    "semantic_type",
    "business_meaning",
    "pii_level",
    "example_values",
    "null_behavior",
    "valid_from",
    "valid_to",
    "created_at",
    "updated_at",
)

# JSON-encoded columns per table
_DATASET_JSON_COLS = {
    "primary_key",
    "join_keys",
    "time_columns",
    "target_columns",
    "known_issues",
}
_COLUMN_JSON_COLS = {"example_values"}

_VALID_PII_LEVELS = {"none", "low", "medium", "high"}


# --------------------------------------------------------------------- helpers


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _dump_json(value: Any) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, default=str)


def _load_json(value: Optional[str]) -> Any:
    if value is None:
        return None
    return json.loads(value)


def _row_to_dataset_dict(row: sqlite3.Row) -> dict:
    d = {col: row[col] for col in _DATASET_COLUMNS}
    for col in _DATASET_JSON_COLS:
        d[col] = _load_json(d[col])
    return d


def _row_to_column_dict(row: sqlite3.Row) -> dict:
    d = {col: row[col] for col in _COLUMN_COLUMNS}
    for col in _COLUMN_JSON_COLS:
        d[col] = _load_json(d[col])
    return d


# --------------------------------------------------------------------- fingerprint
# ``compute_fingerprint`` moved to :mod:`mads.dataset.fingerprint` in v0.0.1.
# In v1.1 it became an alias for :func:`semantic_fingerprint`. Re-exported
# here so the legacy import path continues to work:
#     from mads.memory.dataset_schema import compute_fingerprint
from mads.dataset.fingerprint import (  # noqa: F401
    compute_fingerprint,
    exact_fingerprint as _exact_fingerprint,
    semantic_fingerprint as _semantic_fingerprint,
)
from mads.dataset.profiler import DatasetContext


def _row_to_context(row: sqlite3.Row) -> DatasetContext:
    d = {col: row[col] for col in _DATASET_CONTEXT_COLUMNS}
    for col in _DATASET_CONTEXT_JSON_COLS:
        d[col] = _load_json(d[col])
    # exact_fingerprint may be NULL in the row -> coerce to empty string
    # to match the DatasetContext schema (exact_fingerprint: str).
    if d.get("exact_fingerprint") is None:
        d["exact_fingerprint"] = ""
    return DatasetContext(
        dataset_name=d["dataset_name"],
        exact_fingerprint=d["exact_fingerprint"],
        semantic_fingerprint=d["semantic_fingerprint"],
        columns=d["columns"],
        dtypes=d["dtypes"],
        row_count=d["row_count"],
        primary_key_candidates=d["primary_key_candidates"],
        join_key_candidates=d["join_key_candidates"],
        time_columns=d["time_columns"],
        numeric_columns=d["numeric_columns"],
        categorical_columns=d["categorical_columns"],
        profile_summary=d["profile_summary"],
    )


# --------------------------------------------------------------------- store


class DatasetSchemaStore:
    """SQLite-backed store for dataset- and column-level memory with versioning."""

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

            # ``check_same_thread=False`` lets langgraph's ToolNode dispatch
            # store calls across threads; ``self._lock`` serialises
            # BEGIN/COMMIT to prevent nested-tx errors on the shared conn.
            self._conn = sqlite3.connect(
                db_path,
                isolation_level=None,  # autocommit; we manage txns explicitly
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON;")
            if db_path != ":memory:":
                self._conn.execute("PRAGMA journal_mode = WAL;")
        # Re-entrant: write paths internally call ``get_*`` for merging.
        # When sharing a connection (MemoryManager) the lock must ALSO be
        # shared across stores — otherwise a write on store A and a write
        # on store B can both hit BEGIN IMMEDIATE simultaneously on the
        # same conn. Callers that share a conn must share the lock.
        self._lock = lock if lock is not None else threading.RLock()
        self._conn.executescript(_DATASET_SCHEMA_SQL)
        self._conn.executescript(_COLUMN_SCHEMA_SQL)
        self._conn.executescript(_DATASET_CONTEXT_SCHEMA_SQL)
        self._conn.execute(_DATASET_INDEX_SQL)
        self._conn.execute(_DATASET_SEMANTIC_INDEX_SQL)
        self._conn.execute(_DATASET_EXACT_INDEX_SQL)
        self._conn.execute(_COLUMN_INDEX_SQL)

    # ============================================================== Dataset side

    def register_dataset(
        self,
        *,
        user_id: str,
        df: pd.DataFrame,
        dataset_name: str,
        table_name: str,
        project_id: Optional[str] = None,
        grain: Optional[str] = None,
        primary_key: Optional[list[str]] = None,
        join_keys: Optional[list[dict]] = None,
        time_columns: Optional[list[str]] = None,
        target_columns: Optional[list[str]] = None,
        known_issues: Optional[list[str]] = None,
        file_bytes: Optional[bytes] = None,
    ) -> tuple[str, str]:
        """Register or update a dataset row. Returns (new_id, semantic_fingerprint).

        If an active row exists for ``(user_id, fingerprint, table_name)``,
        it is closed and a new row is inserted carrying the *merged* state:
        caller-supplied non-None fields override; None means "keep prior".

        ``file_bytes`` (optional) populates the new ``exact_fingerprint``
        column. The ``semantic_fingerprint`` column is always populated and
        is identical to ``dataset_fingerprint`` (kept for backward compat).
        """
        sem_fp = _semantic_fingerprint(df)
        exact_fp = _exact_fingerprint(file_bytes) if file_bytes is not None else None
        # ``fingerprint`` is the legacy name; post-v1.1 it == semantic_fp.
        fingerprint = sem_fp

        # Merge with prior active row, if any.
        prior = self.get_dataset(
            user_id=user_id, fingerprint=fingerprint, table_name=table_name
        )

        def _pick(new, key):
            if new is not None:
                return new
            return prior.get(key) if prior else None

        merged = {
            "project_id": _pick(project_id, "project_id"),
            "dataset_name": _pick(dataset_name, "dataset_name"),
            "grain": _pick(grain, "grain"),
            "primary_key": _pick(primary_key, "primary_key"),
            "join_keys": _pick(join_keys, "join_keys"),
            "time_columns": _pick(time_columns, "time_columns"),
            "target_columns": _pick(target_columns, "target_columns"),
            "known_issues": _pick(known_issues, "known_issues"),
            # exact_fingerprint: prefer the caller-supplied bytes; else carry
            # over the prior row's value if one exists.
            "exact_fingerprint": (
                exact_fp if exact_fp is not None
                else (prior.get("exact_fingerprint") if prior else None)
            ),
        }

        new_id = uuid.uuid4().hex
        now = _utcnow_iso()

        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute("BEGIN IMMEDIATE;")
                cur.execute(
                    """
                    UPDATE dataset_memory
                       SET valid_to = ?, updated_at = ?
                     WHERE user_id = ?
                       AND dataset_fingerprint = ?
                       AND table_name = ?
                       AND valid_to IS NULL
                    """,
                    (now, now, user_id, fingerprint, table_name),
                )
                cur.execute(
                    """
                    INSERT INTO dataset_memory (
                        id, user_id, project_id, dataset_fingerprint, dataset_name,
                        table_name, grain, primary_key, join_keys, time_columns,
                        target_columns, known_issues,
                        exact_fingerprint, semantic_fingerprint,
                        valid_from, valid_to, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
                    """,
                    (
                        new_id,
                        user_id,
                        merged["project_id"],
                        fingerprint,
                        merged["dataset_name"],
                        table_name,
                        merged["grain"],
                        _dump_json(merged["primary_key"]),
                        _dump_json(merged["join_keys"]),
                        _dump_json(merged["time_columns"]),
                        _dump_json(merged["target_columns"]),
                        _dump_json(merged["known_issues"]),
                        merged["exact_fingerprint"],
                        sem_fp,
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

        return new_id, fingerprint

    def get_dataset(
        self, *, user_id: str, fingerprint: str, table_name: str
    ) -> Optional[dict]:
        """Return the active dataset row, or None."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT * FROM dataset_memory
                 WHERE user_id = ?
                   AND dataset_fingerprint = ?
                   AND table_name = ?
                   AND valid_to IS NULL
                 LIMIT 1
                """,
                (user_id, fingerprint, table_name),
            ).fetchone()
        return _row_to_dataset_dict(row) if row is not None else None

    def list_datasets(self, user_id: str) -> list[dict]:
        """Return all active datasets for `user_id`, sorted by (dataset_name, table_name)."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM dataset_memory
                 WHERE user_id = ?
                   AND valid_to IS NULL
                 ORDER BY dataset_name ASC, table_name ASC
                """,
                (user_id,),
            ).fetchall()
        return [_row_to_dataset_dict(r) for r in rows]

    def forget_dataset(
        self, *, user_id: str, fingerprint: str, table_name: str
    ) -> bool:
        """Close the active dataset row. No replacement inserted."""
        now = _utcnow_iso()
        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute("BEGIN IMMEDIATE;")
                cur.execute(
                    """
                    UPDATE dataset_memory
                       SET valid_to = ?, updated_at = ?
                     WHERE user_id = ?
                       AND dataset_fingerprint = ?
                       AND table_name = ?
                       AND valid_to IS NULL
                    """,
                    (now, now, user_id, fingerprint, table_name),
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

    def list_dataset_history(
        self, *, user_id: str, fingerprint: str, table_name: str
    ) -> list[dict]:
        """All rows (active+closed) for the dataset identity, oldest first."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM dataset_memory
                 WHERE user_id = ?
                   AND dataset_fingerprint = ?
                   AND table_name = ?
                 ORDER BY valid_from ASC
                """,
                (user_id, fingerprint, table_name),
            ).fetchall()
        return [_row_to_dataset_dict(r) for r in rows]

    # =============================================================== Column side

    def register_column(
        self,
        *,
        fingerprint: str,
        table_name: str,
        column_name: str,
        semantic_type: Optional[str] = None,
        business_meaning: Optional[str] = None,
        pii_level: Optional[str] = None,
        example_values: Optional[list] = None,
        null_behavior: Optional[str] = None,
    ) -> str:
        """Register or update a column row. Returns new row id.

        Merges with the prior active row for ``(fingerprint, table_name,
        column_name)``. ``pii_level`` if provided must be one of
        ``{'none','low','medium','high'}`` or ``ValueError`` is raised.
        """
        if pii_level is not None and pii_level not in _VALID_PII_LEVELS:
            raise ValueError(
                f"pii_level must be one of {_VALID_PII_LEVELS} if provided, "
                f"got {pii_level!r}"
            )

        prior = self.get_column(
            fingerprint=fingerprint,
            table_name=table_name,
            column_name=column_name,
        )

        def _pick(new, key):
            if new is not None:
                return new
            return prior.get(key) if prior else None

        merged = {
            "semantic_type": _pick(semantic_type, "semantic_type"),
            "business_meaning": _pick(business_meaning, "business_meaning"),
            "pii_level": _pick(pii_level, "pii_level"),
            "example_values": _pick(example_values, "example_values"),
            "null_behavior": _pick(null_behavior, "null_behavior"),
        }

        new_id = uuid.uuid4().hex
        now = _utcnow_iso()

        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute("BEGIN IMMEDIATE;")
                cur.execute(
                    """
                    UPDATE column_memory
                       SET valid_to = ?, updated_at = ?
                     WHERE dataset_fingerprint = ?
                       AND table_name = ?
                       AND column_name = ?
                       AND valid_to IS NULL
                    """,
                    (now, now, fingerprint, table_name, column_name),
                )
                cur.execute(
                    """
                    INSERT INTO column_memory (
                        id, dataset_fingerprint, table_name, column_name,
                        semantic_type, business_meaning, pii_level,
                        example_values, null_behavior,
                        valid_from, valid_to, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
                    """,
                    (
                        new_id,
                        fingerprint,
                        table_name,
                        column_name,
                        merged["semantic_type"],
                        merged["business_meaning"],
                        merged["pii_level"],
                        _dump_json(merged["example_values"]),
                        merged["null_behavior"],
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

    def get_column(
        self, *, fingerprint: str, table_name: str, column_name: str
    ) -> Optional[dict]:
        """Return the active column row, or None."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT * FROM column_memory
                 WHERE dataset_fingerprint = ?
                   AND table_name = ?
                   AND column_name = ?
                   AND valid_to IS NULL
                 LIMIT 1
                """,
                (fingerprint, table_name, column_name),
            ).fetchone()
        return _row_to_column_dict(row) if row is not None else None

    def list_columns(self, *, fingerprint: str, table_name: str) -> list[dict]:
        """All active column rows for the dataset table, ordered by column_name."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM column_memory
                 WHERE dataset_fingerprint = ?
                   AND table_name = ?
                   AND valid_to IS NULL
                 ORDER BY column_name ASC
                """,
                (fingerprint, table_name),
            ).fetchall()
        return [_row_to_column_dict(r) for r in rows]

    def forget_column(
        self, *, fingerprint: str, table_name: str, column_name: str
    ) -> bool:
        """Close the active column row. No replacement inserted."""
        now = _utcnow_iso()
        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute("BEGIN IMMEDIATE;")
                cur.execute(
                    """
                    UPDATE column_memory
                       SET valid_to = ?, updated_at = ?
                     WHERE dataset_fingerprint = ?
                       AND table_name = ?
                       AND column_name = ?
                       AND valid_to IS NULL
                    """,
                    (now, now, fingerprint, table_name, column_name),
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

    def list_column_history(
        self, *, fingerprint: str, table_name: str, column_name: str
    ) -> list[dict]:
        """All rows (active+closed) for the column identity, oldest first."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM column_memory
                 WHERE dataset_fingerprint = ?
                   AND table_name = ?
                   AND column_name = ?
                 ORDER BY valid_from ASC
                """,
                (fingerprint, table_name, column_name),
            ).fetchall()
        return [_row_to_column_dict(r) for r in rows]

    # =============================================================== Context side

    def register_context(self, ctx: DatasetContext) -> None:
        """Upsert a :class:`DatasetContext` row keyed on ``ctx.semantic_fingerprint``.

        If the row exists, every field is overwritten in place (no
        versioning — the context is a current-snapshot view of the logical
        dataset). If absent, a new row is inserted.
        """
        now = _utcnow_iso()
        with self._lock:
            existing = self._conn.execute(
                "SELECT 1 FROM dataset_context_memory WHERE semantic_fingerprint = ?",
                (ctx.semantic_fingerprint,),
            ).fetchone()

            exact_fp = ctx.exact_fingerprint if ctx.exact_fingerprint else None

            if existing is None:
                self._conn.execute(
                    """
                    INSERT INTO dataset_context_memory (
                        semantic_fingerprint, dataset_name, exact_fingerprint,
                        columns, dtypes, row_count,
                        primary_key_candidates, join_key_candidates,
                        time_columns, numeric_columns, categorical_columns,
                        profile_summary, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ctx.semantic_fingerprint,
                        ctx.dataset_name,
                        exact_fp,
                        _dump_json(ctx.columns),
                        _dump_json(ctx.dtypes),
                        ctx.row_count,
                        _dump_json(ctx.primary_key_candidates),
                        _dump_json(ctx.join_key_candidates),
                        _dump_json(ctx.time_columns),
                        _dump_json(ctx.numeric_columns),
                        _dump_json(ctx.categorical_columns),
                        ctx.profile_summary,
                        now,
                        now,
                    ),
                )
            else:
                self._conn.execute(
                    """
                    UPDATE dataset_context_memory
                       SET dataset_name = ?,
                           exact_fingerprint = ?,
                           columns = ?,
                           dtypes = ?,
                           row_count = ?,
                           primary_key_candidates = ?,
                           join_key_candidates = ?,
                           time_columns = ?,
                           numeric_columns = ?,
                           categorical_columns = ?,
                           profile_summary = ?,
                           updated_at = ?
                     WHERE semantic_fingerprint = ?
                    """,
                    (
                        ctx.dataset_name,
                        exact_fp,
                        _dump_json(ctx.columns),
                        _dump_json(ctx.dtypes),
                        ctx.row_count,
                        _dump_json(ctx.primary_key_candidates),
                        _dump_json(ctx.join_key_candidates),
                        _dump_json(ctx.time_columns),
                        _dump_json(ctx.numeric_columns),
                        _dump_json(ctx.categorical_columns),
                        ctx.profile_summary,
                        now,
                        ctx.semantic_fingerprint,
                    ),
                )

    def get_context(self, semantic_fingerprint: str) -> Optional[DatasetContext]:
        """Fetch a :class:`DatasetContext` by ``semantic_fingerprint``, or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM dataset_context_memory WHERE semantic_fingerprint = ?",
                (semantic_fingerprint,),
            ).fetchone()
        if row is None:
            return None
        return _row_to_context(row)

    def lookup_dataset(
        self,
        *,
        user_id: str,
        df: pd.DataFrame,
        table_name: str,
        file_bytes: Optional[bytes] = None,
    ) -> Optional[dict]:
        """Cross-session lookup: exact-fingerprint first, then semantic fallback.

        Returns ``{"matched_by": "exact"|"semantic", "row": <dataset_memory row>}``
        or ``None`` if neither match. The semantic fallback uses the
        ``semantic_fingerprint`` column (populated for all v1.1+ rows). Rows
        registered before v1.1 will have ``semantic_fingerprint IS NULL``
        and will not match the semantic fallback — which is fine because
        their ``dataset_fingerprint`` already equals the semantic_fp post-
        alias and the caller can call ``get_dataset`` directly.
        """
        sem_fp = _semantic_fingerprint(df)

        with self._lock:
            if file_bytes is not None:
                exact_fp = _exact_fingerprint(file_bytes)
                row = self._conn.execute(
                    """
                    SELECT * FROM dataset_memory
                     WHERE user_id = ?
                       AND exact_fingerprint = ?
                       AND table_name = ?
                       AND valid_to IS NULL
                     LIMIT 1
                    """,
                    (user_id, exact_fp, table_name),
                ).fetchone()
                if row is not None:
                    return {"matched_by": "exact", "row": _row_to_dataset_dict(row)}

            row = self._conn.execute(
                """
                SELECT * FROM dataset_memory
                 WHERE user_id = ?
                   AND semantic_fingerprint = ?
                   AND table_name = ?
                   AND valid_to IS NULL
                 LIMIT 1
                """,
                (user_id, sem_fp, table_name),
            ).fetchone()
            if row is not None:
                return {"matched_by": "semantic", "row": _row_to_dataset_dict(row)}

        return None

    # ---------------------------------------------------------------- lifecycle

    def close(self) -> None:
        if self._conn is not None and getattr(self, "_owns_conn", True):
            self._conn.close()
            self._conn = None  # type: ignore[assignment]
