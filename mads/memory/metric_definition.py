"""SQLite-backed store for the `metric_definition` memory category.

Identity tuple: ``(project_id, normalized_name)``. Metrics are a property of
the project, not the user — multiple users in the same project share one
metric library. ``user_id`` is recorded on every row (NOT NULL) so we know
who created or last edited a version, but does NOT participate in identity.

Versioning rule: any change to one of the *definitional* fields
(``business_definition``, ``sql_definition``, ``python_definition``,
``default_filters``) closes the prior active row (``valid_to = now()``) and
inserts a brand-new row, both inside one ``BEGIN IMMEDIATE`` transaction.
Metadata changes (``confidence``, ``source``) update the active row in
place. Identical inputs are a no-op.

Rows are immutable once closed; history is reconstructed by ordering rows on
``(project_id, normalized_name, valid_from ASC)``.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import struct
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from mads.memory.embedding import Embedder
from mads.memory.sql_column_parser import extract_columns_from_sql

# ----------------------------------------------------------------- migration

_MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "infra"
    / "migrations"
    / "0003_metric_memory.sql"
)

# B3: sibling migration adding the sqlite-vec virtual table that backs
# semantic retrieval over active metric rows. Applied AFTER the base
# 0003 migration, on demand: only when ``embedder is not None`` (so
# stores constructed without an embedder don't pay for the vec table).
_VEC_MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "infra"
    / "migrations"
    / "0008_metric_embeddings.sql"
)

# v1.4: required_columns_json column for schema-coupling against cat 2.
# Idempotently ALTER-applied (PRAGMA-guarded) — see init_schema().
_REQUIRED_COLUMNS_MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "infra"
    / "migrations"
    / "0009_metric_required_columns.sql"
)

# Fallback inline migration if the file is unavailable (e.g. when the package
# is consumed from a wheel that doesn't ship ``infra/``).
_INLINE_MIGRATION_SQL = """
CREATE TABLE IF NOT EXISTS metric_memory (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    business_definition TEXT NOT NULL,
    sql_definition TEXT,
    python_definition TEXT,
    default_filters TEXT,
    valid_from TIMESTAMP NOT NULL DEFAULT (CURRENT_TIMESTAMP),
    valid_to TIMESTAMP,
    confidence REAL NOT NULL DEFAULT 1.0 CHECK (confidence BETWEEN 0.0 AND 1.0),
    source TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT (CURRENT_TIMESTAMP),
    updated_at TIMESTAMP NOT NULL DEFAULT (CURRENT_TIMESTAMP)
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_metric_active
    ON metric_memory(project_id, normalized_name)
    WHERE valid_to IS NULL;

CREATE INDEX IF NOT EXISTS idx_metric_history
    ON metric_memory(project_id, normalized_name, valid_from);
"""

# B3: inline fallback for the vec migration. Kept in sync with
# infra/migrations/0008_metric_embeddings.sql.
_INLINE_VEC_MIGRATION_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS metric_vec USING vec0(
    rowid INTEGER PRIMARY KEY,
    project_id TEXT,
    embedding float[384]
);
"""

# v1.4: inline fallback for the required_columns migration. The DDL is a
# plain ALTER TABLE ADD COLUMN (nullable, default NULL). Applied via a
# PRAGMA table_info guard so init_schema stays idempotent.
_INLINE_MIGRATION_SQL_REQUIRED_COLUMNS = (
    "ALTER TABLE metric_memory ADD COLUMN required_columns_json TEXT;"
)


_COLUMNS = (
    "id",
    "user_id",
    "project_id",
    "metric_name",
    "normalized_name",
    "business_definition",
    "sql_definition",
    "python_definition",
    "default_filters",
    "valid_from",
    "valid_to",
    "confidence",
    "source",
    "created_at",
    "updated_at",
    "required_columns_json",
)

# Fields that, when changed, trigger a new version (close + insert).
_DEFINITIONAL_FIELDS = (
    "business_definition",
    "sql_definition",
    "python_definition",
    "default_filters",
)

# Fields that, when changed alone, update the active row in place.
_METADATA_FIELDS = ("confidence", "source")

UpsertAction = Literal["inserted", "versioned", "updated_in_place", "noop"]


# --------------------------------------------------------------------- helpers


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_NORMALIZE_RE = re.compile(r"[^0-9a-z]+")


def _normalize(name: str) -> str:
    """Deterministic, identity-bearing form of a metric name.

    Lower-cases, collapses any non-alphanumeric run to a single underscore,
    strips leading/trailing underscores. Raises ``ValueError`` if the result
    is empty.
    """
    if not isinstance(name, str):
        raise ValueError("metric_name must be a non-empty string")
    cleaned = _NORMALIZE_RE.sub("_", name.lower()).strip("_")
    if not cleaned:
        raise ValueError("metric_name must be non-empty after normalization")
    return cleaned


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {col: row[col] for col in _COLUMNS}


def _pack_vector(vec: list[float]) -> bytes:
    """Pack a float vector for sqlite-vec storage (little-endian float32)."""
    return struct.pack(f"{len(vec)}f", *vec)


def _build_embed_text(metric_name: str, business_definition: Optional[str]) -> str:
    """Concatenation the embedder sees.

    Format: ``"<metric_name>. <business_definition>"``. If business_definition
    is empty/None, fall back to the bare metric_name — embedding nothing but
    punctuation would give a degenerate vector.

    PII note: metric names and business definitions are LLM-authored narrative
    about dataset semantics ("qualified_customer = tenure > 12 AND
    MonthlyCharges > 50"), not user-PII records. Mirrors cat 5's stance on
    embedding analyst notes. Flag if a future code path ever embeds raw user
    fields here.
    """
    name = metric_name or ""
    body = (business_definition or "").strip()
    if not body:
        return name
    return f"{name}. {body}"


# --------------------------------------------------------------------- model


class MetricDefinition(BaseModel):
    """Typed view of one row in ``metric_memory``.

    The store works in plain dicts internally (matching the conventions of
    cats 1 and 2). ``MetricDefinition`` is the typed handoff at the public
    boundary — store methods return instances of this model.

    v1.4 ``required_columns`` semantics:

    * ``None`` — unknown (assume applicable; default for legacy rows
      written before the v1.4 migration and for new rows whose caller
      omitted both ``required_columns`` and ``sql_definition``).
    * ``[]`` — explicit "no columns required" (e.g., a row-count metric).
    * ``["col_a", "col_b"]`` — applicability requires ``{col_a, col_b}
      ⊆ df.columns``.
    """

    id: str
    user_id: str
    project_id: str
    metric_name: str
    normalized_name: str
    business_definition: str
    sql_definition: Optional[str] = None
    python_definition: Optional[str] = None
    default_filters: Optional[str] = None
    valid_from: str
    valid_to: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    source: Optional[str] = None
    created_at: str
    updated_at: str
    required_columns: Optional[list[str]] = None

    @classmethod
    def from_row(cls, row: sqlite3.Row | dict) -> "MetricDefinition":
        if isinstance(row, sqlite3.Row):
            data = _row_to_dict(row)
        else:
            data = dict(row)
        # JSON-decode the required_columns column (NULL stays None; "[]"
        # decodes to an explicit empty list).
        raw = data.pop("required_columns_json", None)
        if raw is None:
            data["required_columns"] = None
        else:
            try:
                decoded = json.loads(raw)
            except (TypeError, ValueError):
                decoded = None
            # Defensive: only accept list-of-string shapes; anything else
            # collapses to None (== unknown).
            if isinstance(decoded, list) and all(isinstance(x, str) for x in decoded):
                data["required_columns"] = decoded
            else:
                data["required_columns"] = None
        return cls.model_validate(data)


# --------------------------------------------------------------------- store


class MetricDefinitionStore:
    """SQLite-backed store for project-scoped metric definitions with versioning."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        *,
        conn: Optional[sqlite3.Connection] = None,
        lock: Optional[threading.RLock] = None,
        embedder: Optional[Embedder] = None,
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

            # ``check_same_thread=False`` + ``self._lock`` is the standard
            # MADS pattern: langgraph's ToolNode dispatches across threads,
            # so the connection must be cross-thread-tolerant and the lock
            # serialises BEGIN/COMMIT to avoid nested-tx errors.
            self._conn = sqlite3.connect(
                db_path,
                isolation_level=None,  # autocommit; we manage txns explicitly
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON;")
            if db_path != ":memory:":
                self._conn.execute("PRAGMA journal_mode = WAL;")

        # See DatasetSchemaStore for why the lock must be shared when the
        # connection is shared (MemoryManager passes both).
        self._lock = lock if lock is not None else threading.RLock()

        # B3: embedder is optional. When None, the store falls back to
        # legacy string-only behavior — `add_metric_definition` skips
        # vec writes, `backfill_embeddings` is a no-op, and any caller
        # building a MetricRetriever on this store will also need to
        # handle the embedder=None graceful-degradation path. This keeps
        # the existing test suite (which does not pass an embedder) green
        # while production callers (`MemoryManager`) thread their shared
        # Embedder through.
        self._embedder: Optional[Embedder] = embedder

        # When an embedder IS present, ensure the sqlite-vec extension is
        # loaded on the connection. MemoryManager already loads it for
        # cat 5; calling enable_load_extension/load again is idempotent
        # and safe. We only do this on an owned connection — a shared
        # conn is the manager's responsibility.
        if self._embedder is not None and self._owns_conn:
            try:
                import sqlite_vec  # local import — cheap
                self._conn.enable_load_extension(True)
                sqlite_vec.load(self._conn)
                self._conn.enable_load_extension(False)
            except Exception:  # noqa: BLE001
                # If sqlite-vec isn't available, demote to string-only.
                # This shouldn't happen in production (dep is in
                # pyproject.toml) but keeps tests robust if it ever does.
                self._embedder = None

        self.init_schema()

    # ------------------------------------------------------------------ schema

    def init_schema(self) -> None:
        """Apply the 0003 migration (and 0008 when an embedder is wired). Idempotent.

        v1.4: also applies the 0009 ``required_columns_json`` column add,
        PRAGMA-guarded so re-init is a no-op on already-migrated DBs.
        """
        sql: str
        try:
            sql = _MIGRATION_PATH.read_text()
        except (FileNotFoundError, OSError):
            sql = _INLINE_MIGRATION_SQL
        self._conn.executescript(sql)
        # B3: only apply the vec migration when an embedder is wired —
        # tests that don't pass one shouldn't pay the cost of the vec
        # table existing (or risk failure if sqlite-vec isn't loaded).
        if self._embedder is not None:
            try:
                vec_sql = _VEC_MIGRATION_PATH.read_text()
            except (FileNotFoundError, OSError):
                vec_sql = _INLINE_VEC_MIGRATION_SQL
            self._conn.executescript(vec_sql)
        # v1.4: ALTER TABLE ADD COLUMN is not natively idempotent (no IF
        # NOT EXISTS for columns), so guard via PRAGMA table_info before
        # applying. Skipped silently when the column is already present.
        cols = [
            r[1]
            for r in self._conn.execute("PRAGMA table_info(metric_memory)").fetchall()
        ]
        if "required_columns_json" not in cols:
            # Prefer the on-disk migration file (single source of truth)
            # with the inline string as fallback for wheel installs.
            try:
                rc_sql = _REQUIRED_COLUMNS_MIGRATION_PATH.read_text()
            except (FileNotFoundError, OSError):
                rc_sql = _INLINE_MIGRATION_SQL_REQUIRED_COLUMNS
            # The migration file contains comments + a single ALTER; both
            # forms are safe for executescript.
            self._conn.executescript(rc_sql)

    # ----------------------------------------------------- embedding helpers

    def _embed_for_metric(
        self, metric_name: str, business_definition: Optional[str]
    ) -> Optional[bytes]:
        """Embed the metric's identifying narrative; return packed bytes.

        Returns None when no embedder is wired. Embedding failures (e.g.
        sentence-transformers download flake) propagate so the calling
        transaction rolls back — better to fail loudly than silently
        leave the vec table out of sync.
        """
        if self._embedder is None:
            return None
        text = _build_embed_text(metric_name, business_definition)
        if not text.strip():
            return None
        vector = self._embedder.embed(text)
        if len(vector) != self._embedder.dim:
            raise RuntimeError(
                f"embedder returned {len(vector)} dims, expected "
                f"{self._embedder.dim}"
            )
        return _pack_vector(vector)

    def _vec_insert(
        self,
        cur: sqlite3.Cursor,
        *,
        rowid: int,
        project_id: str,
        packed: Optional[bytes],
    ) -> None:
        """INSERT into metric_vec, no-op when embedder is None."""
        if packed is None:
            return
        cur.execute(
            "INSERT INTO metric_vec(rowid, project_id, embedding) "
            "VALUES (?, ?, ?)",
            (rowid, project_id, packed),
        )

    def _vec_delete(
        self, cur: sqlite3.Cursor, *, rowid: int
    ) -> None:
        """DELETE from metric_vec by rowid, no-op when embedder is None."""
        if self._embedder is None:
            return
        cur.execute("DELETE FROM metric_vec WHERE rowid = ?", (rowid,))

    # ------------------------------------------------------------------ writes

    def upsert(
        self,
        *,
        user_id: str,
        project_id: str,
        metric_name: str,
        business_definition: str,
        sql_definition: Optional[str] = None,
        python_definition: Optional[str] = None,
        default_filters: Optional[str] = None,
        confidence: float = 1.0,
        source: Optional[str] = None,
        required_columns: Optional[list[str]] = None,
    ) -> tuple[MetricDefinition, UpsertAction]:
        """Register or update a metric.

        Returns ``(definition, action)`` where ``action`` is one of
        ``"inserted"``, ``"versioned"``, ``"updated_in_place"``, ``"noop"``.

        Algorithm:
          1. If no active row → INSERT, action = ``"inserted"``.
          2. If active row and all four definitional fields match:
             - identical confidence/source → ``"noop"``;
             - else UPDATE active row in place → ``"updated_in_place"``.
          3. If any definitional field differs → close old + insert new in one
             transaction → ``"versioned"``.

        v1.4: ``required_columns`` semantics:

        * ``None`` AND non-empty ``sql_definition`` → auto-parse columns
          out of the SQL via :func:`extract_columns_from_sql`.
        * ``None`` AND empty ``sql_definition`` → persist NULL ("unknown").
        * ``[]`` → persist an explicit empty list ("no columns required").
        * ``[<col>, ...]`` → persist as JSON.

        ``required_columns`` is treated as a definitional field for
        versioning purposes (parallel to ``sql_definition``): a change
        triggers a new version, not an in-place edit, because the
        applicability surface is part of the metric's definition.
        """
        # ----- input validation (cheap, before touching the DB)
        if not isinstance(user_id, str) or not user_id:
            raise ValueError("user_id must be a non-empty string")
        if not isinstance(project_id, str) or not project_id:
            raise ValueError("project_id must be a non-empty string")
        if not isinstance(metric_name, str) or not metric_name.strip():
            raise ValueError("metric_name must be a non-empty string")
        if not isinstance(business_definition, str) or not business_definition.strip():
            raise ValueError("business_definition must be a non-empty string")
        if not (0.0 <= float(confidence) <= 1.0):
            raise ValueError(
                f"confidence must be in [0, 1], got {confidence!r}"
            )
        if required_columns is not None:
            if not isinstance(required_columns, list) or not all(
                isinstance(c, str) for c in required_columns
            ):
                raise ValueError(
                    "required_columns must be a list[str] or None"
                )

        # v1.4: derive required_columns from sql_definition if the caller
        # left it None. Empty/whitespace sql -> stays None (unknown).
        effective_required: Optional[list[str]] = required_columns
        if effective_required is None and sql_definition and sql_definition.strip():
            parsed = extract_columns_from_sql(sql_definition)
            # Persist [] vs None deliberately: parser returned an empty
            # list means we tried and found nothing => "unknown" stays
            # the safer default. Only persist a non-empty parse result.
            if parsed:
                effective_required = parsed

        required_json: Optional[str] = (
            json.dumps(effective_required) if effective_required is not None else None
        )

        normalized = _normalize(metric_name)

        with self._lock:
            prior_row = self._conn.execute(
                """
                SELECT rowid AS _rowid_, * FROM metric_memory
                 WHERE project_id = ?
                   AND normalized_name = ?
                   AND valid_to IS NULL
                 LIMIT 1
                """,
                (project_id, normalized),
            ).fetchone()

            now = _utcnow_iso()

            # ----- Case 1: no active row -> INSERT
            if prior_row is None:
                new_id = uuid.uuid4().hex
                # Embed BEFORE opening the transaction. If embedding
                # raises (e.g. ST model download flake), the metric row
                # never lands — the two tables can't diverge.
                packed = self._embed_for_metric(metric_name, business_definition)
                cur = self._conn.cursor()
                try:
                    cur.execute("BEGIN IMMEDIATE;")
                    cur.execute(
                        """
                        INSERT INTO metric_memory (
                            id, user_id, project_id, metric_name, normalized_name,
                            business_definition, sql_definition, python_definition,
                            default_filters,
                            valid_from, valid_to, confidence, source,
                            created_at, updated_at, required_columns_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?)
                        """,
                        (
                            new_id,
                            user_id,
                            project_id,
                            metric_name,
                            normalized,
                            business_definition,
                            sql_definition,
                            python_definition,
                            default_filters,
                            now,
                            float(confidence),
                            source,
                            now,
                            now,
                            required_json,
                        ),
                    )
                    new_rowid = cur.lastrowid
                    self._vec_insert(
                        cur, rowid=new_rowid, project_id=project_id, packed=packed
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

                new_row = self._conn.execute(
                    "SELECT * FROM metric_memory WHERE id = ?", (new_id,)
                ).fetchone()
                return MetricDefinition.from_row(new_row), "inserted"

            # ----- Case 2 / 3: classify against the prior active row
            prior = _row_to_dict(prior_row)
            new_values = {
                "business_definition": business_definition,
                "sql_definition": sql_definition,
                "python_definition": python_definition,
                "default_filters": default_filters,
            }
            defs_changed = any(prior[k] != new_values[k] for k in _DEFINITIONAL_FIELDS)
            # v1.4: required_columns_json is part of the definitional surface
            # (applicability is a property of the metric, not metadata).
            # NULL == NULL on the prior path is fine because we compare the
            # JSON strings — both NULL means "still unknown, no change".
            prior_required = prior.get("required_columns_json")
            defs_changed = defs_changed or (prior_required != required_json)

            if not defs_changed:
                meta_changed = (
                    float(prior["confidence"]) != float(confidence)
                    or prior["source"] != source
                )
                if not meta_changed:
                    return MetricDefinition.from_row(prior_row), "noop"

                # Update the active row in place. Note: we intentionally do NOT
                # touch user_id on a metadata-only edit -- the row already records
                # who created this version; an in-place tweak of confidence does
                # not rewrite authorship.
                cur = self._conn.cursor()
                try:
                    cur.execute("BEGIN IMMEDIATE;")
                    cur.execute(
                        """
                        UPDATE metric_memory
                           SET confidence = ?, source = ?, updated_at = ?
                         WHERE id = ?
                        """,
                        (float(confidence), source, now, prior["id"]),
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

                updated_row = self._conn.execute(
                    "SELECT * FROM metric_memory WHERE id = ?", (prior["id"],)
                ).fetchone()
                return MetricDefinition.from_row(updated_row), "updated_in_place"

            # ----- Case 3: a definitional field changed -> version it
            new_id = uuid.uuid4().hex
            # Embed BEFORE opening the tx so embedder failures don't
            # leave a half-written transaction.
            packed = self._embed_for_metric(metric_name, business_definition)
            prior_rowid = prior_row["_rowid_"]
            cur = self._conn.cursor()
            try:
                cur.execute("BEGIN IMMEDIATE;")
                cur.execute(
                    """
                    UPDATE metric_memory
                       SET valid_to = ?, updated_at = ?
                     WHERE id = ?
                    """,
                    (now, now, prior["id"]),
                )
                # Drop the closed row's vector — vec table tracks active
                # rows only (simpler retrieval; no valid_to JOIN filter).
                self._vec_delete(cur, rowid=prior_rowid)
                cur.execute(
                    """
                    INSERT INTO metric_memory (
                        id, user_id, project_id, metric_name, normalized_name,
                        business_definition, sql_definition, python_definition,
                        default_filters,
                        valid_from, valid_to, confidence, source,
                        created_at, updated_at, required_columns_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?)
                    """,
                    (
                        new_id,
                        user_id,
                        project_id,
                        metric_name,
                        normalized,
                        business_definition,
                        sql_definition,
                        python_definition,
                        default_filters,
                        now,
                        float(confidence),
                        source,
                        now,
                        now,
                        required_json,
                    ),
                )
                new_rowid = cur.lastrowid
                self._vec_insert(
                    cur, rowid=new_rowid, project_id=project_id, packed=packed
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

            new_row = self._conn.execute(
                "SELECT * FROM metric_memory WHERE id = ?", (new_id,)
            ).fetchone()
            return MetricDefinition.from_row(new_row), "versioned"

    # ------------------------------------------------------------------- reads

    def get_active(
        self, *, project_id: str, metric_name: str
    ) -> Optional[MetricDefinition]:
        """Return the currently-active metric for (project_id, normalized name), or None."""
        try:
            normalized = _normalize(metric_name)
        except ValueError:
            return None
        with self._lock:
            row = self._conn.execute(
                """
                SELECT * FROM metric_memory
                 WHERE project_id = ?
                   AND normalized_name = ?
                   AND valid_to IS NULL
                 LIMIT 1
                """,
                (project_id, normalized),
            ).fetchone()
        return MetricDefinition.from_row(row) if row is not None else None

    def list_active(self, *, project_id: str) -> list[MetricDefinition]:
        """All active metrics in the project, ordered by metric_name ASC (case-insensitive)."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM metric_memory
                 WHERE project_id = ?
                   AND valid_to IS NULL
                 ORDER BY LOWER(metric_name) ASC
                """,
                (project_id,),
            ).fetchall()
        return [MetricDefinition.from_row(r) for r in rows]

    def search_active(
        self, *, project_id: str, query: str
    ) -> list[MetricDefinition]:
        """Case-insensitive substring search over (metric_name, business_definition)."""
        if not isinstance(query, str):
            return []
        like = f"%{query.lower()}%"
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM metric_memory
                 WHERE project_id = ?
                   AND valid_to IS NULL
                   AND (
                        LOWER(metric_name) LIKE ?
                     OR LOWER(business_definition) LIKE ?
                   )
                 ORDER BY LOWER(metric_name) ASC
                """,
                (project_id, like, like),
            ).fetchall()
        return [MetricDefinition.from_row(r) for r in rows]

    def history(
        self, *, project_id: str, metric_name: str
    ) -> list[MetricDefinition]:
        """Full version history (oldest first) for the metric in this project."""
        try:
            normalized = _normalize(metric_name)
        except ValueError:
            return []
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM metric_memory
                 WHERE project_id = ?
                   AND normalized_name = ?
                 ORDER BY valid_from ASC, created_at ASC
                """,
                (project_id, normalized),
            ).fetchall()
        return [MetricDefinition.from_row(r) for r in rows]

    # ---------------------------------------------------------- semantic search

    def find_similar(
        self, *, project_id: str, query: str, k: int = 5
    ) -> list[tuple[MetricDefinition, float]]:
        """Semantic similarity search over active metrics in this project.

        Returns ``[(metric, distance)]`` sorted ascending by distance.
        Returns ``[]`` when no embedder is wired or no rows match.
        sqlite-vec returns L2 distance over unit-norm embeddings →
        bounded in ``[0, 2]``; callers can convert to a similarity
        score with ``1.0 - clamp(distance, 0, 2)/2``.
        """
        if self._embedder is None:
            return []
        if not project_id:
            return []
        if not isinstance(query, str) or not query.strip():
            return []
        if not isinstance(k, int) or k <= 0:
            return []

        vector = self._embedder.embed(query)
        packed = _pack_vector(vector)

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT m.rowid AS _rowid_, m.*, v.distance AS distance
                  FROM metric_vec AS v
                  JOIN metric_memory AS m ON m.rowid = v.rowid
                 WHERE v.embedding MATCH ?
                   AND v.project_id = ?
                   AND v.k = ?
                   AND m.valid_to IS NULL
                 ORDER BY v.distance ASC
                """,
                (packed, project_id, k),
            ).fetchall()

        out: list[tuple[MetricDefinition, float]] = []
        for r in rows:
            metric = MetricDefinition.from_row({c: r[c] for c in _COLUMNS})
            out.append((metric, float(r["distance"])))
        return out

    def backfill_embeddings(self) -> int:
        """Backfill embeddings for every active row missing from ``metric_vec``.

        Returns the number of rows newly embedded. Useful for migrating
        existing databases when an embedder is wired into a store whose
        prior writes happened without one (e.g. eval DBs created before
        B3). Idempotent — already-embedded rows are skipped.

        No-op when ``self._embedder is None``.
        """
        if self._embedder is None:
            return 0
        # Find all active rows whose rowid is not yet in metric_vec.
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT m.rowid AS _rowid_, m.project_id, m.metric_name,
                       m.business_definition
                  FROM metric_memory AS m
                 WHERE m.valid_to IS NULL
                   AND NOT EXISTS (
                       SELECT 1 FROM metric_vec v WHERE v.rowid = m.rowid
                   )
                """
            ).fetchall()
        n = 0
        for r in rows:
            packed = self._embed_for_metric(r["metric_name"], r["business_definition"])
            if packed is None:
                continue
            with self._lock:
                cur = self._conn.cursor()
                try:
                    cur.execute("BEGIN IMMEDIATE;")
                    self._vec_insert(
                        cur,
                        rowid=r["_rowid_"],
                        project_id=r["project_id"],
                        packed=packed,
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
            n += 1
        return n

    # ---------------------------------------------- v1.4 applicability surface

    @staticmethod
    def is_applicable(
        metric: MetricDefinition, available_columns: set[str]
    ) -> tuple[bool, list[str]]:
        """Decide whether a stored metric applies to the current df schema.

        Returns ``(applicable, missing_columns_sorted)``.

        * ``metric.required_columns is None`` → ``(True, [])``: legacy or
          unknown — assume applicable (don't penalize rows the LLM hadn't
          annotated at write time).
        * ``metric.required_columns == []`` → ``(True, [])``: explicit
          "no columns required" (e.g., a row-count metric).
        * Otherwise: ``required ⊆ available_columns`` → applicable, else
          ``missing = sorted(required - available_columns)``.

        ``available_columns`` is a ``set[str]``; passing a ``list`` works
        too but ``set`` makes the membership check explicit at the call
        site (see :class:`MetricRetriever`).
        """
        req = metric.required_columns
        if req is None or len(req) == 0:
            return True, []
        avail = set(available_columns) if not isinstance(available_columns, set) else available_columns
        missing = sorted(set(req) - avail)
        return (len(missing) == 0), missing

    def backfill_required_columns(self) -> int:
        """Backfill ``required_columns_json`` for active rows that left it NULL.

        Only fills rows that (a) have ``valid_to IS NULL``, (b) have
        ``required_columns_json IS NULL``, AND (c) have a non-empty
        ``sql_definition`` we can parse. Rows with empty/NULL
        ``sql_definition`` stay NULL (= unknown).

        Returns the number of rows updated. Idempotent — second call
        returns 0. Called explicitly from scripts / tests, never from
        :meth:`upsert`.

        v1.4: kept deliberately separate from :meth:`upsert` so a fresh
        ``add_metric_definition`` from the tool surface stays a single
        atomic write, and so callers can choose to backfill an
        existing eval database opportunistically.
        """
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, sql_definition FROM metric_memory
                 WHERE valid_to IS NULL
                   AND required_columns_json IS NULL
                   AND sql_definition IS NOT NULL
                   AND TRIM(sql_definition) != ''
                """
            ).fetchall()
        n = 0
        for r in rows:
            parsed = extract_columns_from_sql(r["sql_definition"])
            if not parsed:
                continue
            payload = json.dumps(parsed)
            with self._lock:
                cur = self._conn.cursor()
                try:
                    cur.execute("BEGIN IMMEDIATE;")
                    cur.execute(
                        "UPDATE metric_memory SET required_columns_json = ?, "
                        "updated_at = ? WHERE id = ?",
                        (payload, _utcnow_iso(), r["id"]),
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
            n += 1
        return n

    # ---------------------------------------------------------------- lifecycle

    def close(self) -> None:
        if self._conn is not None and self._owns_conn:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]
