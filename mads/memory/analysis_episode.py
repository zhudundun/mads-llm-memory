"""SQLite-backed store for the ``analysis_episode`` memory category (cat 5).

Episodes are APPEND-ONLY events, not versioned definitions. There is no
identity tuple, no merge classifier, no ``valid_from``/``valid_to``. Each
episode gets an auto-generated UUID primary key and a ``created_at``
timestamp; that's the only ordering. The only mutable field is
``user_feedback`` — every other field is immutable after INSERT.

The embedding lives in a sibling virtual table (``analysis_episode_vec``)
backed by the ``sqlite-vec`` extension. The two tables stay in sync at the
application layer: every INSERT, DELETE, or feedback-update wraps both
operations in a single ``BEGIN IMMEDIATE`` transaction so divergence is
not externally observable. The synchronization invariant is:

  *Every row in ``analysis_episode_memory`` has a matching row in
  ``analysis_episode_vec`` (by ``episode_id``) and vice versa, and the
  vec-table row reflects the embedding of the current
  ``question + conclusion + assumptions + user_feedback`` text.*

The embedded text is the concatenation of the four free-text fields
(``question``, ``conclusion``, ``assumptions``, ``user_feedback``)
separated by ``"\\n\\n"``. Structured fields (``analysis_type``,
``datasets_used``, ``metrics_used``, ``methods_used``, ``code_refs``,
``chart_refs``) are NOT embedded — they're filter/display metadata.
"""

from __future__ import annotations

import os
import sqlite3
import struct
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import sqlite_vec
from pydantic import BaseModel

from mads.memory.embedding import Embedder, SentenceTransformerEmbedder

# ----------------------------------------------------------------- migration

_MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "infra"
    / "migrations"
    / "0005_analysis_episode_memory.sql"
)

# Fallback inline migration if the file is unavailable (e.g. when the package
# is consumed from a wheel that doesn't ship ``infra/``).
_INLINE_MIGRATION_SQL = """
CREATE TABLE IF NOT EXISTS analysis_episode_memory (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    question TEXT NOT NULL,
    analysis_type TEXT,
    datasets_used TEXT,
    metrics_used TEXT,
    methods_used TEXT,
    code_refs TEXT,
    chart_refs TEXT,
    conclusion TEXT NOT NULL,
    assumptions TEXT,
    user_feedback TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT (CURRENT_TIMESTAMP),
    updated_at TIMESTAMP NOT NULL DEFAULT (CURRENT_TIMESTAMP)
);

CREATE INDEX IF NOT EXISTS idx_episode_project_created
    ON analysis_episode_memory(project_id, created_at DESC);

CREATE VIRTUAL TABLE IF NOT EXISTS analysis_episode_vec USING vec0(
    episode_id TEXT PRIMARY KEY,
    project_id TEXT,
    embedding float[384]
);
"""


_COLUMNS = (
    "id",
    "user_id",
    "project_id",
    "question",
    "analysis_type",
    "datasets_used",
    "metrics_used",
    "methods_used",
    "code_refs",
    "chart_refs",
    "conclusion",
    "assumptions",
    "user_feedback",
    "created_at",
    "updated_at",
)


# --------------------------------------------------------------------- helpers


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {col: row[col] for col in _COLUMNS}


def _build_embed_text(
    question: str,
    conclusion: str,
    assumptions: Optional[str],
    user_feedback: Optional[str],
) -> str:
    """Concatenate the four free-text fields the embedder sees.

    The structured fields (analysis_type, datasets_used, …) are intentionally
    excluded — they're metadata, not searchable narrative.
    """
    parts = [question, conclusion]
    if assumptions:
        parts.append(assumptions)
    if user_feedback:
        parts.append(user_feedback)
    return "\n\n".join(parts)


def _pack_vector(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


# --------------------------------------------------------------------- model


class AnalysisEpisode(BaseModel):
    """Typed view of one row in ``analysis_episode_memory``."""

    id: str
    user_id: str
    project_id: str
    question: str
    analysis_type: Optional[str] = None
    datasets_used: Optional[str] = None
    metrics_used: Optional[str] = None
    methods_used: Optional[str] = None
    code_refs: Optional[str] = None
    chart_refs: Optional[str] = None
    conclusion: str
    assumptions: Optional[str] = None
    user_feedback: Optional[str] = None
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row | dict) -> "AnalysisEpisode":
        data = _row_to_dict(row) if isinstance(row, sqlite3.Row) else dict(row)
        return cls.model_validate(data)


# --------------------------------------------------------------------- store


class AnalysisEpisodeStore:
    """SQLite-backed store for project-scoped, append-only analysis episodes.

    Backed by two tables (``analysis_episode_memory`` for metadata and
    ``analysis_episode_vec`` for embeddings), kept in sync via single-tx
    writes. The embedder is configurable at construction time so tests can
    plug in :class:`mads.memory.embedding.HashEmbedder` and skip the
    sentence-transformers model load.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        *,
        conn: Optional[sqlite3.Connection] = None,
        embedder: Optional[Embedder] = None,
        lock: Optional[threading.RLock] = None,
    ) -> None:
        if conn is not None:
            self._owns_conn = False
            self._db_path = None
            self._conn = conn
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
                isolation_level=None,  # autocommit; we manage txns explicitly
                check_same_thread=False,
            )
            if db_path != ":memory:":
                self._conn.execute("PRAGMA journal_mode = WAL;")

        # sqlite-vec requires the loadable-extension toggle. Python's
        # bundled sqlite on macOS blocks extensions by default.
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        self._conn.row_factory = sqlite3.Row

        self._embedder: Embedder = embedder or SentenceTransformerEmbedder()

        # Re-entrant: write paths SELECT to confirm the row exists, then
        # write — needs to tolerate same-thread re-acquisition. Caller can
        # inject a shared lock to coordinate writes with sibling stores
        # that share the connection (MemoryManager does this).
        self._lock = lock if lock is not None else threading.RLock()
        self.init_schema()

    # ------------------------------------------------------------------ schema

    def init_schema(self) -> None:
        """Apply the 0005 migration. Idempotent."""
        sql: str
        try:
            sql = _MIGRATION_PATH.read_text()
        except (FileNotFoundError, OSError):
            sql = _INLINE_MIGRATION_SQL
        self._conn.executescript(sql)

    # ------------------------------------------------------------------ writes

    def record(
        self,
        *,
        user_id: str,
        project_id: str,
        question: str,
        conclusion: str,
        analysis_type: Optional[str] = None,
        datasets_used: Optional[str] = None,
        metrics_used: Optional[str] = None,
        methods_used: Optional[str] = None,
        code_refs: Optional[str] = None,
        chart_refs: Optional[str] = None,
        assumptions: Optional[str] = None,
        user_feedback: Optional[str] = None,
    ) -> AnalysisEpisode:
        """Insert a new episode and its embedding atomically.

        Generates a fresh UUID, embeds the concatenated free-text fields,
        writes both rows inside a single ``BEGIN IMMEDIATE`` transaction.
        If embedding generation raises, NEITHER table sees a row.
        """
        # Validate inputs before touching the DB or the embedder.
        if not isinstance(user_id, str) or not user_id:
            raise ValueError("user_id must be a non-empty string")
        if not isinstance(project_id, str) or not project_id:
            raise ValueError("project_id must be a non-empty string")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if not isinstance(conclusion, str) or not conclusion.strip():
            raise ValueError("conclusion must be a non-empty string")

        new_id = uuid.uuid4().hex
        now = _utcnow_iso()

        embed_text = _build_embed_text(
            question, conclusion, assumptions, user_feedback
        )
        # Embed BEFORE opening the transaction. If embedding fails, no
        # transaction was opened, so nothing to roll back.
        vector = self._embedder.embed(embed_text)
        if len(vector) != self._embedder.dim:
            raise RuntimeError(
                f"embedder returned {len(vector)} dims, expected "
                f"{self._embedder.dim}"
            )
        packed = _pack_vector(vector)

        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute("BEGIN IMMEDIATE;")
                cur.execute(
                    """
                    INSERT INTO analysis_episode_memory (
                        id, user_id, project_id,
                        question, analysis_type, datasets_used, metrics_used,
                        methods_used, code_refs, chart_refs,
                        conclusion, assumptions, user_feedback,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        new_id,
                        user_id,
                        project_id,
                        question,
                        analysis_type,
                        datasets_used,
                        metrics_used,
                        methods_used,
                        code_refs,
                        chart_refs,
                        conclusion,
                        assumptions,
                        user_feedback,
                        now,
                        now,
                    ),
                )
                cur.execute(
                    "INSERT INTO analysis_episode_vec(episode_id, project_id, embedding) "
                    "VALUES (?, ?, ?)",
                    (new_id, project_id, packed),
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

            out_row = self._conn.execute(
                "SELECT * FROM analysis_episode_memory WHERE id = ?", (new_id,)
            ).fetchone()
            return AnalysisEpisode.from_row(out_row)

    def update_feedback(
        self,
        *,
        user_id: str,
        project_id: str,
        episode_id: str,
        feedback: str,
    ) -> Optional[AnalysisEpisode]:
        """Update ``user_feedback`` on an existing episode and re-embed.

        This is the ONLY field that can be edited after creation. Bumps
        ``updated_at`` and replaces the vec-table row with a fresh
        embedding of the new combined text. Returns ``None`` if no
        matching episode exists in this project.

        ``user_id`` is recorded on the call for parity with cat 4's
        ``resolve`` (audit-style argument) but does not gate visibility —
        episodes are project-scoped, like data-quality issues.
        """
        if not isinstance(user_id, str) or not user_id:
            raise ValueError("user_id must be a non-empty string")
        if not isinstance(project_id, str) or not project_id:
            raise ValueError("project_id must be a non-empty string")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("episode_id must be a non-empty string")
        if not isinstance(feedback, str) or not feedback.strip():
            raise ValueError("feedback must be a non-empty string")

        # Look up the existing row outside the write transaction (it's
        # read-only and we want to short-circuit a missing-id case before
        # paying for the embedding).
        with self._lock:
            prior = self._conn.execute(
                "SELECT * FROM analysis_episode_memory WHERE id = ? AND project_id = ?",
                (episode_id, project_id),
            ).fetchone()
        if prior is None:
            return None

        new_text = _build_embed_text(
            prior["question"],
            prior["conclusion"],
            prior["assumptions"],
            feedback,
        )
        new_vector = self._embedder.embed(new_text)
        packed = _pack_vector(new_vector)
        now = _utcnow_iso()

        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute("BEGIN IMMEDIATE;")
                cur.execute(
                    """
                    UPDATE analysis_episode_memory
                       SET user_feedback = ?, updated_at = ?
                     WHERE id = ? AND project_id = ?
                    """,
                    (feedback, now, episode_id, project_id),
                )
                # Replace the vec row. DELETE-then-INSERT is the supported
                # update path for vec0 PRIMARY KEY rows.
                cur.execute(
                    "DELETE FROM analysis_episode_vec WHERE episode_id = ?",
                    (episode_id,),
                )
                cur.execute(
                    "INSERT INTO analysis_episode_vec(episode_id, project_id, embedding) "
                    "VALUES (?, ?, ?)",
                    (episode_id, project_id, packed),
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

            out_row = self._conn.execute(
                "SELECT * FROM analysis_episode_memory WHERE id = ?", (episode_id,)
            ).fetchone()
            return AnalysisEpisode.from_row(out_row)

    # ------------------------------------------------------------------- reads

    def get(
        self, *, project_id: str, episode_id: str
    ) -> Optional[AnalysisEpisode]:
        """Fetch a single episode by id, scoped to ``project_id``.

        Returns ``None`` if no episode by that id exists in this project
        (including the case where the episode exists in a DIFFERENT
        project — tenancy filter is mandatory).
        """
        if not project_id or not episode_id:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM analysis_episode_memory "
                " WHERE id = ? AND project_id = ?",
                (episode_id, project_id),
            ).fetchone()
        if row is None:
            return None
        return AnalysisEpisode.from_row(row)

    def list_recent(
        self, *, project_id: str, limit: int = 10
    ) -> list[AnalysisEpisode]:
        """Newest-first list of episodes in this project, capped at ``limit``."""
        if not project_id:
            return []
        if not isinstance(limit, int) or limit <= 0:
            return []
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM analysis_episode_memory
                 WHERE project_id = ?
                 ORDER BY created_at DESC, id DESC
                 LIMIT ?
                """,
                (project_id, limit),
            ).fetchall()
        return [AnalysisEpisode.from_row(r) for r in rows]

    def search(
        self, *, project_id: str, query: str
    ) -> list[AnalysisEpisode]:
        """Case-insensitive substring search over question, conclusion,
        assumptions, and user_feedback.

        Structured fields (analysis_type, datasets_used, …) are NOT
        searched here — they are not part of the narrative surface of an
        episode. Use :meth:`find_similar` for semantic queries.
        """
        if not project_id or not isinstance(query, str) or not query:
            return []
        like = f"%{query.lower()}%"
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM analysis_episode_memory
                 WHERE project_id = ?
                   AND (
                        LOWER(question) LIKE ?
                     OR LOWER(conclusion) LIKE ?
                     OR LOWER(COALESCE(assumptions, '')) LIKE ?
                     OR LOWER(COALESCE(user_feedback, '')) LIKE ?
                   )
                 ORDER BY created_at DESC, id DESC
                """,
                (project_id, like, like, like, like),
            ).fetchall()
        return [AnalysisEpisode.from_row(r) for r in rows]

    def find_similar(
        self, *, project_id: str, query: str, k: int = 5
    ) -> list[tuple[AnalysisEpisode, float]]:
        """Semantic similarity search.

        Embeds ``query``, runs a KNN scan against the vec table with the
        project filter pushed into the MATCH clause, joins back to the
        metadata table, and returns ``(episode, distance)`` tuples sorted
        by ascending distance.
        """
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
                SELECT m.*, v.distance AS distance
                  FROM analysis_episode_vec AS v
                  JOIN analysis_episode_memory AS m ON m.id = v.episode_id
                 WHERE v.embedding MATCH ?
                   AND v.project_id = ?
                   AND v.k = ?
                 ORDER BY v.distance ASC
                """,
                (packed, project_id, k),
            ).fetchall()

        out: list[tuple[AnalysisEpisode, float]] = []
        for r in rows:
            episode = AnalysisEpisode.from_row(
                {c: r[c] for c in _COLUMNS}
            )
            out.append((episode, float(r["distance"])))
        return out

    # ---------------------------------------------------------------- lifecycle

    def close(self) -> None:
        if self._conn is not None and self._owns_conn:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]


__all__ = ["AnalysisEpisode", "AnalysisEpisodeStore"]
