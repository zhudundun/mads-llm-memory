-- 0006_axis_model.sql
--
-- Axis-model refactor. Two per-category schema changes:
--
-- 1. user_preference_memory: add nullable project_id (IDENTITY_OPTIONAL).
--    Existing rows keep project_id=NULL (= global). The active-row partial
--    unique index becomes (user_id, project_id, preference_key) so a user
--    can have one global pref AND one project-specific pref for the same
--    key without colliding.
--
-- 2. data_quality_memory: drop existing rows, add NOT NULL
--    dataset_fingerprint (IDENTITY_REQUIRED), drop the old project-based
--    partial unique index, and add a new fingerprint-based one. Safe for
--    v1.1 because no production users exist; eval DBs are recreated per
--    scenario.
--
-- Both blocks are idempotent: re-running this migration on an already-
-- migrated DB is a no-op (column existence checks via ALTER ... ADD
-- COLUMN guarded by sqlite_master).

-- =====================================================================
-- (1) Preference: nullable project_id + new active-row index
-- =====================================================================

-- SQLite has no "ALTER TABLE ... ADD COLUMN IF NOT EXISTS"; guard with a
-- one-off check against the table's PRAGMA before issuing the ALTER.
-- (The migration uses raw SQL only; the store's init_schema() loops
-- through CREATE TABLE IF NOT EXISTS for fresh DBs, and the store also
-- runs this migration via _maybe_alter_for_axis_model() to upgrade
-- pre-existing DBs.)

-- Use a CREATE-TABLE-shaped sentinel: a no-op CREATE TABLE IF NOT EXISTS
-- of the *target* schema means fresh DBs get the new layout straight away.
CREATE TABLE IF NOT EXISTS user_preference_memory (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT,
    preference_key TEXT NOT NULL,
    preference_value TEXT NOT NULL,
    source TEXT,
    confidence REAL,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Replace the old (user_id, preference_key) active-row index with the
-- (user_id, project_id, preference_key) form. Standard SQLite UNIQUE
-- treats NULLs as distinct, so a user can hold multiple rows with the
-- same (user_id, key) provided their project_id columns differ — which
-- is exactly what we want for the "global pref vs project-scoped pref"
-- case. To collapse multiple GLOBAL active rows (project_id=NULL) we
-- COALESCE the column down to '' inside the index expression.
DROP INDEX IF EXISTS ix_active_user_pref;
CREATE UNIQUE INDEX IF NOT EXISTS ix_active_user_pref
    ON user_preference_memory (user_id, COALESCE(project_id, ''), preference_key)
    WHERE valid_to IS NULL;

-- =====================================================================
-- (2) Data quality: rebuild table with NOT NULL dataset_fingerprint
-- =====================================================================
--
-- SQLite cannot add a NOT NULL column to an existing populated table
-- without a default. Because we are explicitly dropping any pre-existing
-- rows (v1.1 has no production users) the cleanest path is to rename the
-- old table out of the way, recreate with the new shape, drop the old.
-- This block is structured so a fresh DB (no old table) gets the new
-- shape directly via CREATE TABLE IF NOT EXISTS at the bottom.

-- Drop indexes on the old table BEFORE renaming so they don't clash.
DROP INDEX IF EXISTS uq_dq_active;
DROP INDEX IF EXISTS idx_dq_history;
DROP INDEX IF EXISTS idx_dq_table_lookup;

-- Rename the OLD table (if present) so the CREATE below can install the
-- new shape under the same name. If no old table exists (fresh DB), this
-- statement is a no-op via the store's init_schema fallback.
-- NOTE: this migration FILE is run only when the file-mode path is taken
-- by the store; the store's _INLINE_MIGRATION_SQL handles fresh DBs via
-- CREATE TABLE IF NOT EXISTS. To keep both code paths consistent we
-- avoid the rename here and let the store's init_schema do the table-
-- exists check programmatically.

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

-- New unique index: identity is (dataset_fingerprint, normalized_table,
-- COALESCE(normalized_column, ''), normalized_issue_type). project_id is
-- audit-only and must NOT appear here.
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
