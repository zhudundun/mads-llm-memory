-- 0007_dataset_axis_for_preferences.sql
--
-- v1.3 axis-model extension: add the DATASET axis to user_preference
-- (cat 1). Identity tuple changes from
--   (user_id, project_id, preference_key)
-- to
--   (user_id, project_id, dataset_fingerprint, preference_key).
--
-- Both project_id and dataset_fingerprint are IDENTITY_OPTIONAL — NULL
-- means "applies to all <axis>" — so the four scopes (global,
-- project-only, dataset-only, project+dataset) can coexist for the same
-- (user, key) tuple.
--
-- Idempotent: SQLite has no "ALTER ... ADD COLUMN IF NOT EXISTS", so
-- we rely on the store's _init_schema() PRAGMA check before calling
-- this migration on file-mode DBs. Fresh DBs get the new shape via the
-- CREATE TABLE IF NOT EXISTS sentinel below.

-- Sentinel: a fresh DB gets the new layout directly. Existing tables
-- are left untouched here; the store's _init_schema runs the ADD COLUMN
-- via PRAGMA when needed.
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

-- Replace the old (user_id, project_id, preference_key) active-row
-- index with the (user_id, project_id, dataset_fingerprint,
-- preference_key) form so the four scope variants can coexist on the
-- same (user, key) tuple. COALESCE collapses NULL → '' inside the
-- expression so the index treats global vs scoped rows as distinct
-- but doesn't accidentally allow two GLOBAL active rows.
DROP INDEX IF EXISTS ix_active_user_pref;
CREATE UNIQUE INDEX IF NOT EXISTS ix_active_user_pref
    ON user_preference_memory (
        user_id,
        COALESCE(project_id, ''),
        COALESCE(dataset_fingerprint, ''),
        preference_key
    )
    WHERE valid_to IS NULL;

-- Covering index for the retrieval filter pattern:
--   WHERE user_id = ?
--     AND (project_id IS NULL OR project_id = ?)
--     AND (dataset_fingerprint IS NULL OR dataset_fingerprint = ?)
--     AND valid_to IS NULL
-- Speeds up PreferenceRetriever lookups on busy DBs.
CREATE INDEX IF NOT EXISTS ix_pref_scope_lookup
    ON user_preference_memory (
        user_id, project_id, dataset_fingerprint, valid_to
    );
