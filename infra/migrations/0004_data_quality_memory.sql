-- 0004_data_quality_memory.sql
--
-- Storage for the `data_quality_issue` memory category (category 4).
--
-- Identity: (project_id, normalized_table, normalized_column, normalized_issue_type).
-- `normalized_column` is NULLABLE -- table-level issues (e.g. row-filter
-- requirements that apply to the whole table) have `column_name = NULL`.
-- `dataset_fingerprint` is advisory metadata recorded on the row but does
-- NOT participate in identity.
--
-- Versioning: any change to one of the *definitional* fields
-- (`issue_description`, `recommended_handling`) closes the prior active row
-- (`valid_to = now`) and inserts a new row inside one `BEGIN IMMEDIATE`
-- transaction. Metadata changes (`severity`, `confidence`,
-- `dataset_fingerprint`) update the active row in place; identical inputs
-- are a no-op. `severity` is the team's current assessment, not a
-- definition -- it tracks the same way `confidence` does in cat 3.
--
-- `normalized_table` / `normalized_column` / `normalized_issue_type` are
-- stored, deterministic forms used for identity / partial-unique enforcement.
-- The user-facing `table_name`, `column_name`, `issue_type` columns preserve
-- the caller's original casing.
--
-- Resolution model: marking an issue resolved sets `valid_to = now` on the
-- active row with NO replacement row inserted. The issue exits the active
-- list but remains in `history`. Re-adding the same identity after a resolve
-- starts a fresh active row -- it is NOT a continuation of the old chain.

CREATE TABLE IF NOT EXISTS data_quality_memory (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    dataset_fingerprint TEXT,
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

-- COALESCE(normalized_column, '') collapses NULLs so that two table-level
-- issues with the same (project_id, table, issue_type) collide as expected.
-- (Standard SQL UNIQUE treats NULLs as distinct, which would silently allow
-- duplicate table-level rows.)
CREATE UNIQUE INDEX IF NOT EXISTS uq_dq_active
    ON data_quality_memory(project_id, normalized_table,
                           COALESCE(normalized_column, ''),
                           normalized_issue_type)
    WHERE valid_to IS NULL;

CREATE INDEX IF NOT EXISTS idx_dq_history
    ON data_quality_memory(project_id, normalized_table,
                           COALESCE(normalized_column, ''),
                           normalized_issue_type, valid_from);

CREATE INDEX IF NOT EXISTS idx_dq_table_lookup
    ON data_quality_memory(project_id, normalized_table, valid_to);
