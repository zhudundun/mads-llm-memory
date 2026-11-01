-- 0003_metric_memory.sql
--
-- Storage for the `metric_definition` memory category (category 3).
--
-- Identity: (project_id, normalized_name) -- metrics are shared across users
-- within a project. `user_id` records who created/last-edited the row but does
-- NOT participate in identity.
--
-- Versioning: any change to one of the *definitional* fields
-- (business_definition, sql_definition, python_definition, default_filters)
-- closes the active row (`valid_to = now`) and inserts a new row inside one
-- transaction. Metadata changes (confidence, source) update the active row in
-- place; identical inputs are a no-op.
--
-- `normalized_name` is the stored, deterministic form of `metric_name` used
-- for identity / partial-unique enforcement; the user-facing `metric_name`
-- preserves the caller's original casing. Both columns are NOT NULL.

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
