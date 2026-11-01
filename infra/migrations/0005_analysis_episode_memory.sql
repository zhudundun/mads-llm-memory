-- 0005_analysis_episode_memory.sql
--
-- Storage for the `analysis_episode` memory category (category 5 — prior
-- analysis conclusions).
--
-- This category is STRUCTURALLY DIFFERENT from cats 1–4. Episodes are
-- append-only EVENTS, not versioned definitions. There is no identity
-- tuple, no merge classifier, no `valid_from` / `valid_to`. Each episode
-- gets an auto-generated UUID primary key plus a `created_at` timestamp;
-- that is the only ordering. The only mutable field is `user_feedback`
-- (and the implicit `updated_at`); any update to `user_feedback` bumps
-- `updated_at` AND regenerates the embedding (since user_feedback is part
-- of the embedded text).
--
-- The vector index lives in a separate virtual table (`analysis_episode_vec`)
-- backed by the sqlite-vec extension. The two tables stay in sync at the
-- application layer: every INSERT/UPDATE/DELETE on `analysis_episode_memory`
-- has a matching write to `analysis_episode_vec`, wrapped in one
-- `BEGIN IMMEDIATE` transaction so divergence is not observable to readers.
--
-- The vec table declares `project_id TEXT` as a metadata column (no `+`
-- prefix). Metadata columns can be filtered inside MATCH queries, so
-- `WHERE embedding MATCH ? AND project_id = ?` pushes the tenancy filter
-- into the KNN scan rather than post-filtering in Python. (Note: the
-- original implementation brief asked for an auxiliary column `+project_id`,
-- but sqlite-vec rejects WHERE constraints on auxiliary columns inside KNN
-- queries — see IMPLEMENTATION_NOTES_ANALYSIS_EPISODE.md.)

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

-- sqlite-vec virtual table holding the 384-dim embeddings. `episode_id` is
-- the join key back to `analysis_episode_memory.id`; `project_id` is a
-- metadata column (filterable inside MATCH queries).
CREATE VIRTUAL TABLE IF NOT EXISTS analysis_episode_vec USING vec0(
    episode_id TEXT PRIMARY KEY,
    project_id TEXT,
    embedding float[384]
);
