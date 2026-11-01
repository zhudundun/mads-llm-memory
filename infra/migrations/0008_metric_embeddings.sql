-- 0008_metric_embeddings.sql
--
-- Adds a sqlite-vec virtual table to back semantic retrieval for
-- category 3 (`metric_definition`). This is the "B3" change — the
-- existing `MetricRetriever` only matched by exact name / substring /
-- whole-token shapes, so a stored `qualified_customer` metric never
-- surfaced for a query like "what percent of qualified customers churn".
--
-- Vector dimensionality MUST match the shared Embedder
-- (`mads.memory.embedding.SentenceTransformerEmbedder` → 384 dims, same
-- as cat 5). The vec table is keyed by metric_memory.rowid (so a plain
-- JOIN connects vectors back to the active metric row) and includes
-- `project_id` as a metadata column so the KNN scan can push the
-- tenancy filter into MATCH rather than post-filtering in Python.
--
-- Lifecycle invariants (enforced at the application layer in
-- `mads/memory/metric_definition.py`):
--   1. INSERT a metric row → INSERT its embedding into `metric_vec`
--      keyed by the row's `rowid` (sqlite ROWID, integer) inside the
--      same `BEGIN IMMEDIATE` transaction.
--   2. CLOSE a metric row (set `valid_to`) → DELETE its embedding from
--      `metric_vec` in the same transaction. We keep the vec table to
--      active rows only — historic versions are not semantically
--      searchable, and this avoids a `WHERE valid_to IS NULL` JOIN
--      filter on every KNN.
--   3. `MetricDefinitionStore(embedder=None)` skips vec writes
--      altogether (graceful degradation for unit tests).
--
-- Backfill: `MetricDefinitionStore.backfill_embeddings()` iterates active
-- rows missing from `metric_vec` and INSERTs their embedding. Called on
-- demand (e.g. by the eval harness migrating existing eval databases),
-- not eagerly at init time.
--
-- PII note: metric names and business definitions are LLM-authored
-- narrative about the dataset semantics, not user records — embedding
-- them is low risk and consistent with cat 5's stance on embedding
-- analyst notes. We document this in the embed callsite as well.

CREATE VIRTUAL TABLE IF NOT EXISTS metric_vec USING vec0(
    rowid INTEGER PRIMARY KEY,
    project_id TEXT,
    embedding float[384]
);
