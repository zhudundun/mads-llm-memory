# Data Quality Issue Memory — Implementation Notes (handoff to Reporter)

Stable contract for category 4 (`data_quality_issue`). Locked for v1.

## File paths

- Migration: `/Users/wang/LLM_memory/infra/migrations/0004_data_quality_memory.sql`
- Store + pydantic model: `/Users/wang/LLM_memory/mads/memory/data_quality.py`
- Tool factory: `/Users/wang/LLM_memory/mads/memory/data_quality_tools.py`
- Re-exported from: `/Users/wang/LLM_memory/mads/memory/__init__.py`
- Tests: `/Users/wang/LLM_memory/tests/test_data_quality.py`
- Demo: `/Users/wang/LLM_memory/eval/demo_data_quality_memory.py`

Shares its SQLite file with cats 1/2/3 (`MADS_DB_PATH` or
`./.mads/memory.sqlite`); `":memory:"` honored. Migration applied
by `init_schema()` on construction (idempotent).

## Identity rule

`(project_id, normalized_table, normalized_column, normalized_issue_type)`.
`user_id` is `NOT NULL` and recorded on every row for audit but does
NOT participate in identity — multiple users in the same project share
one data-quality library.

`normalized_column` is **nullable**: `column_name = None` means the issue
is anchored at the whole-table grain (e.g. row-filter requirements).
`dataset_fingerprint` is stored as advisory metadata only.

User-facing columns (`table_name`, `column_name`, `issue_type`) preserve
the caller's original casing; the three `normalized_*` columns are stored
and used for identity / partial-unique enforcement.

## Normalization (`_normalize_table` / `_normalize_column` / `_normalize_issue_type`)

1. Lowercase.
2. Replace any non-`[0-9a-z]` run with a single `_`.
3. Strip leading/trailing `_`.
4. If result is empty → `ValueError`.

`_normalize_column(None)` passes `None` through unchanged. Examples:
`"Sessions"` → `"sessions"`; `"Temporal Lag"` → `"temporal_lag"`;
`"Row-Filter (required)"` → `"row_filter_required"`.

## Version rule (heart of the store)

Inside one `BEGIN IMMEDIATE` transaction, `upsert(...)` SELECTs the active
row for the identity tuple and classifies the call:

| State of active row vs. inputs | Action | Effect |
|---|---|---|
| no active row | `inserted` | INSERT, `valid_from = now`, `valid_to = NULL` |
| same defs + same metadata | `noop` | nothing written; returns existing row |
| same defs, different metadata | `updated_in_place` | UPDATE `severity` / `confidence` / `dataset_fingerprint` / `updated_at`; same `id` |
| any definitional field differs | `versioned` | UPDATE old `valid_to = now`; INSERT new in same tx |

Definitional fields: `issue_description`, `recommended_handling`.
Metadata fields: `severity`, `confidence`, `dataset_fingerprint`.
`severity` is the team's current assessment, not a definition — same
treatment as cat 3's `confidence`.

The SELECT for the prior active row happens **inside** `BEGIN IMMEDIATE`
(a tightening over cat 3, whose note flagged a TOCTOU window with the
SELECT outside the txn). `now` is captured once in Python before the txn
opens, so on a `versioned` upsert the closed row's `valid_to` exactly
equals the new row's `valid_from` (tested).

The partial unique index `uq_dq_active` is the backstop:

```sql
CREATE UNIQUE INDEX uq_dq_active
    ON data_quality_memory(project_id, normalized_table,
                           COALESCE(normalized_column, ''),
                           normalized_issue_type)
    WHERE valid_to IS NULL;
```

`COALESCE(normalized_column, '')` is load-bearing — without it SQLite's
NULL-distinct UNIQUE semantics would silently permit duplicate table-level
rows. Both column-level and table-level duplicate rejection are tested
via direct SQL.

## Resolution model

`resolve(...)` sets `valid_to = now` on the active row and inserts NO
replacement. The issue exits `list_active_for_table` /
`get_active_for_column` but remains in `history`. Re-adding the same
identity after a resolve starts a **fresh** active row (action ==
`inserted`); it is NOT a continuation of the old chain. Double-resolve is
safe: a second call returns `None` from the store and an error string
from the tool layer.

## Schema tightenings (vs. user-provided columns)

- `user_id`, `project_id`, `table_name`, `issue_type`, `issue_description`,
  and the three `normalized_*` columns: all `NOT NULL`.
- `column_name` / `normalized_column` / `dataset_fingerprint` /
  `recommended_handling`: nullable.
- `severity TEXT NOT NULL DEFAULT 'warning'`,
  `CHECK (severity IN ('info', 'warning', 'critical'))`.
- `confidence REAL NOT NULL DEFAULT 1.0`,
  `CHECK (confidence BETWEEN 0.0 AND 1.0)`.
- `valid_from` / `created_at` / `updated_at`:
  `NOT NULL DEFAULT CURRENT_TIMESTAMP`.

## Tool surface

Factory entrypoint: `make_data_quality_factory(store, user_id, project_id)
-> DataQualityToolFactory`. `.tools()` returns these 6 `@tool` callables:

1. `add_data_quality_issue(table_name, issue_type, issue_description, column_name=None, recommended_handling=None, severity="warning", confidence=1.0, dataset_fingerprint=None) -> str`
2. `get_data_quality_issues_for_column(table_name, column_name) -> str`
3. `list_data_quality_issues_for_table(table_name) -> str`
4. `search_data_quality_issues(query) -> str`
5. `get_data_quality_issue_history(table_name, issue_type, column_name=None) -> str`
6. `resolve_data_quality_issue(table_name, issue_type, column_name=None) -> str`

`user_id` / `project_id` are bound into closures; the LLM never sees them
(tested via `tool.args_schema`). The add tool's return string surfaces
all four upsert action labels (`inserted` / `noop` / `updated_in_place`
/ `versioned`) for analyst visibility.

## Store API

`DataQualityStore` exposes `init_schema()`, `upsert(...)` (returns
`(DataQualityIssue, UpsertAction)`), `resolve(...)` (returns
`DataQualityIssue | None`), `get_active_for_column(...)`,
`list_active_for_table(...)`, `search_active(...)`, `history(...)`, and
`close()`. All accept keyword-only args. Reads return `DataQualityIssue`
(pydantic v2) at the public boundary; rows are plain `sqlite3.Row`
internally. `UpsertAction = Literal["inserted", "versioned",
"updated_in_place", "noop"]`. `Severity = Literal["info", "warning",
"critical"]`.

## Edge cases

- Empty / pure-punctuation `table_name` / `issue_type` /
  `issue_description` → `ValueError`. Invalid `severity` or `confidence`
  outside `[0, 1]` → `ValueError` before the DB; CHECK is a backstop.
- `search_active(query="")` → `[]`; tool layer surfaces an error rather
  than dumping every row. `get_active_for_column` / `history` with a name
  that normalizes to empty return `[]` (do not raise).
- `get_active_for_column` does NOT return table-level (NULL column)
  rows; use `list_active_for_table` for the unified view.
- `list_active_for_table` orders table-level rows first, then by column,
  then by issue_type. `history` orders by `valid_from ASC, created_at ASC`.

## Project-level multi-tenancy

Same-project factories with different `user_id` see the same library
(tested). Different-`project_id` factories are fully isolated (tested).
`user_id` lands on every write so the audit trail survives in `history()`.

## Deferred (not built here)

- `resolution_note` field — explicitly deferred by brief; resolve carries
  no free-text note in v1. Could land as a metadata-only field later.
- A `status` column was rejected — resolution is purely via `valid_to`,
  matching the versioning-is-universal contract in CLAUDE.md.
- PII scrub (`mads/memory/scrub.py` doesn't exist yet); embeddings (lives
  entirely in SQL; pgvector is the `memories` rollup's job).

## Decisions surfaced

1. `DataQualityIssue` is pydantic v2 at the public boundary; rows stay
   `sqlite3.Row`/dict internally.
2. `updated_in_place` keeps original `id`, `valid_from`, `user_id` — a
   metadata bump doesn't rewrite authorship.
3. The prior-row SELECT runs INSIDE `BEGIN IMMEDIATE` (tightening over
   cat 3) so classification can't race a concurrent writer.
4. Resolve uses `valid_to` only — no `status`, no `resolution_note`.
   Re-opening a resolved identity is a fresh INSERT.
5. `search` is case-insensitive substring over five columns:
   `issue_description`, `recommended_handling`, `table_name`,
   `column_name`, `issue_type`. No FTS, no embeddings.
6. Tool layer surfaces all 4 action labels in `add_data_quality_issue`'s
   return string for analyst visibility.
