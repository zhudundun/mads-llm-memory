# DatasetContext

DatasetContext: dual-fingerprint identity + descriptive profile for cross-session dataset recognition.

## Why this exists

Pre-v1.1, `mads` carried a single `compute_fingerprint(df, sample_k=5)` that mixed schema, row-count
log-bucket, and per-column sample values into one hash. That made it unable to recognize a dataset
across snapshots: the same logical table loaded last week and again today returned different
fingerprints whenever the rows or sample values drifted. The demo we ran on snapshot A and B of the
same logical orders table produced content-addressed fingerprints `e4e2d6e7...` and `cb6844a5...` —
two different IDs for one dataset — while the schema-only fingerprint was identical at `9fbf7f32...`
in both cases. v1.1 fixes this by splitting identity into two orthogonal hashes plus a structured
profile, all keyed by the schema-only hash.

## The two fingerprints

`mads/dataset/fingerprint.py` exposes:

- **`exact_fingerprint(file_bytes: bytes) -> str`** — SHA256 of the raw file bytes. Bytes-equal files
  produce identical fingerprints; any byte-level change (a single appended row, a re-saved CSV with
  different line endings, pandas re-export) breaks the match. No parsing required. Returns a 64-char
  lowercase hex digest.
- **`semantic_fingerprint(df: pd.DataFrame) -> str`** — SHA256 of `{sorted column names, dtypes
  aligned to that sorted order}`. Order-invariant in column ordering. Stable across snapshots of the
  same logical dataset (different row counts, different sample values). Returns a 64-char lowercase
  hex digest.
- **`compute_fingerprint`** is a backward-compat alias: `compute_fingerprint is semantic_fingerprint`
  evaluates to `True`. Pre-v1.1 callers (the legacy `dataset_memory` writer, `dataset_tools.py`,
  demos, eval scenarios) transparently switch to schema-only fingerprinting with no code changes.

## `DatasetContext` model

`mads/dataset/profiler.py` declares the structured profile as a pydantic model with exactly 12 fields:

| Field | Type | Notes |
|---|---|---|
| `dataset_name` | `str` | Caller-supplied logical name (e.g. `"orders"`). |
| `exact_fingerprint` | `str` | `exact_fingerprint(file_bytes)`, or `""` if bytes were not provided. |
| `semantic_fingerprint` | `str` | `semantic_fingerprint(df)`. Always populated. |
| `columns` | `list[str]` | Column names in dataframe order. |
| `dtypes` | `dict[str, str]` | `{column_name: str(dtype)}` for every column. |
| `row_count` | `int` | `len(df)`. |
| `primary_key_candidates` | `list[str]` | Columns that are fully unique AND non-null. |
| `join_key_candidates` | `list[str]` | Foreign-key-shaped column names, minus PK candidates. |
| `time_columns` | `list[str]` | Pandas datetime64 dtypes only (no string-date parsing). |
| `numeric_columns` | `list[str]` | Numeric dtypes, excluding bool. |
| `categorical_columns` | `list[str]` | object/string/bool with `nunique(dropna=True) <= 20`. |
| `profile_summary` | `str` | Deterministic 6-line template, prompt-injectable. |

All list fields preserve dataframe column ordering.

## `profile(df, *, dataset_name, file_bytes=None) -> DatasetContext`

Pure-Python, deterministic, no LLM call, no API key needed. Computes the dual fingerprints, runs
each heuristic against the dataframe, and renders the summary template. `file_bytes` is optional —
when omitted, `exact_fingerprint` is `""` and the caller can re-profile later once bytes are
available without rerunning the heuristics.

## Column-role heuristics

All rules are conservative; overlap between numeric and categorical is allowed by the spec but the
v1.1 implementation excludes already-numeric columns from `categorical_columns` (this is pinned by
`test_numeric_and_categorical_overlap_allowed_for_low_card_int`).

- **`primary_key_candidates`** — column is fully unique (`nunique(dropna=False) == len(df)`) AND has
  zero nulls (`isna().sum() == 0`). Empty dataframes yield `[]`.
- **`join_key_candidates`** — column name matches the regex `(_id|_key|Id|Key)$` (covers snake_case
  and camelCase suffixes); OR the name ends in `_id` / `_key` AND `nunique(dropna=True) > 0.5 *
  len(df)` (high cardinality). PK candidates are excluded.
- **`time_columns`** — `pd.api.types.is_datetime64_any_dtype(col)` is true (or `str(dtype)` starts
  with `"datetime"`). NO string-date parsing — `"2026-01-01"` as `object` dtype does not match.
- **`numeric_columns`** — `pd.api.types.is_numeric_dtype(col)` AND NOT `is_bool_dtype(col)`. Booleans
  are routed to categorical, never numeric.
- **`categorical_columns`** — object / string / bool dtype AND `nunique(dropna=True) <= 20`. Already
  in `numeric_columns` or `time_columns`? Skipped.

## `profile_summary` template

Six lines, no trailing newline. Verbatim from `mads/dataset/profiler.py::_render_summary`:

```
Dataset: {dataset_name}
{row_count:,} rows, {n_cols} columns ({n_numeric} numeric, {n_time} datetime, {n_categorical} categorical).
PK candidates: {pk or "none"}.
Join keys: {join_keys or "none"}.
Time columns: {time_columns or "none"}.
exact_fingerprint: {exact_fp[:8]}...  semantic_fingerprint: {sem_fp[:8]}...
```

Example (orders dataset, 1,000 rows, no file_bytes):

```
Dataset: orders
1,000 rows, 4 columns (3 numeric, 1 datetime, 0 categorical).
PK candidates: order_id.
Join keys: customer_id.
Time columns: ts.
exact_fingerprint: ...  semantic_fingerprint: 9fbf7f32...
```

When `file_bytes` is absent, the `exact_fingerprint` preview is just `"..."` (empty prefix +
ellipsis); the summary still renders.

## Storage: `dataset_context_memory`

New SQLite table created by `DatasetSchemaStore.__init__`. Distinct from `dataset_memory` — this one
is **not versioned**; `register_context` upserts in place.

```sql
CREATE TABLE IF NOT EXISTS dataset_context_memory (
    semantic_fingerprint TEXT PRIMARY KEY,
    dataset_name TEXT NOT NULL,
    exact_fingerprint TEXT,
    columns TEXT NOT NULL,                 -- JSON list[str]
    dtypes TEXT NOT NULL,                  -- JSON dict[str, str]
    row_count INTEGER NOT NULL,
    primary_key_candidates TEXT NOT NULL,  -- JSON list[str]
    join_key_candidates TEXT NOT NULL,     -- JSON list[str]
    time_columns TEXT NOT NULL,            -- JSON list[str]
    numeric_columns TEXT NOT NULL,         -- JSON list[str]
    categorical_columns TEXT NOT NULL,     -- JSON list[str]
    profile_summary TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT (CURRENT_TIMESTAMP),
    updated_at TIMESTAMP NOT NULL DEFAULT (CURRENT_TIMESTAMP)
);
```

JSON-encoded columns are written via `json.dumps(..., sort_keys=True, default=str)` and rehydrated
on read. `exact_fingerprint` is nullable — when the source `DatasetContext.exact_fingerprint` is the
empty string `""`, it is coerced to SQL `NULL`. The primary key is `semantic_fingerprint`, NOT
`(user_id, semantic_fingerprint)` — the context describes the logical dataset, not the user's view
of it. Tenancy lives on `dataset_memory`, which references the same `semantic_fingerprint`.

`dataset_memory` itself gains two new nullable columns: `exact_fingerprint` and
`semantic_fingerprint`, plus partial indexes `ix_dataset_exact_fp` and `ix_dataset_semantic_fp` for
the cross-session lookup path. The legacy `dataset_fingerprint` column is retained and now carries
the same value as `semantic_fingerprint` post-alias.

## `DatasetSchemaStore` extensions

Three new methods on the existing store class. All other methods (`register_dataset`,
`register_column`, `get_dataset`, etc.) are unchanged — `register_dataset` now accepts an optional
`file_bytes` kwarg that populates the new `exact_fingerprint` column.

- **`register_context(ctx: DatasetContext) -> None`** — in-place upsert keyed on
  `ctx.semantic_fingerprint`. Second call with a new `dataset_name` or `file_bytes` overwrites
  every field; the row count stays at one.
- **`get_context(semantic_fingerprint: str) -> Optional[DatasetContext]`** — fetch by SF; returns
  the hydrated pydantic model or `None`.
- **`lookup_dataset(*, user_id, df, table_name, file_bytes=None) -> Optional[dict]`** — cross-session
  lookup. Computes `semantic_fingerprint(df)` always; when `file_bytes` is supplied, also computes
  `exact_fingerprint(file_bytes)` and queries `dataset_memory` by `(user_id, exact_fingerprint,
  table_name)` first. Falls back to `(user_id, semantic_fingerprint, table_name)`. Returns
  `{"matched_by": "exact"|"semantic", "row": <dataset_memory row dict>}` or `None`. User scoping is
  never relaxed.

## Cross-session recognition flow

The new surface is consumed by the wrapper / `router/context_builder.py`, NOT exposed as LLM tools.
The intended call sequence:

- User uploads a file. Wrapper reads the bytes and parses to a `df`.
- Wrapper calls `store.lookup_dataset(user_id=..., df=df, table_name=..., file_bytes=...)`.
- `matched_by == "exact"` -> same file as a prior session; reuse the prior `dataset_memory` row
  silently.
- `matched_by == "semantic"` -> same logical dataset, different snapshot; ask the user whether to
  apply prior memory (grain, PK, known issues) to this session.
- `lookup_dataset` returned `None` -> new dataset; call `profile(df, dataset_name=...,
  file_bytes=...)` then `store.register_context(ctx)` and `store.register_dataset(...,
  file_bytes=...)`.
- The `profile_summary` string is appropriate for prompt injection on recognized datasets so the LLM
  sees the schema overview without re-reading the file.

## What's NOT in v1.1

- No new LLM tool surface. `mads/memory/dataset_tools.py` is unchanged (still 6 `@tool` callables);
  the profiler and context store are Python-API only.
- No string-date parsing in `time_columns`. A column of `"2026-01-01"` strings will not be detected
  as time.
- No LLM-assisted heuristics. PK / join-key / categorical detection is pure pandas + name regex.
- No drop of the legacy `dataset_fingerprint` column. It is now equal to `semantic_fingerprint` for
  every v1.1+ row and kept for backward compatibility.
- No LLM-generated variant of `profile_summary`. The template is the only renderer.

## Tests

```bash
.venv/bin/pytest tests/test_fingerprint.py tests/test_dataset_profiler.py tests/test_dataset_schema.py -q
```

62 tests across the three files (11 + 18 + 33). Coverage areas:

- `test_fingerprint.py` — exact fingerprint determinism, semantic fingerprint row-count invariance
  and column-order invariance, alias identity check.
- `test_dataset_profiler.py` — DatasetContext 12-field shape, JSON round-trip, each heuristic, the
  bool-to-categorical routing, low-card int overlap pin, empty-frame edge case, 6-line summary
  template.
- `test_dataset_schema.py` — original schema-store coverage plus 9 new tests for `register_context`
  upsert, `get_context`, all four `lookup_dataset` branches (exact, semantic fallback when bytes
  differ, semantic when no bytes, miss), and dual-column population on `register_dataset(file_bytes=)`.

## Related

- `/Users/wang/LLM_memory/docs/architecture.md` — overall v1 + v1.1 architecture.
- `/Users/wang/LLM_memory/docs/memory_design.md` — the 5-category memory subsystem.
- `/Users/wang/LLM_memory/V1_DESIGN_BRIEF.md` — original design brief for the memory layer.
- `/Users/wang/LLM_memory/mads/memory/IMPLEMENTATION_NOTES_DATASET.md` — cat-2 contract; the v1.1
  addendum at the bottom is the authoritative description of the new surface.
