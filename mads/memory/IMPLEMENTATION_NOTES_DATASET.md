# Dataset/Schema Memory — Implementation Notes (Phase 1 contract)

This file is the public contract Phase 2 (tests + demo) consumes. The schema,
versioning rule, and signatures below are stable for v1.

## File paths

- Store: `/Users/wang/LLM_memory/mads/memory/dataset_schema.py`
- Tool factory: `/Users/wang/LLM_memory/mads/memory/dataset_tools.py`
- Re-exported from: `/Users/wang/LLM_memory/mads/memory/__init__.py`
- This notes file: `/Users/wang/LLM_memory/mads/memory/IMPLEMENTATION_NOTES_DATASET.md`

The SQLite DB file is shared with `PreferenceStore` (same `MADS_DB_PATH` /
`./.mads/memory.sqlite` default). Special path `":memory:"` is honored.

## Schema (SQLite, verbatim)

```sql
CREATE TABLE dataset_memory (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    project_id TEXT,
    dataset_fingerprint TEXT,
    dataset_name TEXT,
    table_name TEXT,
    grain TEXT,
    primary_key TEXT,           -- JSON list[str]
    join_keys TEXT,             -- JSON list[{"table": "...", "left": [...], "right": [...]}]
    time_columns TEXT,          -- JSON list[str]
    target_columns TEXT,        -- JSON list[str]
    known_issues TEXT,          -- JSON list[str]
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE column_memory (
    id TEXT PRIMARY KEY,
    dataset_fingerprint TEXT,
    table_name TEXT,
    column_name TEXT,
    semantic_type TEXT,
    business_meaning TEXT,
    pii_level TEXT,             -- 'none' | 'low' | 'medium' | 'high'
    example_values TEXT,        -- JSON list
    null_behavior TEXT,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE UNIQUE INDEX ix_active_dataset
  ON dataset_memory(user_id, dataset_fingerprint, table_name)
  WHERE valid_to IS NULL;

CREATE UNIQUE INDEX ix_active_column
  ON column_memory(dataset_fingerprint, table_name, column_name)
  WHERE valid_to IS NULL;
```

Timestamps are UTC ISO-8601 strings (e.g. `2026-06-01T18:42:07+00:00`), stored
as TEXT. Row IDs are 32-char `uuid.uuid4().hex`. JSON columns are written via
`json.dumps(..., sort_keys=True)` and rehydrated on read.

**`column_memory` intentionally has no `user_id`** — column facts are a
property of the data (keyed by fingerprint), not the user. `dataset_memory`
*is* user-scoped.

## Identity tuples (active-row uniqueness)

- `dataset_memory`: `(user_id, dataset_fingerprint, table_name)` — one active
  row max, enforced by `ix_active_dataset`.
- `column_memory`: `(dataset_fingerprint, table_name, column_name)` — one
  active row max, enforced by `ix_active_column`.

## Fingerprint formula (one paragraph)

`compute_fingerprint(df, sample_k=5)` returns a sha256 hex digest of a JSON
payload (`sort_keys=True`, UTF-8) containing: (1) the dataframe's column
names sorted alphabetically; (2) the per-column dtypes aligned to that
sorted order; (3) a row-count log-bucket = `"0"` for an empty frame else
`str(int(math.log10(len(df))))`; (4) per-column samples = the first
`sample_k` entries of `sorted(stringified(unique non-null values))`. This
makes the fingerprint **order-invariant** in column ordering and stable
across row reshuffles / minor row-count drift within the same log-bucket.

## Versioning rule (one paragraph)

On every `register_dataset(...)` and `register_column(...)`, the prior active
row for the identity tuple (if any) is closed in the same `BEGIN IMMEDIATE`
transaction by stamping `valid_to = updated_at = now()`, and a brand-new row
with a fresh `uuid4().hex` id and `valid_from = now()`, `valid_to = NULL` is
inserted. The new row carries the **merged** state: caller-supplied non-None
fields win; None means "keep prior". Rows are immutable after close — there
is no `superseded_by` column; history for an identity tuple is reconstructed
by ordering rows on `(identity tuple, valid_from ASC)`, with the at-most-one
active row being the trailing entry where `valid_to IS NULL`.
`forget_dataset` / `forget_column` close the active row without inserting a
replacement.

## `DatasetSchemaStore` — public API

```python
class DatasetSchemaStore:
    def __init__(self, db_path: str | None = None) -> None: ...

    # Dataset side
    def register_dataset(self, *, user_id: str, df: pd.DataFrame,
                         dataset_name: str, table_name: str,
                         project_id: str | None = None,
                         grain: str | None = None,
                         primary_key: list[str] | None = None,
                         join_keys: list[dict] | None = None,
                         time_columns: list[str] | None = None,
                         target_columns: list[str] | None = None,
                         known_issues: list[str] | None = None) -> tuple[str, str]:
        """Returns (new_id, fingerprint)."""

    def get_dataset(self, *, user_id: str, fingerprint: str,
                    table_name: str) -> dict | None: ...
    def list_datasets(self, user_id: str) -> list[dict]: ...
    def forget_dataset(self, *, user_id: str, fingerprint: str,
                       table_name: str) -> bool: ...
    def list_dataset_history(self, *, user_id: str, fingerprint: str,
                             table_name: str) -> list[dict]: ...

    # Column side
    def register_column(self, *, fingerprint: str, table_name: str,
                        column_name: str,
                        semantic_type: str | None = None,
                        business_meaning: str | None = None,
                        pii_level: str | None = None,
                        example_values: list | None = None,
                        null_behavior: str | None = None) -> str:
        """Returns new row id. Raises ValueError on bad pii_level."""

    def get_column(self, *, fingerprint: str, table_name: str,
                   column_name: str) -> dict | None: ...
    def list_columns(self, *, fingerprint: str,
                     table_name: str) -> list[dict]: ...
    def forget_column(self, *, fingerprint: str, table_name: str,
                      column_name: str) -> bool: ...
    def list_column_history(self, *, fingerprint: str, table_name: str,
                            column_name: str) -> list[dict]: ...

    def close(self) -> None: ...
```

- `db_path` defaults to `$MADS_DB_PATH` or `./.mads/memory.sqlite` (same
  file as `PreferenceStore`). Pass `":memory:"` for an in-memory store. The
  tables and partial indexes are created on init via `IF NOT EXISTS`.
- All `list_*` methods return rows sorted: `list_datasets` by
  `(dataset_name, table_name)`, `list_columns` by `column_name`, both history
  methods by `valid_from ASC`.
- `pii_level`, if provided to `register_column`, must be in
  `{'none','low','medium','high'}`; otherwise `ValueError`.
- JSON-encoded columns (`primary_key`, `join_keys`, `time_columns`,
  `target_columns`, `known_issues`, `example_values`) are returned
  pre-parsed (as Python lists/dicts) on every read.

## LangChain tool factory

```python
def make_dataset_tools(
    store: DatasetSchemaStore,
    df: pd.DataFrame,
    *,
    dataset_name: str,
    table_name: str,
    user_id: str,
    project_id: str | None = None,
) -> list[BaseTool]: ...
```

The factory computes the fingerprint **once** from `df` and binds
`(store, df, fingerprint, dataset_name, table_name, user_id, project_id)`
into every tool's closure. The LLM never sees these. Returns six
`@tool`-decorated callables in this order, with tool `.name` matching the
function name:

```python
register_dataset(grain: str = None,
                 primary_key: list[str] = None,
                 join_keys: list[dict] = None,
                 time_columns: list[str] = None,
                 target_columns: list[str] = None,
                 known_issues: list[str] = None) -> str
recall_dataset() -> str
list_datasets() -> str
register_column(column_name: str,
                semantic_type: str = None,
                business_meaning: str = None,
                pii_level: str = None,
                example_values: list = None,
                null_behavior: str = None) -> str
recall_column(column_name: str) -> str
list_columns() -> str
```

**Bound (closure) parameters per tool** — NOT visible to the LLM:

- All six tools: `store`, `fingerprint`, `table_name`.
- `register_dataset`, `recall_dataset`, `list_datasets`, `register_column`:
  additionally `user_id` (datasets are user-scoped; for columns it's used
  only by the implicit `dataset_name`/`project_id` pass-through on the
  dataset side).
- `register_dataset`, `recall_dataset`, `list_datasets`: also `dataset_name`,
  `project_id`, and `df`.
- `register_column`: also `bound_columns` (the set of `df.columns`) for
  validating `column_name`.

**Validation in tools:**

- `register_column` validates `column_name` against the bound df's columns
  and returns an error string (not an exception) if absent — this is the
  more useful surface for an LLM.
- `register_column` returns the `ValueError` message as a string on bad
  `pii_level`.

## Decisions surfaced (anything the prompt left open)

1. **`list_datasets` sort order**: spec said "all active datasets for this
   user, multi-line, empty case message"; I sorted by
   `(dataset_name, table_name)` for stable output. (Method on the store
   matches; tool just renders that.)
2. **Empty-frame fingerprint**: when `len(df) == 0`, `row_bucket = "0"` to
   avoid `log10(0)`. This means an empty frame with the same columns/dtypes
   gets the same fingerprint as a 1-row frame; acceptable since fingerprints
   are deliberately coarse.
3. **JSON dump uses `default=str`** so pandas/numpy scalars in
   `example_values` or fingerprint samples serialize without raising.
4. **Tool error surface**: `register_column` returns an error *string* (not
   raise) for both unknown column names and bad `pii_level` — the prompt
   said either was fine but a string is more useful to the LLM. The store
   layer still raises `ValueError` for bad `pii_level`.
5. **`recall_dataset` / `recall_column` empty case** wording matches the
   spec verbatim where given; `list_columns` empty case is
   `"No column memory for table '{table_name}'."` (spec only specified
   `list_datasets` empty wording).
6. **Tool docstrings** were kept short and action-oriented because the LLM
   reads them as tool descriptions.

## v1.1 addendum — dataset profiler + dual fingerprint

`v1.1` adds cross-session dataset recognition. The original cat-2 contract
above (versioning, identity tuples, tool factory, etc.) is unchanged.

**New module** `mads/dataset/profiler.py`:

- `DatasetContext` (pydantic model, 12 fields): `dataset_name`,
  `exact_fingerprint`, `semantic_fingerprint`, `columns`, `dtypes`,
  `row_count`, `primary_key_candidates`, `join_key_candidates`,
  `time_columns`, `numeric_columns`, `categorical_columns`,
  `profile_summary`.
- `profile(df, *, dataset_name, file_bytes=None) -> DatasetContext` —
  deterministic, pure-Python (no LLM). `file_bytes` is optional; when
  omitted, `exact_fingerprint` is `""`.
- Conservative heuristics: PK = unique-and-no-nulls; join keys = name
  ends in `_id`/`_key`/`Id`/`Key` OR (high-card AND `_id`/`_key`
  suffix), minus PKs; time = pandas datetime64 dtype only;
  numeric = numeric AND not bool; categorical = object/string/bool
  with `nunique(dropna=True) <= 20`.
- Overlap allowed (e.g. low-card int may appear in both numeric and
  categorical) — the current implementation puts low-card ints in
  numeric only; the test
  `test_numeric_and_categorical_overlap_allowed_for_low_card_int` pins
  this to prevent silent drift.
- `profile_summary` is a deterministic 6-line template; the dual
  fingerprint preview line uses the first 8 hex chars of each fingerprint
  (or `"..."` if empty).

**Dual fingerprint** (`mads/dataset/fingerprint.py`):

- `exact_fingerprint(file_bytes)` — sha256 of raw bytes.
- `semantic_fingerprint(df)` — sha256 of sorted columns + dtypes,
  row-count-invariant. Order-invariant in column ordering.
- `compute_fingerprint = semantic_fingerprint` (backward-compat alias).
  The pre-v1.1 row-bucket + per-column-sample formula is REMOVED;
  legacy callers transparently get schema-only fingerprinting.

**Store extensions** (`mads/memory/dataset_schema.py`):

- `dataset_memory` gets two new nullable columns: `exact_fingerprint`,
  `semantic_fingerprint`. Pre-v1.1 rows (none in production) will have
  both NULL. New rows always populate `semantic_fingerprint`;
  `exact_fingerprint` is populated only when caller passes `file_bytes`.
- New non-unique index `ix_dataset_semantic_fp` on
  `(user_id, semantic_fingerprint, table_name) WHERE valid_to IS NULL`
  plus `ix_dataset_exact_fp` on `(user_id, exact_fingerprint) WHERE
  valid_to IS NULL` to speed the cross-session lookup.
- New table `dataset_context_memory` keyed by `semantic_fingerprint`
  (PRIMARY KEY). Stores the `DatasetContext` as JSON columns + scalar
  fields. **No versioning** — the context is a current-snapshot of the
  logical dataset; re-registering UPDATEs in place. Distinct from
  `dataset_memory`'s versioned identity-tuple model.
- `register_dataset(..., file_bytes=None)` — new kwarg; populates
  `exact_fingerprint`. Return value unchanged: `(new_id,
  semantic_fingerprint)`. The legacy `dataset_fingerprint` column
  continues to receive the same value (semantic_fingerprint of the df)
  so existing tests round-trip cleanly.
- `register_context(ctx: DatasetContext) -> None` — upsert.
- `get_context(semantic_fingerprint) -> DatasetContext | None`.
- `lookup_dataset(*, user_id, df, table_name, file_bytes=None) -> dict |
  None` — cross-session lookup. Returns
  `{"matched_by": "exact"|"semantic", "row": <dataset_memory dict>}`.
  Exact match tried first when `file_bytes` supplied; semantic fallback
  otherwise. Never relaxes user-scoping.

**LLM-visible tool surface** is intentionally **unchanged** in v1.1.
`dataset_tools.py` was not touched. The profiler + context store is a
Python-API extension consumed (in a future phase) by the router /
context builder, not by direct LLM tool calls.
