# Metric Definition Memory — Implementation Notes (handoff to Reporter)

Stable contract for category 3 (`metric_definition`). Locked for v1.

## File paths

- Migration: `/Users/wang/LLM_memory/infra/migrations/0003_metric_memory.sql`
- Store + pydantic model: `/Users/wang/LLM_memory/mads/memory/metric_definition.py`
- Tool factory: `/Users/wang/LLM_memory/mads/memory/metric_tools.py`
- Re-exported from: `/Users/wang/LLM_memory/mads/memory/__init__.py`
- Tests: `/Users/wang/LLM_memory/tests/test_metric_definition.py`
- Demo: `/Users/wang/LLM_memory/eval/demo_metric_memory.py`

Shares its SQLite file with cats 1/2 (`MADS_DB_PATH` or
`./.mads/memory.sqlite`); `":memory:"` honored. Migration is applied
by `init_schema()` on construction (idempotent).

## Identity rule

`(project_id, normalized_name)`. `project_id` and `user_id` are both
`NOT NULL`, but `user_id` does **not** participate in identity — it records
who created/last-edited a version. Multiple users in the same project share
one metric library, by design.

`metric_name` keeps the caller's original casing. `normalized_name` is the
identity-bearing column.

## Normalization (`_normalize`)

1. Lowercase.
2. Replace any non-`[0-9a-z]` run with a single `_`.
3. Strip leading/trailing `_`.
4. If result is empty: `ValueError("metric_name must be non-empty after normalization")`.

Examples:
- `"Active User"` → `"active_user"`
- `"Conversion Rate (CR)"` → `"conversion_rate_cr"`
- `"  retention  "` → `"retention"`

## Version rule (heart of the store)

Inside one `BEGIN IMMEDIATE` transaction, `upsert(...)` looks up the active
row for `(project_id, normalized_name)` and classifies the call:

| State of active row vs. inputs | Action | Effect |
|---|---|---|
| no active row | `inserted` | INSERT, `valid_from = now`, `valid_to = NULL` |
| same defs + same metadata | `noop` | nothing written; returns existing row |
| same defs, different metadata | `updated_in_place` | UPDATE `confidence`/`source`/`updated_at`; same `id` |
| any definitional field differs | `versioned` | UPDATE old `valid_to = now`; INSERT new in same tx |

Definitional fields: `business_definition`, `sql_definition`,
`python_definition`, `default_filters`. Metadata fields: `confidence`,
`source`. Both `now` writes inside a `versioned` upsert use the same
timestamp value, so `old.valid_to == new.valid_from` is exact (tested).

The partial unique index `uq_metric_active(project_id, normalized_name)
WHERE valid_to IS NULL` is the backstop — even if someone bypasses
`upsert` and inserts directly, a duplicate active row raises
`sqlite3.IntegrityError` (also tested).

## Schema tightenings (vs. user-provided columns)

- `user_id`, `project_id`, `metric_name`, `business_definition`,
  `normalized_name` all `NOT NULL`.
- `valid_from`, `created_at`, `updated_at` all
  `NOT NULL DEFAULT CURRENT_TIMESTAMP`.
- `confidence REAL NOT NULL DEFAULT 1.0` with
  `CHECK (confidence BETWEEN 0.0 AND 1.0)`.
- Added `normalized_name TEXT NOT NULL` (stored) for the partial unique
  index. `metric_name` preserves the caller's casing.

## Tool surface — exact signatures (verbatim)

The factory entrypoint is:

```python
def make_metric_factory(
    store: MetricDefinitionStore,
    user_id: str,
    project_id: str,
) -> MetricToolFactory: ...
```

`MetricToolFactory.tools()` returns these 5 `@tool`-decorated callables in
this order:

```python
@tool
def add_metric_definition(
    metric_name: str,
    business_definition: str,
    sql_definition: Optional[str] = None,
    python_definition: Optional[str] = None,
    default_filters: Optional[str] = None,
    confidence: float = 1.0,
    source: Optional[str] = None,
) -> str: ...

@tool
def get_metric_definition(metric_name: str) -> str: ...

@tool
def list_metric_definitions() -> str: ...

@tool
def search_metric_definitions(query: str) -> str: ...

@tool
def get_metric_history(metric_name: str) -> str: ...
```

`user_id` and `project_id` are bound into the closures returned by
`tools()`; the LLM never sees them. Tested via `tool.args_schema`.

## Store API

```python
class MetricDefinitionStore:
    def __init__(self, db_path: str | None = None, *,
                 conn: sqlite3.Connection | None = None) -> None: ...
    def init_schema(self) -> None: ...
    def upsert(self, *, user_id, project_id, metric_name,
               business_definition, sql_definition=None,
               python_definition=None, default_filters=None,
               confidence=1.0, source=None,
               ) -> tuple[MetricDefinition, UpsertAction]: ...
    def get_active(self, *, project_id, metric_name) -> MetricDefinition | None: ...
    def list_active(self, *, project_id) -> list[MetricDefinition]: ...
    def search_active(self, *, project_id, query) -> list[MetricDefinition]: ...
    def history(self, *, project_id, metric_name) -> list[MetricDefinition]: ...
    def close(self) -> None: ...
```

`UpsertAction = Literal["inserted", "versioned", "updated_in_place", "noop"]`.

All store reads return `MetricDefinition` (pydantic v2 model) so the public
boundary is typed; rows are still plain `sqlite3.Row` internally to match
cat 1 / 2 conventions.

## Edge cases handled

- Empty / whitespace / pure-punctuation `metric_name` → `ValueError`.
- Empty `business_definition` → `ValueError` (definitional field).
- `confidence` outside `[0, 1]` → `ValueError` before the DB; CHECK is a backstop.
- `search_active(query="")` → `[]`; tool layer surfaces an error rather than
  dumping every row.
- `get_active` / `history` with a name that normalizes-to-empty return
  `None` / `[]` (do not raise) — symmetric with "absent".
- `list_active` orders by `LOWER(metric_name) ASC`; `history` orders by
  `valid_from ASC, created_at ASC` (tiebreaker for same-second writes).

## Project-level multi-tenancy

Two factories with the same `project_id` but different `user_id` see the
same metric library (tested); two factories with different `project_id`
are fully isolated (tested). `user_id` lands on every write so the audit
trail ("who edited v3 of Conversion?") survives in `history()`.

## Deferred (not built here)

- PII scrub: `mads/memory/scrub.py` doesn't exist yet; per brief, flagged
  but not built. Definitional strings here are author-written, not raw row
  contents, so the scrub gate is less urgent than for `analysis_conclusion`.
- Embeddings: this category lives entirely in SQL; pgvector wiring is the
  eventual `memories` rollup's job.

## Decisions surfaced

1. **`MetricDefinition` is a pydantic v2 model** at the public boundary;
   rows stay `sqlite3.Row`/dict internally.
2. **`updated_in_place` keeps original `id`, `valid_from`, `user_id`.** An
   in-place metadata bump doesn't rewrite authorship.
3. **`add_metric_definition` reports the version number** by re-querying
   `history()` after the upsert (one indexed query).
4. **`search` is case-insensitive substring** (no FTS, no embeddings).
5. **Demo lives at `eval/demo_metric_memory.py`** per the brief — cats 1/2
   used the repo root, but the brief is explicit here.
