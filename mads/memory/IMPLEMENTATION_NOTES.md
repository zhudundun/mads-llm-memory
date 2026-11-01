# User-Preference Memory — Implementation Notes (Phase 1 contract)

This file is the public contract Phase 2 (tests + demo) consumes. The schema,
versioning rule, and signatures below are stable for v1.

## Schema (SQLite, verbatim)

```sql
CREATE TABLE user_preference_memory (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    preference_key TEXT NOT NULL,
    preference_value TEXT NOT NULL,
    source TEXT,
    confidence REAL,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE UNIQUE INDEX ix_active_user_pref
    ON user_preference_memory (user_id, preference_key)
    WHERE valid_to IS NULL;
```

All timestamps are UTC ISO-8601 strings, e.g. `2026-06-01T18:42:07+00:00`,
stored as TEXT. Row IDs are 32-char `uuid.uuid4().hex`.

## PreferenceStore — public API

```python
class PreferenceStore:
    def __init__(self, db_path: str | None = None) -> None: ...
    def set(self, user_id: str, key: str, value: str, *, source: str | None = None, confidence: float | None = None) -> str: ...
    def get(self, user_id: str, key: str) -> dict | None: ...
    def list_active(self, user_id: str) -> list[dict]: ...
    def forget(self, user_id: str, key: str) -> bool: ...
    def list_history(self, user_id: str, key: str) -> list[dict]: ...
    def close(self) -> None: ...
```

- `db_path` defaults to `$MADS_DB_PATH` or `./.mads/memory.sqlite`. Pass
  `":memory:"` for an in-memory store. Parent directory is `mkdir -p`'d on
  init for file paths. The table and the partial active-row index are created
  on init via `IF NOT EXISTS`.
- `set` returns the **new** row's id (str). `forget` returns True iff an
  active row was closed. `get` returns a dict with keys `id, user_id,
  preference_key, preference_value, source, confidence, valid_from, valid_to,
  created_at, updated_at` — closed rows have a non-null `valid_to`.
- `list_active` is ordered by `preference_key ASC`; `list_history` is ordered
  by `valid_from ASC` (oldest first).
- `confidence`, if provided, must be in `[0, 1]` — otherwise `set` raises
  `ValueError`.

## LangChain tool factory

```python
def make_preference_tools(store: PreferenceStore, user_id: str) -> list[BaseTool]: ...
```

Returns exactly four `@tool`-decorated callables, in this order, with
`user_id` bound by closure (the LLM never sees it):

```python
remember_preference(key: str, value: str, source: str = "user_explicit", confidence: float = 1.0) -> str
recall_preference(key: str) -> str
list_preferences() -> str
forget_preference(key: str) -> str
```

Tool `.name` attributes are exactly `remember_preference`, `recall_preference`,
`list_preferences`, `forget_preference` (inferred from the function names by
`@tool`).

## Versioning rule (one paragraph)

On every `set(user_id, key, value)`, any active row for `(user_id, key)` (the
one with `valid_to IS NULL`) is closed in the same transaction by stamping
`valid_to = updated_at = now()`, and a brand-new row with a fresh `uuid4().hex`
id and `valid_from = now()`, `valid_to = NULL` is inserted. Rows are
immutable after that close — there is no `superseded_by` column; the history
chain for a `(user_id, preference_key)` is reconstructed by ordering rows by
`valid_from ASC`, where the single active row (at most one, enforced by the
partial unique index) is the last entry with `valid_to IS NULL`. `forget`
closes the active row without inserting a replacement.
