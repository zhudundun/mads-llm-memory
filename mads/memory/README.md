# User-Preference Memory

This is the first of the 8 memory categories defined for `mads`
(`user_preference`; see `CLAUDE.md` for the full list). It stores per-user
key/value preferences in a SQLite table with append-only versioning: every
write closes the prior active row and inserts a new one, so history is
preserved. It does **not** compute embeddings, does **not** rank or score
retrieval, and does **not** mine preferences from conversation on the LLM
side. v1 is explicit set/get only — the LLM writes a preference iff it
calls `remember_preference` itself.

## Status

| Component | File | Status |
|---|---|---|
| store | `mads/memory/user_preference.py` | implemented |
| tools | `mads/memory/preference_tools.py` | implemented |
| tests | `tests/test_user_preference.py` | 14/14 PASS |
| demo  | `demo_preferences.py` | PASS (transcript: `demo_preferences.out`) |

## Schema

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

`id` is a 32-character `uuid.uuid4().hex`. All four timestamp columns hold
UTC ISO-8601 strings (e.g. `2026-06-01T21:56:55+00:00`) — stored as TEXT
since SQLite has no native timestamp type, but parseable directly by
`datetime.fromisoformat`. The partial unique index on
`(user_id, preference_key) WHERE valid_to IS NULL` enforces at-most-one
active row per `(user, key)` at the database level — concurrent writes
cannot race two active rows into existence.

## Versioning semantics

Every `set(user_id, key, value)` runs inside a single `BEGIN IMMEDIATE`
transaction that:

1. UPDATEs any active row for `(user_id, key)` — stamps
   `valid_to = updated_at = now()`.
2. INSERTs a brand-new row with a fresh `uuid4().hex`, `valid_from = now()`,
   `valid_to = NULL`.

Rows are immutable after close. There is no `superseded_by` column — the
history chain for `(user_id, preference_key)` is reconstructed by ordering
rows by `valid_from ASC`; the single open row (with `valid_to IS NULL`) is
the current value. `forget(user_id, key)` closes the active row but does
not insert a replacement.

Example timeline for `(alice, chart.library)`:

```
t0  set "plotly"                       -> row A: valid_from=t0, valid_to=NULL
t1  set "plotly + matplotlib ..."      -> row A: valid_to=t1   (closed)
                                          row B: valid_from=t1, valid_to=NULL
t2  forget                             -> row B: valid_to=t2   (closed, no new row)
```

After `t2`, `get` returns `None`; `list_history` returns `[A, B]` ordered ASC.

## Configuration

Single env var: `MADS_DB_PATH` — path to the SQLite file. Default
`./.mads/memory.sqlite` (parent dir is `mkdir -p`'d on store init).
Pass `db_path=":memory:"` to `PreferenceStore(...)` for an in-memory store
(used by every test).

## Python API

```python
class PreferenceStore:
    def __init__(self, db_path: str | None = None) -> None:
        """Open (or create) the SQLite store and ensure the table/index exist."""

    def set(self, user_id: str, key: str, value: str, *,
            source: str | None = None,
            confidence: float | None = None) -> str:
        """Set a preference, superseding any active row for (user_id, key).
        Returns the new row's UUID-hex id. Raises ValueError if confidence is
        provided and is not in [0, 1]."""

    def get(self, user_id: str, key: str) -> dict | None:
        """Return the active row dict for (user_id, key), or None."""

    def list_active(self, user_id: str) -> list[dict]:
        """Return all active rows for user_id, ordered by preference_key ASC."""

    def list_history(self, user_id: str, key: str) -> list[dict]:
        """Return all rows (active+closed) for (user_id, key), oldest first."""

    def forget(self, user_id: str, key: str) -> bool:
        """Close the active row for (user_id, key). Returns True iff a row
        was closed. No replacement is inserted."""

    def close(self) -> None:
        """Close the underlying SQLite connection."""
```

Each returned row dict has keys: `id, user_id, preference_key,
preference_value, source, confidence, valid_from, valid_to, created_at,
updated_at`.

Minimal usage:

```python
from mads.memory import PreferenceStore

store = PreferenceStore(":memory:")
store.set("alice", "chart.library", "plotly")
store.get("alice", "chart.library")["preference_value"]   # "plotly"

store.set("alice", "chart.library", "plotly + matplotlib")
store.get("alice", "chart.library")["preference_value"]   # superseded

history = store.list_history("alice", "chart.library")
# len(history) == 2; history[0]['valid_to'] is set, history[-1]['valid_to'] is None
store.close()
```

## LangChain tools API

```python
def make_preference_tools(store: PreferenceStore, user_id: str) -> list[BaseTool]: ...
```

Returns exactly four `@tool`-decorated callables (LangChain `BaseTool`s).
`user_id` is captured in the closure so the LLM never sees it.

| Tool name | Signature | Returns |
|---|---|---|
| `remember_preference` | `(key: str, value: str, source: str = "user_explicit", confidence: float = 1.0) -> str` | `"Saved preference '<key>' = '<value>' (source=..., confidence=..., id=...)."` |
| `recall_preference`   | `(key: str) -> str` | `"Your preference for '<key>' is: <value>."` or `"No preference set for '<key>'."` |
| `list_preferences`    | `() -> str` | `"No preferences set yet."` or `"You have N preferences:\n- key: value (source=..., confidence=...)\n- ..."` |
| `forget_preference`   | `(key: str) -> str` | `"Forgot preference '<key>'."` or `"No active preference for '<key>'."` |

Wiring:

```python
from mads.memory import PreferenceStore, make_preference_tools
store = PreferenceStore()                      # or PreferenceStore(":memory:")
tools = make_preference_tools(store, user_id="alice")
# Bind to an LLM:
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o").bind_tools(tools)
# The LLM never sees user_id — it's baked into the closure.
```

## Running the tests

```bash
cd /Users/wang/LLM_memory && .venv/bin/python -m pytest tests/test_user_preference.py -v --tb=short
```

`pytest` is installed in `.venv` (Python 3.13.2, pytest 9.0.3). All 14
tests use a `:memory:` SQLite fixture — no filesystem is touched.

## Running the demo

```bash
cd /Users/wang/LLM_memory && .venv/bin/python demo_preferences.py > demo_preferences.out 2>&1
```

The demo writes a SQLite file at `/Users/wang/LLM_memory/.demo_preferences.sqlite`
(removed at start) and exercises, in order:

- 5 prefs remembered for `alice`
- `list_preferences` shows all 5 in key-ASC order
- `recall_preference` hits and misses
- Supersession of `chart.library` and `recall` returning the NEW value
- `list_history("alice", "chart.library")` showing 2 rows (closed + open)
- `forget_preference` on `sql.show_assumptions`, then a second forget
  returning the "no active preference" message
- Per-user scoping: a fresh tool set built for `bob` reports
  `"No preferences set yet."`

See `demo_preferences.out` for the captured transcript.

## Known limitations

- `preference_value` is a single TEXT column — structured values (lists,
  dicts) must be JSON-stringified by the caller. There is no JSON
  validation at the store layer.
- No embeddings or semantic retrieval. `recall_preference` is an exact-key
  lookup; there is no fuzzy/similar-key matching. Ranked retrieval is a
  later memory category's job (or a v2 extension to this one).
- No LLM-side inference loop. Preferences are written only when the LLM
  explicitly calls `remember_preference`. A post-turn extractor that mines
  preferences from conversation is out of scope for v1.
- No ADS wiring in this pass. `make_preference_tools(...)` is not yet
  bound to any ADS agent — that's the next concrete step.
- SQLite only. The Postgres + pgvector adapter referenced in `CLAUDE.md`
  is future work; for v1 the store is single-file SQLite.
- `confidence` is stored but not used by any retrieval logic — it is
  metadata for future ranking. The store validates the `[0, 1]` range and
  nothing more.

## How this fits

This module is category 1 of 8 (`user_preference`) under `CLAUDE.md`'s
memory taxonomy. The same shape — a `*Store` class plus a `make_*_tools`
factory that closes over `(store, user_id)` — is the template intended to
repeat for the other 7 categories: `schema_fact`, `metric_definition`,
`data_quality_issue`, `analysis_conclusion`, `workflow_recipe`,
`pipeline_decision`, `experiment_record`. The immediate next step is to
wire `make_preference_tools(...)` into one ADS agent (via
`mads/adapters/ads.py`) so we can demo a live remember/recall loop end to
end.

## File map

```
mads/memory/
  __init__.py                  # re-exports PreferenceStore, make_preference_tools
  user_preference.py           # PreferenceStore: SQLite store + versioning
  preference_tools.py          # make_preference_tools: 4 @tool callables, user_id via closure
  IMPLEMENTATION_NOTES.md      # Phase 1 contract (schema, signatures, versioning rule)
  README.md                    # this file

tests/
  __init__.py                  # empty, makes tests/ a package
  test_user_preference.py      # 14 tests: 8 store-level, 6 tool-level (all :memory:)
  results/
    test_user_preference.md    # tester's result summary (req -> test map)
    test_user_preference.out   # raw pytest stdout
    test_user_preference.err   # raw pytest stderr (empty)

demo_preferences.py            # end-to-end CLI demo (no LLM; direct tool.invoke)
demo_preferences.out           # captured transcript of the demo run
```
