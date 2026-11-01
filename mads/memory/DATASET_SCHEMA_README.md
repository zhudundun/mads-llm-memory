# Dataset / Schema Memory

Second of the 8 memory categories defined for `mads` (see `CLAUDE.md`),
after `user_preference` (see `mads/memory/README.md`). Stores what the
agent has learned about specific datasets — grain, primary key, join
keys, time/target columns, known issues — plus per-column semantic facts
(`semantic_type`, `business_meaning`, `pii_level`, `example_values`,
`null_behavior`). Datasets are identified by a **content-addressed
fingerprint** of the DataFrame, not by filename, so a renamed/re-saved
CSV with the same content still binds to the same memory. Versioning
follows the same close-and-insert pattern as `user_preference`; both
categories share one `MADS_DB_PATH` SQLite file by default.

## Status

| Component | File | Status |
|---|---|---|
| store | `mads/memory/dataset_schema.py` | implemented |
| fingerprint | `mads/memory/dataset_schema.py` (`compute_fingerprint`) | implemented |
| tools | `mads/memory/dataset_tools.py` | implemented |
| tests | `tests/test_dataset_schema.py` | 24/24 PASS |
| demo  | `demo_dataset_schema.py` | 17/17 PASS (transcript: `demo_dataset_schema.out`) |

## Schemas

```sql
CREATE TABLE IF NOT EXISTS dataset_memory (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    project_id TEXT,
    dataset_fingerprint TEXT,
    dataset_name TEXT,
    table_name TEXT,
    grain TEXT,
    primary_key TEXT,
    join_keys TEXT,
    time_columns TEXT,
    target_columns TEXT,
    known_issues TEXT,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS column_memory (
    id TEXT PRIMARY KEY,
    dataset_fingerprint TEXT,
    table_name TEXT,
    column_name TEXT,
    semantic_type TEXT,
    business_meaning TEXT,
    pii_level TEXT,
    example_values TEXT,
    null_behavior TEXT,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS ix_active_dataset
    ON dataset_memory (user_id, dataset_fingerprint, table_name)
    WHERE valid_to IS NULL;

CREATE UNIQUE INDEX IF NOT EXISTS ix_active_column
    ON column_memory (dataset_fingerprint, table_name, column_name)
    WHERE valid_to IS NULL;
```

`valid_from` / `valid_to` are the two versioning columns required by the
universal contract in `CLAUDE.md` (definition versioning applies to every
memory category): writes are append-only, the prior active row is closed
by stamping `valid_to = now()`, and a new row with `valid_to = NULL` is
inserted.

The partial unique indexes enforce **at most one active row per identity
tuple** — `(user_id, fingerprint, table_name)` for datasets,
`(fingerprint, table_name, column_name)` for columns. Concurrent writers
cannot race two open rows into existence.

`column_memory` deliberately has **no `user_id`**: column-level semantics
are a property of the data, not of the user who happened to observe it.
This keeps column facts shareable across users for the same fingerprint.
Trade-off: a multi-tenant deployment where tenants must keep column
semantics private from each other would need a per-tenant namespace or a
migration adding `tenant_id`.

## Dataset fingerprint

`compute_fingerprint` is a deterministic sha256 hex digest of a JSON
payload (`sort_keys=True`, UTF-8) containing:

1. The DataFrame's column names, sorted alphabetically.
2. The per-column dtypes (stringified), aligned to that sorted order.
3. A row-count log-bucket: `"0"` for an empty frame, else
   `str(int(math.log10(len(df))))`.
4. Per-column samples: the first `sample_k` (default 5) entries of
   `sorted(stringified(unique non-null values))`.

**Order-invariant** w.r.t. column ordering and row ordering: reversing
the column list, shuffling rows, or re-saving a CSV under a different
filename all produce the same fingerprint. This is how the store
recognises "the same dataset" across sessions.

**What does change the fingerprint:** adding/removing a column, changing
a column's dtype, crossing a log10 row-count bucket boundary (e.g. 99
rows -> bucket "1", 100 rows -> bucket "2"), or changing the sorted
top-K unique stringified values of any column.

The implementer chose `row_bucket = "0"` for empty frames to dodge
`log10(0)`. Consequence: an empty frame and a 1-row frame with the same
columns/dtypes share a fingerprint — acceptable since fingerprints are
deliberately coarse.

```python
def compute_fingerprint(df: pd.DataFrame, *, sample_k: int = 5) -> str: ...
```

## Versioning semantics

Mirrors `user_preference`. On every `register_dataset(...)` or
`register_column(...)`, inside a single `BEGIN IMMEDIATE` transaction:

1. UPDATE any active row for the identity tuple — stamp
   `valid_to = updated_at = now()`.
2. INSERT a brand-new row with a fresh `uuid4().hex` id,
   `valid_from = now()`, `valid_to = NULL`.

The new row carries the **merged** state: caller-supplied non-None
fields win; `None` means "keep prior". This is what makes incremental
fact accumulation cheap — you can register grain first, then later add
`primary_key` without re-stating grain. `forget_dataset` and
`forget_column` close the active row without inserting a replacement.

Example timeline for `(alice, fp=abc..., table=customers)`:

```
t0  register_dataset(grain="one row per customer")
       -> row A: valid_from=t0, valid_to=NULL,
                 grain="one row per customer"

t1  register_dataset(primary_key=["customerID"])
       -> row A: valid_to=t1   (closed)
          row B: valid_from=t1, valid_to=NULL,
                 grain="one row per customer" (carried),
                 primary_key=["customerID"]   (new)

t2  forget_dataset(...)
       -> row B: valid_to=t2   (closed, no new row)
```

`list_dataset_history(...)` returns `[A, B]` ordered by `valid_from ASC`.

## Configuration

Single env var, shared with `user_preference`: `MADS_DB_PATH` — path to
the SQLite file. Default `./.mads/memory.sqlite` (parent dir is
`mkdir -p`'d on store init). Pass `db_path=":memory:"` for an in-memory
store (used by every test). The two categories live in the same SQLite
file by default — two tables, one connection per `*Store` instance.

## Python API

```python
class DatasetSchemaStore:
    def __init__(self, db_path: str | None = None) -> None:
        """Open the SQLite store; ensure tables and partial unique indexes exist."""

    # ---- Dataset side
    def register_dataset(self, *, user_id: str, df: pd.DataFrame,
                         dataset_name: str, table_name: str,
                         project_id: str | None = None,
                         grain: str | None = None,
                         primary_key: list[str] | None = None,
                         join_keys: list[dict] | None = None,
                         time_columns: list[str] | None = None,
                         target_columns: list[str] | None = None,
                         known_issues: list[str] | None = None
                         ) -> tuple[str, str]:
        """Register/update a dataset row. Returns (new_id, fingerprint).
        Merges with the prior active row for (user_id, fingerprint, table_name)."""

    def get_dataset(self, *, user_id, fingerprint, table_name) -> dict | None:
        """Return the active dataset row, or None."""
    def list_datasets(self, user_id: str) -> list[dict]:
        """All active datasets for user_id, sorted by (dataset_name, table_name)."""
    def forget_dataset(self, *, user_id, fingerprint, table_name) -> bool:
        """Close the active dataset row. No replacement inserted."""
    def list_dataset_history(self, *, user_id, fingerprint, table_name) -> list[dict]:
        """All rows (active+closed) for the dataset identity, oldest first."""

    # ---- Column side
    def register_column(self, *, fingerprint: str, table_name: str,
                        column_name: str,
                        semantic_type: str | None = None,
                        business_meaning: str | None = None,
                        pii_level: str | None = None,
                        example_values: list | None = None,
                        null_behavior: str | None = None) -> str:
        """Register/update a column row. Returns new row id.
        Raises ValueError if pii_level is provided and not in
        {'none','low','medium','high'}."""

    def get_column(self, *, fingerprint, table_name, column_name) -> dict | None:
        """Return the active column row, or None."""
    def list_columns(self, *, fingerprint, table_name) -> list[dict]:
        """All active column rows for the dataset table, ordered by column_name."""
    def forget_column(self, *, fingerprint, table_name, column_name) -> bool:
        """Close the active column row. No replacement inserted."""
    def list_column_history(self, *, fingerprint, table_name, column_name) -> list[dict]:
        """All rows (active+closed) for the column identity, oldest first."""

    def close(self) -> None:
        """Close the underlying SQLite connection."""
```

JSON-encoded columns (`primary_key`, `join_keys`, `time_columns`,
`target_columns`, `known_issues`, `example_values`) are returned
pre-parsed as Python lists/dicts on every read.

Minimal usage:

```python
import pandas as pd
from mads.memory import DatasetSchemaStore

store = DatasetSchemaStore(":memory:")
df = pd.read_csv("baseline/data/churn_data.csv")

new_id, fp = store.register_dataset(
    user_id="alice", df=df,
    dataset_name="telecom_churn_2026Q1",
    table_name="customers",
    grain="one row per customer",
)
store.get_dataset(user_id="alice", fingerprint=fp, table_name="customers")["grain"]
# 'one row per customer'

# Merge: only primary_key is new; grain carries over.
store.register_dataset(
    user_id="alice", df=df,
    dataset_name="telecom_churn_2026Q1",
    table_name="customers",
    primary_key=["customerID"],
)
active = store.get_dataset(user_id="alice", fingerprint=fp, table_name="customers")
active["grain"]        # 'one row per customer'  (carried)
active["primary_key"]  # ['customerID']          (new)

history = store.list_dataset_history(
    user_id="alice", fingerprint=fp, table_name="customers"
)
# len(history) == 2; history[0]['valid_to'] is set, history[-1]['valid_to'] is None
store.close()
```

## LangChain tools API

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

Returns exactly six `@tool`-decorated callables. The factory computes
the fingerprint once at build time and binds `(store, df, fingerprint,
dataset_name, table_name, user_id, project_id)` into each tool's
closure. The LLM never sees any of these.

| Tool name | LLM-facing signature | Returns (string) |
|---|---|---|
| `register_dataset` | `(grain=None, primary_key=None, join_keys=None, time_columns=None, target_columns=None, known_issues=None) -> str` | `"Registered dataset '<name>' (table=..., fingerprint=..., id=...). <fields set>."` |
| `recall_dataset`   | `() -> str` | Pretty-printed active dataset row, or `"No dataset memory for '<name>' (table=...)."` |
| `list_datasets`    | `() -> str` | `"You have memory for N dataset(s):\n- ..."` or `"No datasets registered yet."` |
| `register_column`  | `(column_name, semantic_type=None, business_meaning=None, pii_level=None, example_values=None, null_behavior=None) -> str` | `"Registered column '<col>' on table '<t>' (id=...)."` or `"Error: ..."` (unknown column or bad `pii_level`, returned as a string — does not raise) |
| `recall_column`    | `(column_name) -> str` | Pretty-printed active column row, or `"No column memory for '<col>'."` |
| `list_columns`     | `() -> str` | `"You have memory for N column(s) on '<t>':\n- ..."` or `"No column memory for table '<t>'."` |

Wiring:

```python
from mads.memory import DatasetSchemaStore, make_dataset_tools
import pandas as pd
store = DatasetSchemaStore()                # default DB path or ":memory:"
df = pd.read_csv("baseline/data/churn_data.csv")
tools = make_dataset_tools(
    store, df,
    dataset_name="telecom_churn_2026Q1",
    table_name="customers",
    user_id="alice",
    project_id="growth",
)
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-sonnet-4-6").bind_tools(tools)
# Closure-bound: LLM never sees user_id, project_id, dataset_name, table_name, or df.
```

## Running the tests

```bash
cd /Users/wang/LLM_memory && .venv/bin/python -m pytest tests/test_dataset_schema.py -v --tb=short
```

24 tests, all using `db_path=":memory:"` — no filesystem touch. Wall
time ~0.4s on Python 3.13.2 / pytest 9.0.3.

## Running the demo

```bash
cd /Users/wang/LLM_memory && .venv/bin/python demo_dataset_schema.py > demo_dataset_schema.out 2>&1
```

17 labeled steps against `baseline/data/churn_data.csv` (7043 rows, 21
columns). Proves: fingerprint stability under column/row shuffles,
register-then-recall, merge-on-supersede, `list_datasets`, registering
5 columns, recall + list, column supersession + history, the two error
paths (unknown column, bad `pii_level`), per-user dataset scoping for
alice vs bob, and the deliberate column-fact sharing across users for
the same fingerprint. Writes a transient SQLite file at
`/Users/wang/LLM_memory/.demo_dataset_schema.sqlite` (removed at start).

## Known limitations

- List/dict fields (`primary_key`, `join_keys`, `time_columns`,
  `target_columns`, `known_issues`, `example_values`) are stored as
  JSON strings via `json.dumps(..., sort_keys=True, default=str)`. The
  store does not validate shape — callers are responsible for the
  documented `list[str]` / `list[dict]` / `list` payloads.
- No embeddings or semantic retrieval. Column lookups are exact-key
  (`column_name` string match). There is no fuzzy/semantic "find
  columns that mean revenue" query. Future work.
- `column_memory` has no `user_id` by design — fine for single-tenant
  or trusted-shared deployments. Multi-tenant deployments where column
  semantics must be private per tenant will need a per-tenant schema
  namespace, an added `tenant_id`, or a migration.
- Fingerprint is robust to row/column reorderings, but a single-row
  append **can** cross a log10 row-count bucket boundary (e.g. 99 rows
  -> bucket "1", 100 rows -> bucket "2") and produce a different
  fingerprint, severing the memory link. Test 7 documents the
  in-bucket insensitivity; bucket boundaries themselves are real
  transitions.
- One-DataFrame-per-factory binding ("Option A" in
  `IMPLEMENTATION_NOTES_DATASET.md`). Sessions touching multiple
  datasets need one `make_dataset_tools(...)` call per dataset. A
  future "Option B" (`df_registry`) that dispatches across datasets by
  name is noted in `IMPLEMENTATION_NOTES_DATASET.md` but not
  implemented in v1.

## How this fits

Category 2 of 8 in the `CLAUDE.md` memory taxonomy (mapped to
`schema_fact` + column-level dictionary). Same shape as
`user_preference`: a `*Store` class plus a `make_*_tools` factory that
closes over identity. Remaining: `metric_definition`,
`data_quality_issue`, `analysis_conclusion`, `workflow_recipe`,
`pipeline_decision`, `experiment_record`. The concrete next step is the
third category — most likely `metric_definition`, since the eval
scenarios lean on named metrics that need versioned definitions.

## File map

```
mads/memory/
  __init__.py                       # re-exports DatasetSchemaStore,
                                    #   make_dataset_tools, compute_fingerprint
                                    #   (alongside PreferenceStore + tools)
  dataset_schema.py                 # DatasetSchemaStore + compute_fingerprint
  dataset_tools.py                  # make_dataset_tools: 6 @tool callables
  IMPLEMENTATION_NOTES_DATASET.md   # implementer's contract
  DATASET_SCHEMA_README.md          # this file
  README.md                         # category 1 (user_preference) README
  user_preference.py                # category 1 store
  preference_tools.py               # category 1 tools
  IMPLEMENTATION_NOTES.md           # category 1 contract

tests/
  test_dataset_schema.py            # 24 tests (in-memory SQLite)
  results/
    test_dataset_schema.md          # tester's result summary
    test_dataset_schema.out         # raw pytest stdout
    test_dataset_schema.err         # raw pytest stderr (empty)

demo_dataset_schema.py              # end-to-end CLI demo (17 labeled steps)
demo_dataset_schema.out             # captured transcript of the demo run
```
