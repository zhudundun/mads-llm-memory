# Metric Definition Memory

Third of the 8 memory categories defined for `mads` (see `CLAUDE.md`).
Stores the canonical, versioned definition of a named business metric in
one project — its prose description, optional SQL/Python implementations,
optional default filters, plus authorship and confidence metadata. The
store guarantees one active definition per `(project, metric)` and
preserves every prior version on the same identity chain.

## When to use this

Analyst pain point: metric definitions drift. "Active user" meant
sessions-in-last-28d in Q1, then sessions-in-last-7d-excluding-bots in
Q2; nobody recorded the cutover and three dashboards now disagree. This
module is the system of record for "what does this metric mean *right
now* in this project, and what did it mean last quarter, and who changed
it?" — written by the agent (or analyst) at the moment a definition is
agreed, read back on every later question that names the metric.

## Architectural place

Category 3 in `CLAUDE.md`'s 8-category table (`metric_definition` in the
`memories.category` enum). Like cats 1 (`user_preference`) and 2
(`schema_fact` / dataset+column memory), cat 3 ships in its own
dedicated SQLite table — `metric_memory` — rather than in the rollup
`memories` table; that rollup is a future consolidation step. Identity,
versioning, and tenancy follow CLAUDE.md's universal contracts:
append-only history via `valid_from` / `valid_to`, `WHERE` clauses
enforced at the store, no inline DataFrames anywhere.

## Schema

Verbatim from `infra/migrations/0003_metric_memory.sql`:

```sql
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
```

| Column | Meaning |
|---|---|
| `id` | 32-char `uuid4().hex`; primary key, immutable. |
| `user_id` | Author of this version. Recorded for audit; does NOT participate in identity. |
| `project_id` | Tenancy boundary. Identity-bearing — same metric name in two projects = two independent metrics. |
| `metric_name` | Caller's original casing (e.g. `"Active user"`). Display only. |
| `normalized_name` | Deterministic identity form (see normalization rule). Stored so the partial unique index can use it. |
| `business_definition` | Prose definition. Required, non-empty. Definitional field. |
| `sql_definition` | Optional SQL snippet. Definitional field. |
| `python_definition` | Optional Python expression / snippet. Definitional field. |
| `default_filters` | Optional WHERE-clause-style filter string. Definitional field. |
| `valid_from` | UTC ISO-8601; when this version became active. |
| `valid_to` | UTC ISO-8601 when closed, else `NULL` for the currently-active row. |
| `confidence` | `[0.0, 1.0]`; how sure the author is. Metadata field; CHECK + pydantic both enforce range. |
| `source` | Free-text origin (e.g. `"analytics_team"`). Metadata field. |
| `created_at`, `updated_at` | UTC ISO-8601 timestamps. |

The partial unique index `uq_metric_active` is the structural guarantee
that two active rows for the same `(project_id, normalized_name)` cannot
coexist, even if a caller bypasses `upsert` and INSERTs directly
(tested).

## Identity & versioning

**Identity tuple:** `(project_id, normalized_metric_name)`. Same name in
different projects = independent metrics; within a project, two users
share one metric library. `project_id` and `user_id` are both `NOT NULL`;
tenancy is enforced at the store, never via LLM args.

**Normalization rule** (`_normalize`): lowercase, replace any non-
`[0-9a-z]` run with a single `_`, strip leading/trailing `_`. Empty
result raises `ValueError`.

Worked examples:
- `"Active User"` → `"active_user"`
- `"Conversion Rate (CR)"` → `"conversion_rate_cr"`

**Versioning split.** A definition has two kinds of fields:

- *Definitional* (the semantic anchor): `business_definition`,
  `sql_definition`, `python_definition`, `default_filters`. A change to
  any of these means the metric *means something different now*, so the
  prior row is closed and a new row is inserted.
- *Metadata* (authorship/calibration): `confidence`, `source`. A change
  here doesn't shift what the metric means; we update the active row in
  place and don't burn a version.

`upsert(...)` runs inside one `BEGIN IMMEDIATE` transaction, classifies
the call against the active row (if any), and returns
`(MetricDefinition, action)` where `action` is one of:

| Action | Trigger | Effect |
|---|---|---|
| `inserted` | no active row exists | INSERT new row, `valid_from = now`, `valid_to = NULL` |
| `noop` | active row exists; all definitional + metadata fields match | nothing written; returns existing row |
| `updated_in_place` | active row exists; definitional fields all match but `confidence` or `source` differs | UPDATE `confidence`/`source`/`updated_at`; same `id`, same `valid_from`, same `user_id` |
| `versioned` | active row exists; any definitional field differs | UPDATE old row `valid_to = now`; INSERT new row in same transaction; `old.valid_to == new.valid_from` exactly |

Active rows have `valid_to IS NULL`. Closed rows have a non-NULL
`valid_to`. Both timestamp writes inside a `versioned` upsert use the
same `now` value, so the closure point is exact (tested).

## Python API

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
    def get_active(self, *, project_id, metric_name
                   ) -> MetricDefinition | None: ...
    def list_active(self, *, project_id) -> list[MetricDefinition]: ...
    def search_active(self, *, project_id, query
                      ) -> list[MetricDefinition]: ...
    def history(self, *, project_id, metric_name
                ) -> list[MetricDefinition]: ...
    def close(self) -> None: ...
```

- `__init__` opens (or reuses) the SQLite connection and applies the
  migration. Honors `MADS_DB_PATH` (default `./.mads/memory.sqlite`,
  shared with cats 1/2) and `":memory:"`.
- `init_schema` applies `0003_metric_memory.sql`; idempotent. Falls back
  to an inline copy if the migration file isn't shipped.
- `upsert` is the only writer; see action table above.
- `get_active` returns the open version for `(project_id, metric_name)`,
  or `None`. Names that normalize to empty also return `None` (not raise)
  — symmetric with "absent".
- `list_active` returns all open versions in a project, ordered by
  `LOWER(metric_name) ASC`.
- `search_active` is case-insensitive substring match over
  `metric_name` and `business_definition`; empty/non-string query returns
  `[]`.
- `history` returns every version (open + closed) for the metric,
  oldest first (`valid_from ASC, created_at ASC`).
- `close` closes the connection if owned.

`MetricDefinition` is a pydantic v2 model; `confidence` is constrained
`ge=0.0, le=1.0` in the model and `CHECK (confidence BETWEEN 0.0 AND
1.0)` in SQL.

## LangChain tool surface

`make_metric_factory(store, user_id, project_id).tools()` returns 5
`@tool`-decorated callables. None of the LLM-visible signatures contain
`user_id` or `project_id`.

| Tool | LLM-visible signature | When the agent calls it |
|---|---|---|
| `add_metric_definition` | `(metric_name: str, business_definition: str, sql_definition: Optional[str] = None, python_definition: Optional[str] = None, default_filters: Optional[str] = None, confidence: float = 1.0, source: Optional[str] = None) -> str` | The analyst stated (or the agent inferred) a definition of a named metric. Always safe to re-call with identical args (no-op). The return string includes `inserted` / `versioned` / `updated_in_place` / `noop` plus the version number. |
| `get_metric_definition` | `(metric_name: str) -> str` | Before answering any question that uses a named metric. Returns the active definition or an empty-state string. |
| `list_metric_definitions` | `() -> str` | The agent wants an overview of what metrics this project knows about — compact table of `name | business_definition | confidence | source`. |
| `search_metric_definitions` | `(query: str) -> str` | The analyst asked about a concept ("conversion-ish things") without naming the exact metric. Substring match over name and business definition. Empty query returns an error string, not every row. |
| `get_metric_history` | `(metric_name: str) -> str` | The analyst asked "what did we mean by X last quarter" or "who changed Y?" — returns oldest-first version list with `valid_from -> valid_to`, author, and one-line definition. |

## Tenancy contract

CLAUDE.md mandates "All memory queries are user-scoped at the store
layer. `WHERE user_id=…` lives in `memory/store.py`, not in prompts." For
cat 3 the scoping column is `project_id` (because the metric library is
shared across users in a project), with `user_id` recorded on every
write for audit. The factory closes over both, so the LLM never sees
them:

```python
from mads.memory import MetricDefinitionStore, make_metric_factory

store = MetricDefinitionStore(":memory:")
factory = make_metric_factory(store, user_id="wang", project_id="growth")
tools = factory.tools()   # 5 BaseTool instances; closures hold (store, "wang", "growth")
```

`MetricToolFactory.__init__` rejects empty `user_id` or `project_id`.

## Usage example

Transcript-style; derived from `eval/demo_metric_memory.py`.

```text
>>> factory = make_metric_factory(store, "alice", "growth")
>>> by_name = {t.name: t for t in factory.tools()}

# Step 2: define Active user with confidence 0.7
>>> by_name["add_metric_definition"].invoke({
...     "metric_name": "Active user",
...     "business_definition": "A distinct user with at least one session in the last 28 days.",
...     "sql_definition": "SELECT COUNT(DISTINCT user_id) FROM sessions WHERE ts >= now() - INTERVAL '28 days'",
...     "confidence": 0.7,
...     "source": "analytics_team",
... })
"Metric 'Active user' inserted (version 1, id=..., confidence=0.7, source=analytics_team)."

# Step 8: bump confidence -> updated_in_place (no new version)
>>> by_name["add_metric_definition"].invoke({
...     "metric_name": "Active user",
...     "business_definition": "A distinct user with at least one session in the last 28 days.",
...     "sql_definition": "SELECT COUNT(DISTINCT user_id) FROM sessions WHERE ts >= now() - INTERVAL '28 days'",
...     "confidence": 0.95,
...     "source": "analytics_team",
... })
"Metric 'Active user' updated_in_place (version 1, id=..., confidence=0.95, source=analytics_team)."

# Step 9: revise Conversion business_definition -> versioned
>>> by_name["add_metric_definition"].invoke({
...     "metric_name": "Conversion",
...     "business_definition": "Orders divided by sessions in the same window, excluding bot sessions.",
...     ...
... })
"Metric 'Conversion' versioned (version 2, id=..., ...)."

# Step 10: history shows both versions, oldest first, with authors
>>> by_name["get_metric_history"].invoke({"metric_name": "Conversion"})
"History for 'Conversion' in project 'growth' (2 version(s)):
  v1: <ts> -> <ts> | by alice | Orders divided by sessions in the same window.
  v2: <ts> -> active | by alice | Orders divided by sessions in the same window, excluding bot sessions."
```

## Testing

```bash
pytest tests/test_metric_definition.py -q
# 36 passed
```

Coverage areas: normalization (3 tests), identity & project scoping (2),
upsert classifier across all 4 actions (8), versioning invariants
including single-active-row, `valid_to == valid_from` of next version,
and partial-unique-index rejection of bypassing writes (3), retrieval —
`get_active` / `list_active` / `search_active` / `history` (4), tenancy
isolation between projects and sharing within a project (2), input
validation — empty names, unnormalizable names, empty business
definition, out-of-range confidence (4), and the tool layer — factory
shape, hidden tenancy, action labels surfaced in returned strings,
empty-state strings, history rendering with authors, cross-project
isolation, empty-query guard, factory rejecting empty tenancy (10).

## Demo

```bash
python eval/demo_metric_memory.py
```

12 labeled steps against an in-memory SQLite store, walking the
analyst-grade story (define 4 metrics; list; search; in-place confidence
bump; versioned business-definition revision; render history; confirm a
fresh project sees nothing).

## Limitations / deferred work

- **No FTS or embeddings.** `search_active` is case-insensitive
  substring over `metric_name` and `business_definition` only. No
  fuzzy match, no semantic recall.
- **No PII scrub.** CLAUDE.md's `memory/scrub.py` contract is not
  wired in for any category yet, including this one. Definitional
  strings are author-written, not raw row contents, so the gap is
  lower-risk here than for `analysis_conclusion`, but it remains a
  contract gap.
- **No migration system.** `init_schema()` (idempotent
  `CREATE TABLE IF NOT EXISTS`) is the only path. There is no migration
  tracker, no down-migrations, and no `make migrate` integration for
  cat-3-specific schema changes — schema evolution past v1 will need
  one.
- **No rollup into `memories`.** Cat 3 lives in its own table
  (`metric_memory`); the universal `memories` rollup with
  `category='metric_definition'` is a future consolidation step.
- **`source` is free text.** No controlled vocabulary, no foreign key
  to a `teams` table.

## Related

- Cat 1: `mads/memory/README.md` (`user_preference` — per-user
  key/value store, same versioning shape).
- Cat 2: `mads/memory/DATASET_SCHEMA_README.md` (`schema_fact` plus
  per-column dictionary, content-addressed by DataFrame fingerprint).
