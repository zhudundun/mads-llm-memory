# mads memory subsystem

Reference for contributors working on `mads/memory/`. Covers the five
memory categories, the SQLite + sqlite-vec storage layout, the tenancy
model, the embedding pipeline, the store / retriever / extractor triad
(only stores are wired in v1), and the `MemoryManager` facade. Pairs
with `docs/architecture.md`.

## The 5 categories

| # | Category | Identity tuple | Versioning | Wired in v1 | In Manager |
|---|---|---|---|---|---|
| 1 | `user_preference` | `(user_id, project_id, key)` | versioned via `valid_to` | shipped, not in Manager | no |
| 2 | `dataset_schema` (`dataset_memory` + `column_memory`) | `(user_id, dataset_fingerprint, table_name)` / `(dataset_fingerprint, table_name, column_name)` | versioned via `valid_to` | shipped, in Manager | yes (late-bound) |
| 3 | `metric_definition` | `(project_id, normalized_name)` | versioned via `valid_to` | shipped, in Manager | yes |
| 4 | `data_quality_issue` | `(project_id, dataset_fingerprint, table_name, column_name, issue_type)` | versioned via `valid_to` | shipped, not in Manager | no |
| 5 | `analysis_episode` | `(id)` — UUID | append-only (single mutable field: `user_feedback`) | shipped, in Manager, embedded | yes |

Cats 1 and 4 are present, tested, and importable from `mads.memory` but
not surfaced by `MemoryManager`, not injected into wrapped agents, and
not addressable from `mads memory ...`. v1.1 wires them in by extending
only Manager + wrapper + CLI registries.

Categories 6 (`workflow_recipe`), 7 (`pipeline_decision`), and 9
(`experiment_record`) are not yet implemented. Item 8 from the original
user request — "temporal changes in definitions" — is the versioning
mechanism on cats 1-4, not a category of its own.

## One shared SQLite file

The default DB path is `./.mads/memory.sqlite`, overridable via the
`MADS_DB_PATH` environment variable. Each category gets its own table
(or pair of tables, for cat 2):

```
.mads/memory.sqlite
  user_preferences           cat 1 versioned rows
  dataset_memory             cat 2 dataset-level versioned rows
  column_memory              cat 2 column-level versioned rows
  metric_memory              cat 3 versioned rows
  data_quality_memory        cat 4 versioned rows
  analysis_episode_memory    cat 5 append-only rows
  analysis_episode_vec       cat 5 vec0 virtual table (sqlite-vec)
```

There is no unified `memories` table. Schema lives in
`infra/migrations/000{N}_<name>_memory.sql` where present, mirrored as
`_INLINE_MIGRATION_SQL` inside the corresponding store class so the
package keeps working when consumed from a wheel that doesn't ship
`infra/`.

The `MemoryManager` opens **one** `sqlite3.Connection` with
`check_same_thread=False` (the eval harness runs sessions in daemon
threads), loads the `sqlite-vec` extension once, then constructs each
store with `conn=self._conn`. All three in-scope stores share that
connection; `manager.close()` is idempotent and closes it exactly once.

## Tenancy model

Tenancy is `(user_id, project_id)`, both required strings. The Manager
validates non-empty strings at construction and closes the pair into
every factory:

```python
manager = MemoryManager(user_id="wang", project_id="churn", ...)
metric_factory = manager.metric         # closes over (wang, churn)
```

LLM-visible tool signatures never include `user_id` or `project_id`.
`tests/test_tenancy_e2e.py` asserts this for all three wired
factories' `args_schema` exclusions.

Identity tuples differ per category, deliberately:

- **metric_definition**: `(project_id, normalized_name)` — user_id is
  recorded on every row (NOT NULL — we want to know who created the
  version) but does **not** participate in identity. Two analysts in
  the same project share one metric library.
- **dataset_schema → dataset_memory**: `(user_id, dataset_fingerprint, table_name)`
  — dataset-level facts are scoped to the user who registered them
  (their notes about a file).
- **dataset_schema → column_memory**: `(dataset_fingerprint, table_name, column_name)`
  — **no** `user_id`. Column-level facts ("`customer_id` is the PK")
  are a property of the data, not the user, so they're shared across
  users sitting on the same fingerprint.
- **analysis_episode**: `(id)` — a UUID per event. Episodes are
  append-only; there's no merge key.

## Versioning

Cats 1-4 use the close-prior-insert-new pattern: on every upsert that
changes a definitional field, the store finds the active row for the
identity tuple (the row where `valid_to IS NULL`), sets its
`valid_to = now()`, and inserts a brand-new row with `valid_from = now()`
and `valid_to = NULL`. Unchanged fields are carried over so callers can
pass partial payloads. History is reconstructed by ordering rows on
`(identity_tuple, valid_from ASC)`.

A unique partial index on each table — `WHERE valid_to IS NULL` — keeps
"at most one active row per identity" honest at the schema level (see
e.g. `uq_metric_active` on `metric_memory`).

Cat 5 is append-only. Each `record_analysis_episode` is immutable
except for one field: `update_user_feedback` mutates `user_feedback` in
place AND re-embeds (the embedded text concatenates four free-text
fields including `user_feedback`, so feedback updates must propagate
to the vec0 row in the same transaction).

The v1.1 `memory/versioning/version_manager.py` scaffold centralizes the
close-and-insert pattern; today it is duplicated across four stores.

## Embedding pipeline

Only cat 5 embeds. The pipeline:

1. Concatenate `question + "\n\n" + conclusion + "\n\n" + assumptions + "\n\n" + user_feedback`.
2. Pass through the configured `Embedder` (`mads/memory/embedding.py`):
   - **Production**: `SentenceTransformerEmbedder` — `all-MiniLM-L6-v2`,
     384-dim, CPU, lazy-loaded so simply importing the class doesn't
     pay the ~80 MB HuggingFace download. Output is L2-normalized so
     vec0's default L2 distance is consistent with cosine.
   - **Tests**: `HashEmbedder` — deterministic, non-semantic, blake2b
     token hashing into 384 buckets, then L2-normalized. Used by every
     episode-touching test so the suite doesn't pay the model download.
     `MemoryManager` defaults to `HashEmbedder` for the same reason; the
     CLI / production path passes a `SentenceTransformerEmbedder`
     explicitly.
3. Insert into `analysis_episode_vec` (a `vec0` virtual table) keyed by
   `episode_id`, in the same `BEGIN IMMEDIATE` transaction as the row
   insert into `analysis_episode_memory`. Same goes for delete and for
   `update_user_feedback`. Divergence between the two tables is not
   externally observable.

Structured fields (`analysis_type`, `datasets_used`, `metrics_used`,
`methods_used`, `code_refs`, `chart_refs`) are **not** embedded — they
exist for filter/display, not for similarity search.

## Store / retriever / extractor triad

v1 keeps the design simple: retrieval is exposed through methods on
each store (`get_metric_definition`, `list_recent_analyses`,
`find_similar_analyses`, `recall_dataset`, ...), and the LangChain
`@tool` factories call those methods directly. There is no separate
retriever layer.

v1.1 scaffolds a different split:

```
mads/memory/retrievers/
  metric_retriever.py       MetricRetriever.search(query, k=5)
  dataset_retriever.py      DatasetRetriever.recall(fingerprint, table)
  episode_retriever.py      EpisodeRetriever.find_similar(query, k=5)
  hybrid_retriever.py       HybridRetriever.retrieve(query, k=10)
```

Each per-category retriever is a thin shim over the corresponding
store's read methods. The `HybridRetriever` fans out to every
per-category retriever, normalizes per-category scores, and applies
`mads/memory/core/scoring.Scorer` to produce one fused ranked list.
That fused list is what `mads/router/context_builder.ContextBuilder`
would splice into agent prompts. All four retrievers are currently
stubs raising `NotImplementedError`.

The write-side mirror is the **extractor** subpackage:

```
mads/memory/extractors/
  metric_extractor.py       MetricExtractor.extract(transcript)
  dataset_extractor.py      DatasetExtractor.extract(transcript, fingerprint)
  episode_extractor.py      EpisodeExtractor.extract(transcript, agent_name)
```

These are deterministic miners — not LLM calls — that would scan the
agent's question + answer + tool-call log for facts the LLM forgot to
record via the explicit tool-call path. Per the brief, extracted
candidates are NEVER silently persisted; they're surfaced to the user
for confirmation. All three are stubs.

## The Memory Manager facade

`mads/manager.py:38` defines `MemoryManager`. Public API:

| Method / property | Where | Purpose |
|---|---|---|
| `__init__(*, user_id, project_id, db_path=None, embedder=None)` | `manager.py:46` | Validate tenancy, resolve db path (arg > env > default), open shared conn, load sqlite-vec. |
| `user_id` / `project_id` / `db_path` / `conn` | `manager.py:110-123` | Read-only accessors. |
| `metric` | `manager.py:152` | Cached `MetricToolFactory` closed over (user, project). |
| `episode` | `manager.py:161` | Cached `AnalysisEpisodeToolFactory` closed over (user, project). |
| `bind_dataframe(df, *, dataset_name, table_name)` | `manager.py:171` | Late-bind a df for dataset tools. Invalidates the cached dataset tool list. |
| `dataset_tools()` | `manager.py:196` | Return 6 dataset `@tool`s bound to the currently-bound df. Raises `RuntimeError` if no df is bound. |
| `all_tools(*, include_dataset=True)` | `manager.py:225` | Union: 5 metric + 6 episode + (optionally) 6 dataset. |
| `close()` | `manager.py:239` | Idempotent close of the shared conn. |

**The load-bearing pattern is late-binding the dataframe.** The dataset
tool factory pre-computes a fingerprint at construction time
(`mads/memory/dataset_tools.py`) — it cannot exist before the analyst
has loaded data. `MemoryManager` resolves this with `bind_dataframe`:
the wrapper binds the df after the loader returns it, then asks for
`dataset_tools()`, which constructs (and caches, keyed by Python
`id(df)`) the 6 dataset tools fingerprinted to **that** df.

This is also why the loader's first turn gets only 11 tools (metric +
episode): no df has been bound yet, so `include_dataset=False` is the
safe default. The EDA and Viz wrappers always bind the df at wrap time
and get the full 17 tools.

## What's deferred and why

Each scaffold lists its v1.1 trigger in its module docstring. Summary:

- **`memory/extractors/*`** — observed gap-closure rate <30% in the
  with-memory eval (i.e. evidence the LLM is forgetting to write).
- **`memory/versioning/version_manager.py`** — sixth category arrives;
  duplicating the close-and-insert dance a fifth time is the breaking
  point.
- **`memory/versioning/conflict_detector.py`** — first parallel-write
  conflict observed (SQLite's single-writer model gives us
  serializability today).
- **`memory/core/schemas.py`** — adding cat 6 or 9 (both carry typed
  references into cats 2 and 3); doing it sooner is premature
  centralization.
- **`memory/core/namespace.py`** — first cross-project read pattern
  ("show me metrics across all my projects"); the two-tuple is fine
  until then.
- **`memory/core/policies.py`** — first ops request for a retention
  knob, or when PII scrubbing actually ships.
- **`memory/core/scoring.py`** — single-signal retrieval visibly
  mis-ranks in the eval.
- **`memory/stores/sqlite_store.py`** — sixth category (same trigger as
  the version manager).
- **`memory/stores/vector_store.py`** — second category needs vector
  search.
- **`memory/stores/artifact_store.py`** — cats 6 and 9 land, both
  producing real artifacts.
- **`memory/retrievers/*`** — context builder lands and needs a single
  ranked recall surface.

## Tests

| Category | File | Count |
|---|---|---|
| user_preference | `tests/test_user_preference.py` | 14 |
| dataset_schema | `tests/test_dataset_schema.py` | 24 |
| metric_definition | `tests/test_metric_definition.py` | 36 |
| data_quality | `tests/test_data_quality.py` | 52 |
| analysis_episode | `tests/test_analysis_episode.py` | 52 |

Per-category minimum coverage is identity / insert / upsert paths,
tenancy isolation, and an `args_schema` assertion that tenancy fields
are absent. Cat 5 additionally tests embedder injection, vec0 sync, and
the `find_similar` ranking.

The cross-layer guardrail is `tests/test_tenancy_e2e.py` (7 cases) —
two managers on the same db with different `project_id` must remain
isolated for metrics and episodes, same-project cross-user must see
shared metrics, column memory is shared by design, and every factory's
`args_schema` must exclude `user_id` / `project_id`.

Full suite: `pytest -q` → **283 passed**.

## Pointers

Per-category user-facing docs:

- `mads/memory/METRIC_DEFINITION_README.md`
- `mads/memory/DATASET_SCHEMA_README.md`
- `mads/memory/ANALYSIS_EPISODE_README.md`
- `mads/memory/README.md` — package-level overview

Per-category implementation handoffs:

- `mads/memory/IMPLEMENTATION_NOTES.md` — top-level
- `mads/memory/IMPLEMENTATION_NOTES_METRIC.md`
- `mads/memory/IMPLEMENTATION_NOTES_DATASET.md`
- `mads/memory/IMPLEMENTATION_NOTES_DATA_QUALITY.md`
- `mads/memory/IMPLEMENTATION_NOTES_ANALYSIS_EPISODE.md`

Cross-cutting:

- `docs/architecture.md` — top-level v1 / v1.1 layout.
- `V1_DESIGN_BRIEF.md` — architect's v1 spec.
- `V1_README.md` — user-facing v1 docs.
- `CLAUDE.md` — architectural contracts.
