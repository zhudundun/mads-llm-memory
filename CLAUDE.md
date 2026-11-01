# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`mads` ("memory-augmented data scientist") is a long-term-memory agent built **on top of**
[`business-science/ai-data-science-team`](https://github.com/business-science/ai-data-science-team) (ADS).
ADS supplies the data-science sub-agents (loader, wrangling, cleaning, EDA, viz, SQL, feature-eng, H2O, MLflow).
`mads` adds the memory layer, orchestration, and (eventually) a LangGraph supervisor.

The repo is at v0.0.1: five memory categories shipped, four more pending; no Makefile, no LangGraph
supervisor yet. Don't fabricate completeness — when you find a stub, either implement it properly or
leave it and surface that to the user.

## Architectural contracts (don't break these)

- **Scoping is a declared axis model, not ad-hoc per category.** Three axes: `USER`,
  `PROJECT`, `DATASET` (see `mads/memory/scope.py`). Each category declares each axis as
  `IDENTITY_REQUIRED` (always filtered), `IDENTITY_OPTIONAL` (filtered with `IS NULL OR =`),
  `AUDIT` (recorded, not filtered), or `REFERENCE` (plain string pointer to another
  category). Stores filter on identity axes; AUDIT axes are recorded only for
  accountability. LangChain tool factories close over the active scope (`user_id`,
  `project_id`, `dataset_fingerprint`) and never expose tenancy or fingerprint as a tool
  argument. The one LLM-visible scope control is `remember_preference(...,
  project_scoped: bool)` — a behaviour selector, not tenancy exposure. Don't trust the
  LLM to filter for the current tenant. Per-category axes:
  - cat 1 `user_preference`: USER required, PROJECT optional (NULL=global).
  - cat 2 `dataset_schema`: DATASET required (by fingerprint), USER audit.
  - cat 3 `metric_definition`: PROJECT required, USER audit, DATASET reference.
  - cat 4 `data_quality_issue`: DATASET required (by fingerprint), USER + PROJECT audit.
  - cat 5 `analysis_episode`: PROJECT required, USER audit, DATASET reference.
- **Definitions are versioned via `valid_to`; episodes are append-only.** Cats 1–4 (`user_preference`,
  `dataset_schema`, `metric_definition`, `data_quality_issue`) close prior rows by setting `valid_to` and
  inserting new rows; identity tuples vary per category. Cat 5 (`analysis_episode`) is append-only — each
  `record_analysis_episode` is immutable except for `user_feedback`. Pick the right pattern when adding a
  new category; don't reach for `valid_from`/`valid_to` reflexively for event-like data.
- **DataFrames travel by reference, not value.** Memory rows hold URIs or content-addressed fingerprints,
  never inline bytes. Pickle-of-df anywhere is forbidden.
- **ADS is wrapped, never imported elsewhere.** Production agent code in `mads/` may not `import
  ai_data_science_team`; only `mads/adapters/ads.py` and `eval/` are exempt. Every other module talks
  to ADS through the adapter's `AgentResult` dataclass and the three handle classes
  (`LoaderHandle` / `EDAHandle` / `VizHandle`).
- **PII is scrubbed before embedding** *(aspirational; scrubbing module not yet built)*. Only cat 5
  currently embeds, and it embeds free-text analyst notes only (`question + conclusion + assumptions +
  user_feedback`). Flag any new code path that would embed user data.
- **Procedural memories will store the exact code that ran** *(deferred to cat 6)*. `workflow_recipe`
  will hash inputs; `run_recipe` will re-execute, never paraphrase.
- **Query-relevance gate sits at `HybridRetriever`, not inside individual retrievers.** Non-data queries
  (poems, jokes, off-topic chat) short-circuit `HybridRetriever.retrieve` to `[]` via
  `mads/memory/retrievers/query_classifier.py::is_data_question` (deterministic keyword check, no LLM).
  This closes the `neg_01_unrelated_query` over-retrieval failure where preferences/episodes were cited
  inside an unrelated poem. The gate is ONE location; sub-retrievers stay query-agnostic.
- **Concurrency: stores serialise their own writes (and reads) with a `threading.RLock`.** Every Store
  class wraps each multi-statement transaction (and read methods, for read-during-write safety) in
  `with self._lock:`. `MemoryManager` constructs ONE re-entrant lock and injects it via `lock=` into
  every store that shares its connection, so cross-category writes can't race on a single conn. WAL
  mode is enabled on file-path DBs (`PRAGMA journal_mode=WAL`); `:memory:` DBs skip WAL (not
  applicable) and rely solely on the lock. `check_same_thread=False` stays — the lock is what makes
  cross-thread connection sharing safe. **New store categories must follow the same pattern**: accept
  an optional `lock=` kwarg in `__init__`, default to `threading.RLock()` when None, wrap every
  BEGIN/COMMIT block, and register a regression test in `tests/test_concurrency.py` that spawns 8+
  threads through `ThreadPoolExecutor` to exercise the new write path. The bug this guards against:
  langgraph's `ToolNode` dispatches tool calls through a `ThreadPoolExecutor`, and unprotected
  `BEGIN IMMEDIATE` blocks on the shared MemoryManager connection raise
  `sqlite3.OperationalError: cannot start a transaction within a transaction`.

## The memory categories

| User req | Category | Status | Source |
|---|---|---|---|
| 1. User preferences | `user_preference` | shipped | `mads/memory/user_preference.py` |
| 2. Dataset/schema knowledge | `dataset_schema` (`dataset_memory` + `column_memory` tables) | shipped | `mads/memory/dataset_schema.py` |
| 3. Metric definitions | `metric_definition` | shipped | `mads/memory/metric_definition.py` |
| 4. Data-quality issues | `data_quality_issue` | shipped | `mads/memory/data_quality.py` |
| 5. Prior analysis conclusions | `analysis_episode` (append-only, embedded) | shipped | `mads/memory/analysis_episode.py` |
| 6. Reusable workflows | `workflow_recipe` | pending | — |
| 7. Past pipeline decisions | `pipeline_decision` | pending | — |
| 8. Temporal changes in definitions | *(versioning mechanism, not a category)* | n/a | `valid_from`/`valid_to` on cats 1–4 |
| 9. Experiment/model history | `experiment_record` | pending | — |

Each shipped category ships its own dedicated SQLite table — there is no unified `memories` table.
Schema lives in `infra/migrations/000{N}_<name>_memory.sql` (where it exists) and is mirrored as
`_INLINE_MIGRATION_SQL` inside the store class for fallback. Pydantic models live next to each store.
Default DB file is `./.mads/memory.sqlite` (override via `MADS_DB_PATH`).

**v1.2 axis-model surface.** Cat 1 (`user_preference`) gained an `IDENTITY_OPTIONAL` `project_id`
column: `NULL` = global preference (applies across every project), non-NULL = scoped to that
project only. `remember_preference(key, value, project_scoped: bool = False)` is the LLM-visible
surface; the closure-bound `project_id` from the factory is attached when `project_scoped=True`.
Both rows can coexist (different identity tuples); the project-scoped row takes precedence in
`get` / `recall` when both apply. Cat 4 (`data_quality_issue`) is now identity by
`dataset_fingerprint` (semantic_fingerprint) — `project_id` demoted to AUDIT, the same dataset
seen in two projects shares one issue library. The 0006 migration drops existing cat-4 rows
(safe pre-prod). `MemoryManager.bind_dataframe(df)` now stashes `semantic_fingerprint(df)` for
the data-quality factory; `MemoryManager.data_quality_tools()` is the new emit surface (NOT in
`all_tools()` — the 15/21 counts are unchanged). `HybridRetriever` and `PreferenceRetriever`
thread `project_id` through so project-scoped prefs surface in addition to global ones; setting
`project_id=None` returns ONLY global prefs (this is what closes `cross_01` at the retriever
level — provided the LLM picks `project_scoped=True` when teaching project-specific rules). See
`mads/memory/scope.py` for the spec module and `infra/migrations/0006_axis_model.sql`.

**v1.1 dataset surface.** Cat 2 now carries dual fingerprints: `exact_fingerprint` (SHA256 of raw
file bytes, snapshot identity) and `semantic_fingerprint` (SHA256 of sorted columns + dtypes,
cross-session logical identity). `compute_fingerprint` is an alias for `semantic_fingerprint(df)`
post-v1.1 — the legacy row-bucket + sample formula was removed. The new `DatasetContext` (pydantic
model, 12 fields including PK/join/time/numeric/categorical candidates plus a 6-line template
`profile_summary`) is produced by `mads/dataset/profiler.py::profile(df, *, dataset_name,
file_bytes=None)` and persists in the new `dataset_context_memory` table (PK = `semantic_fingerprint`,
upserted in place, NOT versioned). Cross-session lookup via
`DatasetSchemaStore.lookup_dataset(*, user_id, df, table_name, file_bytes=None)` returns
`{"matched_by": "exact"|"semantic", "row": ...}` (exact-first, semantic-fallback) or `None`. No new
LLM tool surface — the profiler is a Python API consumed by the wrapper / context builder. See
`docs/dataset_context.md`.

**v1.1 wrapper tool surface.** As of action C, `MemoryManager.all_tools()` emits four memory
categories' tool factories — metric (5), preference (4), episode (6), and optionally dataset (6).
Counts: `include_dataset=False` → 15 tools (loader's first turn before df-binding);
`include_dataset=True` → 21 tools (EDA and viz after `bind_dataframe`). The preference tools
(`remember_preference`, `recall_preference`, `list_preferences`, `forget_preference`) bind
`user_id` via closure (NOT `project_id` — preferences are user-scoped). `PreferenceStore` accepts
optional `conn=`/`lock=` kwargs (matching the post-concurrency-fix pattern of all other stores)
so it shares the manager's connection and RLock.

**v1.1 retrieval surface (complete).** `HybridRetriever` now fans out to **4** sub-retrievers
(metric, dataset, episode, preference). The 4th sub-retriever — `PreferenceRetriever` in
`mads/memory/retrievers/preference_retriever.py` — returns ALL active preferences for the
current user with uniform `score=1.0` and `reason=f"active user preference: '{key}'"`; there is
no tokenized matching and no relevance threshold because preferences are user-state, not
query-context. Preferences are returned **in addition to** the `k` cap on the relevance-ranked
metric/dataset/episode rows (capping them by k would arbitrarily drop user-state). The hybrid
sort key is `(-score, category_order)` with `category_order = {preference: 0, metric: 1,
dataset: 2, episode: 3}` so score-1.0 preferences surface first in the injected memory block.
`MemoryManager.format_context` renders preference rows as `[memory:preference:KEY] (score=1.00)`
blocks with a single `value: <preference_value>` line; `mads/composer.py::_summary_for` has a
matching `"preference"` branch. Wiring this closed the eda_03 gap end-to-end (LLM cites
`[memory:preference:churn_analysis_slicing_rule]` and emits all 4 tenure buckets); eda_04
also closes end-to-end after a one-line widen to `_check_eda_04`'s header regex
(`(?:#+\s*[^\w\s]*\s*)?` tolerates decorative emoji/glyphs between `##` and the heading
keyword) plus a one-scenario re-run — see `eval/results/EVAL_RESULTS.md` §0a & §4.
Rubric-reported gap-closed is now 5/5; substantive and rubric counts agree.

## Common commands

There is no Makefile.

```bash
# One-time setup
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                         # ~3 min cold (torch download)
pip install -e ".[eval]"                        # add ai-data-science-team for eval/

# Test
pytest -q                                       # full suite
pytest tests/test_metric_definition.py -q       # single category

# CLI (after pip install -e ".[dev]")
mads --help
mads ask "Load /path/to/data.csv"                            --user alice
mads ask "Describe the dataset"  --data /path/to/data.csv    --user alice
mads memory list metric                                       --user alice
mads memory list episode --limit 20                           --user alice
mads memory show metric revenue                               --user alice
mads memory history metric revenue                            --user alice
mads memory similar episode "churn vs tenure" --k 3           --user alice

# Per-category demos
python eval/demo_metric_memory.py
python eval/demo_data_quality_memory.py
python eval/demo_analysis_episode_memory.py     # first run downloads ~80MB embedding model

# ADS baseline (cost: ~$0.83 / 30 Anthropic calls)
cd eval && python run_loader.py                 # also run_eda.py, run_viz.py

# Memory-augmented eval (1 scenario smoke ≈ $0.05; full 15 ≈ $1–2)
.venv/bin/python eval/run_with_memory.py --agent loader --scenario loader_01_snake_case
.venv/bin/python eval/run_with_memory.py --agent all     # all 15 scenarios

# Compare baseline vs with-memory
.venv/bin/python eval/compare.py --agent all
cat eval/results/COMPARE.md

# Re-apply local ADS regex patch after any pip install --force-reinstall
python eval/install/apply_viz_patch.py
```

Dependencies are declared in `pyproject.toml` (hatchling backend); install with `pip install -e ".[dev]"`.
Runtime deps: `pydantic`, `langchain-core`, `sqlite-vec`, `sentence-transformers` (pulls torch ~2GB),
`pandas`, `numpy`. Dev extra adds `pytest`. The eval baseline additionally needs `langchain-anthropic`
and `ai-data-science-team` (patched locally; see `eval/install/apply_viz_patch.py`) — these are NOT in
`pyproject.toml` since they're only required by `eval/`, not by importing `mads.memory` or running tests.

## When extending

**Adding a memory category** — mirror cats 3, 4, or 5:

1. Decide append-only (cat 5) vs. versioned (cats 1–4) BEFORE writing code. Event-like categories
   (decisions, experiments, runs) are append-only; definitional ones are versioned.
   Declare a `CategorySpec` in `mads/memory/scope.py` (`Axis` → `AxisRole`) BEFORE schema work:
   pick which of USER/PROJECT/DATASET axes are IDENTITY (filtered), AUDIT (recorded), or
   REFERENCE (plain string pointer). E.g. cat 6 `workflow_recipe` is likely PROJECT identity +
   USER audit; cat 7 `pipeline_decision` is likely PROJECT identity + DATASET reference;
   cat 9 `experiment_record` is PROJECT identity + DATASET reference + USER audit.
2. Add `infra/migrations/000{N}_<name>_memory.sql` (idempotent) + mirror as `_INLINE_MIGRATION_SQL` in
   the store class.
3. Implement `mads/memory/<name>.py` (pydantic model + `<Name>Store` class). Tenancy filtering belongs
   HERE, not in tools.
4. Implement `mads/memory/<name>_tools.py` with `make_<name>_factory(store, user_id, project_id)`
   returning a factory whose `.tools()` exposes `@tool`-decorated callables. LLM-visible tool signatures
   must NOT include `user_id` or `project_id`.
5. Re-export public names from `mads/memory/__init__.py`.
6. Tests in `tests/test_<name>.py`; minimum coverage: identity, insert/upsert paths, tenancy isolation,
   `args_schema` assertion that tenancy fields are absent.
7. Demo in `eval/demo_<name>_memory.py` covering at least the user-provided examples.
8. `IMPLEMENTATION_NOTES_<NAME>.md` (design handoff) + `<NAME>_README.md` (user-facing).

**Cross-category coupling.** Categories must not import each other's source modules. References
between categories (e.g. cat 5's `datasets_used` pointing at cat 2 fingerprints) travel as plain
strings. Shared infrastructure lives in `mads/memory/embedding.py` (the `Embedder` protocol) and
should grow there, not as per-category copies.

## eval/

`eval/` holds the ADS baseline (15 cross-session-pair scenarios across Loader / EDA / Viz, 5 each)
and the forthcoming memory-comparison harness. Scenarios in `eval/scenarios.json` (now **27 total**:
15 original + **12 broad-EDA** scenarios covering Consistency / Learning-from-feedback / Efficiency /
Cross-dataset isolation / Multi-step / Adversarial / Negative); agent factories in
`eval/agent_factories.py`; results in `eval/results/<agent>/`. Current baseline: 15/15 PASS,
$0.83 across 30 Anthropic calls (`claude-sonnet-4-6`). See `eval/README.md` for repro commands.

**Broad EDA eval (12 new scenarios).** Each scenario is scored TWO ways: (1) a ground-truth-anchored
programmatic check from `eval/ground_truth.py` (uses constants computed from `baseline/data/churn_data.csv`
at import: `TELCO_CHURN_PCT`, `TELCO_CHURN_BY_CONTRACT_PCT`, `TELCO_QUALIFIED_CUSTOMER_CHURN`, etc.);
(2) a single-shot LLM judge in `eval/judge.py` (model=`claude-sonnet-4-6`, same as the agent — no
model-mixing). The two signals are reported side-by-side via `eval/compare.py --two-signal`; the
`agree` predicate is `(gt_passed implies judge_winner=="memory-aware") AND (not gt_passed implies
judge_winner!="memory-aware")`. Cross-dataset scenarios use a deterministic 120-row synthetic sales
dataset (`eval/synthetic_dataset.py`, seed=42, 7 columns); the runner materializes it to a tmp CSV
and substitutes `{SYNTHETIC_CSV}` in session B prompts. Run with:
`.venv/bin/python eval/run_with_memory.py --agent eda --both-variants --judge --scenarios <ids>`.
See `eval/results/BROAD_EVAL_RESULTS.md` for the synthesis.

## What NOT to do

- Don't add a second vector store. Cat 5 ships on sqlite-vec; the project is otherwise on SQLite.
  If you ever feel like reaching for chroma/lancedb/pgvector, route the decision through the user.
- Don't store dataframes inline in any memory row; use external artifacts + a URI reference.
- Don't expose `user_id` / `project_id` as LLM-visible tool arguments — close over them in the factory.
- Don't reach for cats 1–4's versioning pattern for event-like categories; they're append-only.
- Don't import `ai_data_science_team` from agent code; the eval harness is the only exempt callsite.
