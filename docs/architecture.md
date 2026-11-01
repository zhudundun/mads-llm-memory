# mads architecture

`mads` is a memory-augmented analyst agent layered over the
[`ai-data-science-team`](https://github.com/business-science/ai-data-science-team)
(ADS) sub-agents. v1 ships a CLI that routes a question to one of three
wrapped ADS sub-agents, injects memory `@tool` callables into that agent's
tool list, captures which memories the LLM read or wrote during the turn,
and emits a pure-template final answer. This document describes the shape
of the v1 stack on disk and the v1.1 architectural scaffolds the
Implementer just landed alongside it.

## The user-facing flow

```
   user question
        |
        v
+----------------+
|   router       |  regex; viz > loader > eda (residual)
+----------------+
        |
        v
+----------------+
|   wrap_*       |  binds df on MemoryManager (eda/viz),
|   (wrapper)    |  injects 11 or 17 memory @tools,
|                |  optionally appends [Memory hint] nudge
+----------------+
        |
        v
+----------------+
| Loader / EDA / |  ADS sub-agent invoked via adapter
| Viz Handle     |  -> AgentResult { response, tool_calls,
| (adapter)      |                   viz_code, tokens }
+----------------+
        |
        v
+----------------+
|  classify      |  split tool_calls into memory_reads /
|  tool_calls    |  memory_writes by name (no SQL re-query)
+----------------+
        |
        v
+----------------+
|  composer      |  template -> markdown:
|                |    Answer / Generated code /
|                |    Memory consulted / Memory written / tokens
+----------------+
        |
        v
   stdout
```

The original design diagram from V1_DESIGN_BRIEF included three additional
boxes between the agent and the composer — Validator, Memory Extractor,
Memory Update. **Those are deferred in v1 and surface only as
scaffolds in v1.1.** Writes happen exclusively when the LLM emits a
`record_*` / `register_*` / `add_*` tool call during the turn; reads
exclusively when it emits a `recall_*` / `get_*` / `find_similar_*` /
`list_*` / `search_*` call. No second LLM pass scrubs the transcript.

## Top-level layout

```
mads/
  __init__.py                                    (shipped) re-exports MemoryManager
  __main__.py                                    (shipped) python -m mads entry
  cli.py                                         (shipped) argparse CLI
  manager.py                                     (shipped) MemoryManager facade
  composer.py                                    (shipped) templating composer
  adapters/
    __init__.py                                  (shipped)
    ads.py                                       (shipped) sole ADS importer
  agents/
    __init__.py                                  (shipped)
    types.py                                     (shipped) AgentResult, TurnArtifacts
    wrapper.py                                   (shipped) CANONICAL wrap_* impl
  wrappers/
    __init__.py                                  (shim)    re-exports wrap_*
    wrapper.py                                   (shim)    re-exports wrap_*
  router/
    __init__.py                                  (shipped) re-exports route
    intent_classifier.py                         (shipped) heuristic router
    context_builder.py                           (stub - v1.1) read-side recall
  dataset/
    __init__.py                                  (shipped)
    fingerprint.py                               (shipped) compute_fingerprint
    profiler.py                                  (stub - v1.1) describe()-replacement
    schema_summary.py                            (stub - v1.1) markdown summary
  validation/
    __init__.py                                  (shipped)
    result_validator.py                          (stub - v1.1) answer sanity
    metric_usage_validator.py                    (stub - v1.1) metric formula check
    memory_consistency_validator.py              (stub - v1.1) cross-cat sweep
  app/
    __init__.py                                  (shipped)
    cli.py                                       (shim)    re-exports mads.cli.main
    streamlit_app.py                             (stub - v1.1) web UI
  memory/
    __init__.py                                  (shipped) re-exports 5 stores + factories
    embedding.py                                 (shipped) Embedder protocol + 2 impls
    user_preference.py                           (shipped) cat 1 store
    preference_tools.py                          (shipped) cat 1 tool factory
    dataset_schema.py                            (shipped) cat 2 store
    dataset_tools.py                             (shipped) cat 2 tool factory
    metric_definition.py                         (shipped) cat 3 store
    metric_tools.py                              (shipped) cat 3 tool factory
    data_quality.py                              (shipped) cat 4 store
    data_quality_tools.py                        (shipped) cat 4 tool factory
    analysis_episode.py                          (shipped) cat 5 store (sqlite-vec)
    analysis_episode_tools.py                    (shipped) cat 5 tool factory
    core/
      __init__.py                                (shipped) re-exports MemoryManager
      manager.py                                 (shim)    re-exports MemoryManager
      schemas.py                                 (stub - v1.1) centralized pydantic
      namespace.py                               (stub - v1.1) Namespace value
      policies.py                                (stub - v1.1) write/retention rules
      scoring.py                                 (stub - v1.1) hybrid Scorer
    stores/
      __init__.py                                (shipped) re-exports 5 concrete stores
      sqlite_store.py                            (stub - v1.1) BaseSqliteStore
      vector_store.py                            (stub - v1.1) BaseVectorStore
      artifact_store.py                          (stub - v1.1) ArtifactStore
    retrievers/
      __init__.py                                (shipped)
      metric_retriever.py                        (stub - v1.1)
      dataset_retriever.py                       (stub - v1.1)
      episode_retriever.py                       (stub - v1.1)
      hybrid_retriever.py                        (stub - v1.1) cross-cat fusion
    extractors/
      __init__.py                                (shipped)
      metric_extractor.py                        (stub - v1.1) mine definitions
      dataset_extractor.py                       (stub - v1.1) mine schema facts
      episode_extractor.py                       (stub - v1.1) mine conclusions
    versioning/
      __init__.py                                (shipped)
      version_manager.py                         (stub - v1.1) centralized close-insert
      conflict_detector.py                       (stub - v1.1) concurrent-write triage
```

## Cross-cutting contracts

These are pulled from `CLAUDE.md`; the code enforces them.

- **Tenancy lives at the store layer.** `WHERE user_id=? AND project_id=?`
  is applied inside each store; tool factories close over tenancy. No
  LLM-visible `args_schema` exposes `user_id` or `project_id`.
- **ADS is imported only from `mads/adapters/ads.py`** (plus `eval/`, transitionally).
  Every other production module sees ADS through `AgentResult` and the
  three handle classes.
- **One shared SQLite connection per `MemoryManager`.** The connection
  is loaded with `sqlite-vec` once at construction and shared across
  all three in-scope stores; `close()` is idempotent.
- **sqlite-vec is the only vector backend.** All embeddings go through
  `mads/memory/embedding.py`; the `vec0` virtual table lives next to
  the `analysis_episode_memory` table.
- **No auto-extraction in v1.** Writes happen only when the agent's LLM
  emits a write-class tool call. The composer is templating only — no
  second LLM call ever sees the transcript.
- **DataFrames travel by reference.** Memory rows hold URIs or
  content-addressed fingerprints, never inline bytes.

## The wrapper shim inversion

The proposed v1.1 layout places the canonical wrapper at
`mads/wrappers/wrapper.py`. The Implementer kept the canonical
implementation at `mads/agents/wrapper.py` and made
`mads/wrappers/wrapper.py` a re-export shim. The reason is purely a
test-mechanics constraint: the existing wrapper tests use
`monkeypatch.setattr("mads.agents.wrapper.make_loader", ...)` to
intercept the underlying `make_loader` / `make_eda` / `make_viz`
adapter calls. `monkeypatch.setattr` only affects the module where the
name is defined, so any `wrap_*` function looking up `make_loader` via
its own module's globals must live in the same module the tests patch.
Physically relocating the wrapper to `mads/wrappers/wrapper.py` would
silently break those interception points without breaking any imports.

The shim direction therefore stays inverted relative to the brief:
`mads/wrappers/wrapper.py` re-exports the public surface from
`mads/agents/wrapper.py`. The shim's docstring records the migration
trigger — when the wrapper grows new surface (a LangGraph supervisor,
per-agent specializations) it'll move to `mads/wrappers/` and the
direction will flip.

## The architectural scaffolds (v1.1+)

Every scaffold raises `NotImplementedError` with a docstring covering
design intent, the precise v1.1 trigger that would justify implementing
it, and the status banner `Status: STUB`.

- **`memory/extractors/`** — three deterministic miners
  (`metric_extractor`, `dataset_extractor`, `episode_extractor`) that
  would scan agent transcripts for facts the LLM forgot to record via
  the tool-call path. Trigger: with-memory eval showing gap-closure
  rate <30%.

- **`memory/versioning/`** — `version_manager.close_and_insert(...)`
  centralizes the close-prior / insert-new pattern duplicated across
  cats 1-4 today; `conflict_detector.detect(...)` triages the case of
  overlapping concurrent writes to the same identity tuple. Trigger:
  6th category arrives or first parallel-write conflict observed.

- **`validation/`** — three post-turn validators
  (`result_validator`, `metric_usage_validator`,
  `memory_consistency_validator`) that would inspect the wrapper's
  `TurnArtifacts` for answer-level errors the per-tool arg validation
  can't catch (forbidden imports, columns that don't exist, named
  metrics computed differently from their stored definition,
  cross-category foreign-key drift). Triggers vary per validator.

- **`app/streamlit_app.py`** — a future Streamlit UI for browsing
  memory and confirming candidate extractions. Deferred because the
  ADS "Pipeline Studio" upstream doesn't exist
  (`docs/base_agent_api_notes.md`) so there's no canonical visual
  idiom to follow, and the CLI is the single tested entry point for v1.

- **`memory/core/{schemas,namespace,policies,scoring}.py`** —
  centralization targets. `schemas` would unify the pydantic models that
  currently live inline in each category. `namespace` would formalize the
  `(user_id, project_id)` tuple into a first-class value. `policies` is
  the seat for retention/redaction config. `scoring` is the multi-signal
  ranker the hybrid retriever would consume.

- **`memory/stores/{sqlite_store,vector_store,artifact_store}.py`** —
  abstract base classes. `BaseSqliteStore` would factor out the
  per-category SQLite boilerplate; `BaseVectorStore` would wrap the
  `vec0` dance currently inline in `analysis_episode.py`;
  `ArtifactStore` mints stable URIs for parquet / PNG / pickle artifacts
  so memory rows can reference them by URI alone.

- **`memory/retrievers/*`** — today retrieval is exposed through each
  store's `get`/`list`/`search`/`find_similar` methods, called directly
  from LangChain tools. The retrievers subpackage scaffolds a future
  indirection so a hybrid retriever can fuse signals across categories
  without crossing category boundaries.

- **`dataset/{profiler,schema_summary}.py`** — `DatasetProfiler.profile(df)`
  would produce a deterministic structural summary (row count, dtypes,
  null rates, candidate PK) that runs before the EDA agent so it doesn't
  waste tokens on cheap `describe()`-equivalents; `SchemaSummary.render(...)`
  formats that profile plus `dataset_memory` / `column_memory` recalls into
  a markdown block the context builder can splice into the prompt.

- **`router/context_builder.py`** — the read-side mirror of the wrapper's
  write-side nudge. After the router picks an agent and before the wrapper
  invokes it, the context builder would query the memory layer for
  top-k relevant recalls and weave them into the prompt. Trigger: the
  V1_README open issue — session B's LLM doesn't proactively call
  `find_similar_analyses` even when relevant episodes exist.

## Build order to v1.1

1. **Close the read-side recall gap.** The single observable functional
   gap in v1 is the LLM not reading its own memory when relevant. The
   context builder + a stronger read-side prompt fragment is the
   highest-impact next step.
2. **Wrap `DataCleaningAgent`.** Per `docs/base_agent_api_notes.md` this
   is the cheapest base-agent addition: same react-graph shape as
   loader/EDA, same `invoke_agent(data_raw, user_instructions)`
   signature, no novel ADS subsystem. Adds the natural home for the
   `data_quality_issue` category (cat 4) to surface in the agent stack.
3. **Prototype one extractor.** `episode_extractor` is the right first
   step: episodes are append-only (false-positive cost is bounded —
   duplicates just inflate the embedding index), and the brief flags
   episode catch-up as the single highest-impact lever for downstream
   recall quality.

## Pointers

- `V1_DESIGN_BRIEF.md` — the architect's v1 spec; ground truth for "should be".
- `V1_README.md` — user-facing v1 docs (install, quickstart, open issues).
- `docs/base_agent_api_notes.md` — ADS sub-agent characterization (six
  agents, three wrapped in v1, two candidates for v1.1).
- `docs/memory_design.md` — deeper dive into the memory subsystem.
- `CLAUDE.md` — architectural contracts, common commands, category status.
