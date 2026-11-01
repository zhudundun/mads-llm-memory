# mads — Memory-Augmented Data Scientist

`mads` is a long-term-memory layer for LLM data-science agents. It wraps [`business-science/ai-data-science-team`](https://github.com/business-science/ai-data-science-team) (ADS) and adds five SQLite-backed memory stores, a hybrid retriever, and a CLI — so the agent remembers metric definitions, dataset schemas, user preferences, data-quality issues, and prior analysis conclusions across sessions.

**Status: v0.1 — five memory categories shipped, three pending (`workflow_recipe`, `pipeline_decision`, `experiment_record`), no LangGraph supervisor yet.**

---

## Architecture

```
   user question
        │
        ▼
┌───────────────┐
│    router     │  regex; viz > loader > eda (residual)
└───────────────┘
        │
        ▼
┌───────────────┐
│   wrap_*      │  binds df on MemoryManager,
│  (wrapper)    │  injects memory @tools,
│               │  optionally appends [Memory hint] nudge
└───────────────┘
        │
        ▼
┌───────────────┐
│  Loader/EDA/  │  ADS sub-agent via adapter
│  Viz Handle   │  → AgentResult { response, tool_calls,
│  (adapter)    │                  viz_code, tokens }
└───────────────┘
        │
        ▼
┌───────────────┐
│   composer    │  template → markdown answer with
│               │  Memory consulted / Memory written
└───────────────┘
        │
        ▼
     stdout
```

Memory writes happen **only** when the LLM emits a `record_*` / `register_*` / `add_*` tool call. Memory reads happen **only** when it emits a `recall_*` / `get_*` / `find_similar_*` / `list_*` / `search_*` call. There is no second LLM pass on transcripts.

---

## Memory categories

| # | Category | Status | Scope axes |
|---|---|---|---|
| 1 | `user_preference` | shipped (not yet wired into agent) | USER required, PROJECT optional |
| 2 | `dataset_schema` | shipped + wired | DATASET required (fingerprint), USER audit |
| 3 | `metric_definition` | shipped + wired | PROJECT required, USER audit, DATASET ref |
| 4 | `data_quality_issue` | shipped (not yet wired into agent) | DATASET required, USER + PROJECT audit |
| 5 | `analysis_episode` | shipped + wired (append-only, vector-indexed) | PROJECT required, USER audit, DATASET ref |
| 6 | `workflow_recipe` | pending | — |
| 7 | `pipeline_decision` | pending | — |
| 9 | `experiment_record` | pending | — |

Cats 1–4 version rows via `valid_from`/`valid_to`. Cat 5 is append-only (each `record_analysis_episode` is immutable except for `user_feedback`). All stores share a single `threading.RLock` to serialise writes across a shared SQLite connection in WAL mode.

---

## Install

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"     # core + pytest (~3 min cold; torch ~2 GB for sentence-transformers)
pip install -e ".[eval]"    # adds ai-data-science-team for eval/ scripts
```

`mads ask` requires `ANTHROPIC_API_KEY` in the environment. Copy `.env.example` to `.env` and fill it in.

The default embedder is `HashEmbedder` (no model download). Only `AnalysisEpisodeStore` with the `SentenceTransformerEmbedder` triggers the ~80 MB model download on first call.

---

## Quickstart

```bash
# Load a CSV
mads ask "load /path/to/data.csv" --user alice

# EDA with memory injection
mads ask "describe the missing values" \
    --data /path/to/data.csv \
    --user alice --project my_project

# Ask about stored metrics
mads ask "what metric definitions do we have?" \
    --data /path/to/data.csv \
    --user alice --project my_project
```

Example output:

```
# Answer (eda)

There are 11 columns with missing-value rates above 0. ...

## Memory consulted (1)
- list_metric_definitions()

## Memory written (1)
- record_analysis_episode(question=describe the missing values, conclusion=...)

_tokens: 4128 (in=3902, out=226)_
```

---

## Memory CLI

All subcommands accept `--user / --project / --db` (or `$MADS_USER` / `$MADS_PROJECT` / `$MADS_DB_PATH`).

| Subcommand | Description |
|---|---|
| `mads memory list metric` | Active metric definitions for the project |
| `mads memory list dataset` | All registered datasets for the user |
| `mads memory list episode --limit N` | Recent analysis episodes (default 10) |
| `mads memory show metric NAME` | Current version of a metric |
| `mads memory show episode ID` | Full episode by ID |
| `mads memory history metric NAME` | All prior versions chronologically |
| `mads memory similar episode "QUERY" --k N` | Vector similarity search (default k=5) |

---

## Project layout

```
mads/
  adapters/ads.py          # ADS boundary — the ONLY file outside eval/ that imports ADS
  agents/wrapper.py        # wrap_loader / wrap_eda / wrap_viz + TurnArtifacts
  app/                     # Streamlit UI (experimental)
  dataset/                 # fingerprinting + profiling
  memory/
    user_preference.py     # cat 1
    dataset_schema.py      # cat 2
    metric_definition.py   # cat 3
    data_quality.py        # cat 4
    analysis_episode.py    # cat 5
    retrievers/            # HybridRetriever + per-category retrievers + query_classifier
    stores/                # BaseSqliteStore stub, VectorStore, ArtifactStore
    extractors/            # LLM-assisted extraction helpers (cat 2, 3, 5)
    core/                  # shared policies, scoring, schemas, namespace
  router/                  # heuristic intent classifier + context builder
  tools/                   # pandas sandboxed exec
  validation/              # memory consistency + metric usage validators
  manager.py               # MemoryManager facade (one SQLite conn, lazy factories)
  composer.py              # pure-Python markdown formatter
  cli.py                   # argparse entrypoint
eval/
  scenarios.json           # 15 eval scenarios
  harness.py               # runner
  compare.py               # baseline vs with-memory diff
  judge.py                 # LLM-as-judge
tests/                     # 283 tests — pytest -q passes
```

---

## Key architectural contracts

- **ADS is wrapped, never imported directly.** Only `mads/adapters/ads.py` and `eval/` may `import ai_data_science_team`.
- **Scoping is a declared axis model.** Each category declares axes (`USER`, `PROJECT`, `DATASET`) as `IDENTITY_REQUIRED`, `IDENTITY_OPTIONAL`, `AUDIT`, or `REFERENCE`. Tenancy is never exposed as a tool argument to the LLM.
- **DataFrames travel by reference.** Memory rows hold URIs or fingerprints, never inline bytes. No `pickle`.
- **PII scrubbing is aspirational.** Only cat 5 currently embeds (free-text analyst notes). Flag any new code path that would embed user data.

---

## Running tests

```bash
pytest -q          # 283 tests, ~30s
pytest tests/test_concurrency.py -v   # concurrency regression suite
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required for `mads ask` |
| `OPENAI_API_KEY` | — | Required for eval judge (LLM-as-judge) |
| `MADS_USER` | `$USER` | Default user identity |
| `MADS_PROJECT` | `default` | Default project scope |
| `MADS_DB_PATH` | `./.mads/memory.sqlite` | SQLite database path |

Copy `.env.example` to `.env` — never commit `.env`.
