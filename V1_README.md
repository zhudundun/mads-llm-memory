# mads v1 — memory-augmented data scientist

`mads` is a long-term-memory layer over [`business-science/ai-data-science-team`](https://github.com/business-science/ai-data-science-team) (ADS). v1 ships a CLI (`mads ask`) that routes a question to a loader / EDA / viz sub-agent, injects memory tools into that agent's tool list, captures which memories it read or wrote during the turn, and formats a final answer with explicit citations. The agent's underlying LLM is responsible for calling the memory tools — there is no second LLM pass that scrapes the transcript.

## What v1 ships

- **Five memory stores** under `mads/memory/` (each a SQLite table with its own pydantic model and `@tool` factory):
  - `user_preference`, `dataset_schema`, `metric_definition`, `data_quality_issue`, `analysis_episode`.
- **Three of the five are wired into the agent stack** through `MemoryManager`: `metric_definition`, `dataset_schema`, `analysis_episode`. The remaining two (`user_preference`, `data_quality_issue`) are shipped, tested, and importable from `mads.memory`, but **not** surfaced via the Manager, **not** injected into wrapped agents, and **not** addressable from `mads memory ...`. v1.1 wires them in by extending only Manager + wrapper + CLI registries.
- **ADS adapter** (`mads/adapters/ads.py`): the only file outside `eval/` allowed to `import ai_data_science_team`. Exposes `LoaderHandle / EDAHandle / VizHandle` with a uniform `AgentResult`.
- **Memory-aware wrapper** (`mads/agents/wrapper.py`): injects memory tools, optionally appends a prompt nudge, and captures `TurnArtifacts` (memory_reads, memory_writes).
- **Heuristic router** (`mads/router.py`): regex-based classification — viz patterns checked before loader so "plot the columns of x.csv" wins for viz. No LLM fallback.
- **Templating composer** (`mads/composer.py`): pure-Python markdown formatting. No LLM call.
- **CLI** (`mads/cli.py` + `mads/__main__.py`): `mads ask` + `mads memory list|show|history|similar ...` over stdlib `argparse`.
- **Eval extension** (`eval/run_with_memory.py` + `eval/compare.py`): drives the same 15 baseline scenarios through wrapped agents and diffs results against the baseline.
- **Test suite**: 283 passing (`pytest -q`). Brief target was ~268; covers manager, adapter, wrapper, router, composer, CLI, tenancy-end-to-end, and compare semantics.

## Architecture (what is shipped)

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

The original design diagram included Validator / Extractor / Update steps that would post-process the agent's transcript to harvest memories. **Those are intentionally omitted in v1.** Memory writes happen exclusively when the LLM emits a `record_*` / `register_*` / `add_*` tool call during the agent turn. Memory reads happen exclusively when it emits a `recall_*` / `get_*` / `find_similar_*` / `list_*` / `search_*` tool call. There is no second LLM pass.

## Install

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"          # core + pytest (~3 min cold; torch is ~2 GB)
pip install -e ".[eval]"         # adds ai-data-science-team for eval/ scripts
```

The `[dev]` extra is enough to run `mads ask`, `mads memory ...`, and the full test suite — ADS is only required when you actually invoke the agent (and that happens via the lazy `mads ask` import path). The category-5 demo and any code path that constructs `AnalysisEpisodeStore` with the default embedder will download an ~80 MB `sentence-transformers` model on first call; the `MemoryManager` default embedder is `HashEmbedder`, which avoids that.

`mads ask` additionally requires `ANTHROPIC_API_KEY` in the environment.

## Quickstart — `mads ask`

```bash
# Loader: load a file. --data is ignored when the router picks "loader" —
# the loader's job is to load.
mads ask "load /Users/wang/LLM_memory/baseline/data/churn_data.csv" --user wang

# EDA: --data is required (the agent does not load it itself).
mads ask "describe the missing values" \
    --data /Users/wang/LLM_memory/baseline/data/churn_data.csv \
    --user wang

# Memory recall: the question itself goes to EDA (the heuristic default).
# You still need --data because EDA expects a df.
mads ask "what metric definitions do we have for this project?" \
    --data /Users/wang/LLM_memory/baseline/data/churn_data.csv \
    --user wang
```

Expected output shape (composer-formatted; abbreviated):

```
# Answer (eda)

There are 11 columns with missing-value rates above 0. ...

## Memory consulted (1)
- list_metric_definitions()

## Memory written (1)
- record_analysis_episode(question=describe the missing values, conclusion=11 columns...)

_tokens: 4128 (in=3902, out=226)_
```

The `Memory consulted` / `Memory written` lists come directly from the agent's tool calls — empty lists mean the LLM did not interact with memory during this turn.

## Memory browsing — `mads memory ...`

These commands open the same SQLite db, build a `MemoryManager`, and invoke a single memory tool with `include_dataset=False` where possible (no df needed for browsing).

| Subcommand | Tool invoked | Notes |
|---|---|---|
| `mads memory list metric` | `list_metric_definitions` | Active metrics for the project. |
| `mads memory list dataset` | `list_datasets` | All registered datasets for the user. Binds a sentinel 1-row df. |
| `mads memory list episode --limit N` | `list_recent_analyses` | Default `--limit 10`. |
| `mads memory show metric NAME` | `get_metric_definition` | Currently-active version. |
| `mads memory show episode ID` | `get_analysis_episode` | Full episode by id. |
| `mads memory show dataset TABLE` | `recall_dataset` | Best-effort browse via sentinel df (fingerprint mismatch will return empty). |
| `mads memory history metric NAME` | `get_metric_history` | All prior versions chronologically. |
| `mads memory similar episode "QUERY" --k N` | `find_similar_analyses` | Vector search, default `--k 5`. |

All subcommands accept `--user / --project / --db`, resolved as: explicit flag > `$MADS_USER` / `$MADS_PROJECT` / `$MADS_DB_PATH` > defaults (`$USER`, `"default"`, `./.mads/memory.sqlite`).

## The memory loop end-to-end

Concrete A-then-B flow against one shared db file:

1. **Session A** — `mads ask "load /path/data.csv and remember we always exclude test users from analyses" --user wang`. The router picks `loader`. `wrap_loader` injects 11 memory tools (metric + episode; no dataset because no df yet) and the nudge prompt fragment. The loader's LLM may call `record_analysis_episode(question=..., conclusion="always exclude test users", assumptions=[...])`. That write hits the shared SQLite db.
2. **Session B** (later, same db) — `mads ask "load /path/data.csv" --user wang`. Same router, same tool injection. The LLM **has access to** `find_similar_analyses` and `list_recent_analyses`, and the nudge encourages recording — but as of v1, **read-side recall is opportunistic, not enforced**. The brief's read-side gap closure depends on the agent choosing to call a recall tool. The 1-scenario smoke run showed reliable A-side writes but inconsistent B-side reads. See **Open issues**.

## The Memory Manager facade

`mads/manager.py` owns exactly one `sqlite3.connect(...)` (with `check_same_thread=False`; see the deviation note below) and hands out three lazily-built factories.

```python
from mads.manager import MemoryManager
from mads.memory import HashEmbedder

mgr = MemoryManager(
    user_id="wang", project_id="churn", db_path="./.mads/memory.sqlite",
    embedder=HashEmbedder(),  # optional; HashEmbedder is the default
)

metric_tools = mgr.metric.tools()        # 5 tools, no user/project in signatures
episode_tools = mgr.episode.tools()      # 6 tools

import pandas as pd
df = pd.read_csv("data.csv")
mgr.bind_dataframe(df, dataset_name="data", table_name="data")
dataset_tools = mgr.dataset_tools()      # 6 tools; fingerprinted to THIS df

all_tools = mgr.all_tools(include_dataset=True)  # 17 tools
mgr.close()                                       # idempotent; releases the conn
```

`bind_dataframe()` is **required** before `dataset_tools()` — calling it without binding raises `RuntimeError`. Re-binding with a different df invalidates the cached tool list.

The constructor uses `check_same_thread=False` on the SQLite connection so the eval harness (which runs each session in a daemon worker thread) and LangGraph's tool-node ThreadPool can both reach the connection. This is a known deviation from a naive single-thread setup; the connection is still used by one thread at a time per turn.

## The ADS adapter

`mads/adapters/ads.py` is the single chokepoint for ADS imports. All callers see `AgentResult`:

```python
@dataclass
class AgentResult:
    response: Optional[str]                       # AI's natural-language reply
    tool_calls: list = field(default_factory=list)
    viz_code: Optional[str] = None                # only set for viz
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
```

`extra_tools=` on `make_loader / make_eda` is **not** passed via a `tools=` constructor kwarg — the ADS `DataLoaderToolsAgent` and `EDAToolsAgent` do not accept one. The adapter uses a `_PatchedTools` context manager that swaps the module-level tools list (`ai_data_science_team.agents.data_loader_tools_agent.tools` and `ai_data_science_team.ds_agents.eda_tools_agent.EDA_TOOLS`) during agent construction and restores it afterward. This is the load-bearing mechanism; if ADS upstream changes those module attribute names, only the adapter needs updating.

`make_viz` accepts `extra_tools=` for API symmetry but the underlying `DataVisualizationAgent` is a code-generation pipeline (it never calls tools). The injected memory tools are stored on the handle for inspection but are **inert** at invoke time. This is documented as a v1 limitation.

## Eval extension

`eval/run_with_memory.py` drives the existing 15 baseline scenarios through `wrap_loader / wrap_eda / wrap_viz`. Sessions A and B share the same SQLite db so A's writes are visible to B.

```bash
# Smoke test — one scenario, fresh db, default nudge ON (~$0.05).
.venv/bin/python eval/run_with_memory.py --agent loader --scenario loader_01_snake_case

# Full run — DO NOT execute without user approval (~$1–2).
.venv/bin/python eval/run_with_memory.py --agent all
```

Flags:
- `--accumulate`: share one db across all scenarios (realistic; default is per-scenario fresh db).
- `--no-nudge-memory`: disable the prompt-level "[Memory hint]" appendix that asks the LLM to record before answering.
- `--user / --project / --timeout-s`: tenancy and per-session timeout (default 180s).

Outputs land in `eval/results/<agent>__with_memory/runs.jsonl` and `SUMMARY.md`.

`eval/compare.py` reads baseline (`eval/results/<agent>/runs.jsonl`) and with-memory runs.jsonl and emits a per-scenario diff table to `eval/results/COMPARE.md`. Labels: `gap-closed`, `unchanged`, `regressed`, `still-helping`, `baseline-missing`, `v1-missing`, `both-missing`. Exit 0 if at least one row is `gap-closed`; exit 1 otherwise.

```bash
.venv/bin/python eval/compare.py --agent all
cat eval/results/COMPARE.md
```

## Testing

```bash
pytest -q                                    # 283 passed
pytest tests/test_manager.py -q              # 23
pytest tests/test_wrapper.py -q              # 12
pytest tests/test_router.py -q               # 15
pytest tests/test_composer.py -q             # 10
pytest tests/test_cli.py -q                  # 19
pytest tests/test_tenancy_e2e.py -q          # 7
pytest tests/test_adapters_ads.py -q         # 11
pytest tests/test_compare.py -q              # 8
```

Coverage:
- **manager**: shared-connection guarantee; bind/swap df; tool counts (11 vs 17); `close()` idempotent; tenancy isolation across projects; db_path precedence (arg > env > default); rejects empty `user_id` / `project_id`; embedder injection; cached factories.
- **adapter**: `AgentResult` shape; handles return `AgentResult`; `extra_tools` reach the module-level patch; token usage flows from `_UsageCallback`; viz_code only set for viz; ADS import errors propagate cleanly.
- **wrapper**: loader injects 11 tools; eda injects 17; `TurnArtifacts` captures reads vs writes; nudge applied/skipped per flag; `df=None` raises.
- **router**: unambiguous classifications; viz wins over loader on "plot the columns"; case-insensitive; deterministic; pure (no I/O).
- **composer**: header; agent_name; viz_code block conditional; memory sections conditional; tokens line; truncation; None response handling.
- **CLI**: each subcommand exits 0 on happy path with mocked wrapper; missing `--user` → exit 2; env var precedence; `--db` wins over env; `--help` exits 0.
- **tenancy_e2e**: two-project isolation for metrics + episodes; same-project cross-user visibility; `args_schema` excludes `user_id` and `project_id` for all 3 factories.
- **compare**: classification labels; exit-code semantics; handles missing files; stable byte output.

## Architectural contracts (what is load-bearing)

These are pulled from `CLAUDE.md`; they are not aspirational — the code enforces them.

- **Tenancy at the store layer.** `WHERE user_id=… AND project_id=…` lives in stores; tool factories close over tenancy. No LLM-visible `args_schema` contains `user_id` or `project_id`. The `test_tenancy_e2e.py` tests assert this for the metric, dataset, and episode factories.
- **ADS imports only in `mads/adapters/ads.py`** (plus `eval/`). Any other production module that imports `ai_data_science_team` breaks the adapter contract.
- **One shared SQLite connection per `MemoryManager`** — used by all three stores. Closed exactly once via `close()`.
- **sqlite-vec is the only vector store.** Embedding lives in `mads/memory/embedding.py`; do not introduce a second vector backend.
- **No auto-extraction.** Memory writes require the LLM to emit a write-class tool call during the turn. Memory reads require it to emit a read-class tool call. The composer is templating; no second LLM pass.
- **DataFrames travel by reference, not value.** Memory rows hold URIs or content-addressed fingerprints; never inline bytes.

## Open issues (before v1 ships externally)

- **Read-side recall gap.** The 1-scenario smoke run showed session A writes succeed reliably under the nudge, but session B's LLM does not proactively call `find_similar_analyses` / `list_recent_analyses` / `get_metric_definition` without a more explicit hint. Without a read-side nudge or a v1.1 supervisor pre-pass, the gap-closure rate measured by `eval/compare.py` will be low.
- **Viz tool injection is inert.** `DataVisualizationAgent` is a code-gen pipeline — it does not call tools. Memory writes from a viz turn currently have no path; only the wrapper's post-invoke classification runs (which sees zero memory tool calls).
- **Full 15-scenario A/B eval not run.** Brief budget is ~$1–2; user must approve.
- **`mads ask` against the real LLM not smoke-tested end-to-end.** The CLI tests use a mocked wrapper. The real-LLM path is exercised only via `eval/run_with_memory.py --scenario`.
- **PII scrubbing module still aspirational.** `CLAUDE.md` flags this. Only cat 5 currently embeds, and it embeds free-text analyst notes only.
- **No real migration runner.** Each store ships an `_INLINE_MIGRATION_SQL` constant and runs it on first connect. Adding a real migration tool is a future cleanup.
- **CLAUDE.md version line drift.** `CLAUDE.md` still describes the repo as "v0.0.1" while `pyproject.toml` and `mads/__init__.py` are `0.1.0`. Minor doc-only inconsistency.

## Roadmap to v1.1

- **Read-side recall nudge.** Either a stronger prompt fragment that names the recall tools explicitly, or a one-pass supervisor that calls `find_similar_analyses` before handing off to the wrapped agent.
- **Viz integration via post-codegen memory write.** Once viz generates code, a small post-step records a `register_dataset` / `record_analysis_episode` for the chart produced.
- **Real PII scrubber.** Stand up `mads/memory/scrub.py` and route every embedding through it.

## Related

- `V1_DESIGN_BRIEF.md` — the architect's spec; ground truth for "should be".
- `CLAUDE.md` — current contracts, common commands, and category status.
- `mads/memory/METRIC_DEFINITION_README.md`, `ANALYSIS_EPISODE_README.md`, `DATASET_SCHEMA_README.md` — per-category user-facing docs.
- `mads/memory/IMPLEMENTATION_NOTES_*.md` — per-category design handoffs.
