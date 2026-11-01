# ADS Baseline

This directory is the **baseline** for the `mads` (memory-augmented data
scientist) project described in `/Users/wang/LLM_memory/CLAUDE.md`. It pins
down the three ADS (`ai-data-science-team`) sub-agents that the memory layer
will sit on top of — `DataLoaderToolsAgent`, `EDAToolsAgent`, and
`DataVisualizationAgent` — verifies they import, instantiate, and expose the
public surface we plan to wrap, and freezes their constructor / method
signatures so that downstream code (especially `mads/adapters/ads.py`) has a
stable contract to depend on. ADS is pre-1.0 (`0.0.0.9017`); the whole point
of this baseline is to catch its drift early.

## TL;DR status

| Agent                  | Import path                                       | Test script        | Status        | Notes                                                                  |
|------------------------|---------------------------------------------------|--------------------|---------------|------------------------------------------------------------------------|
| `DataLoaderToolsAgent` | `ai_data_science_team.agents`                     | `test_loader.py`   | OFFLINE PASS  | 5-node langgraph; no `data_raw` arg; loads data itself.                |
| `EDAToolsAgent`        | `ai_data_science_team.ds_agents` (NOT `.agents`)  | `test_eda.py`      | OFFLINE PASS  | `data_raw` is a kwarg on `invoke_agent`.                               |
| `DataVisualizationAgent`| `ai_data_science_team.agents`                    | `test_viz.py`      | OFFLINE PASS  | `data_raw` is the **first positional** arg on `invoke_agent`.          |

All three exited `0`, stderr empty. None of the three has been run **LIVE**
yet because `OPENAI_API_KEY` was not set; live runs are gated solely on that
env var. See `## Going live`.

## Repro from scratch

Absolute paths throughout. Working directory does not matter; the scripts use
absolute paths internally.

```bash
# 1. Python + venv (already done; recreate only if the venv is wiped)
/opt/homebrew/bin/python3.13 -m venv /Users/wang/LLM_memory/.venv
source /Users/wang/LLM_memory/.venv/bin/activate

# 2. Install ADS + langchain stack. Exact frozen versions are in
#    SETUP_REPORT.md §2 (do not duplicate here). At minimum:
pip install \
  ai-data-science-team langchain langchain-openai langchain-community \
  langchain-experimental langgraph openai pandas plotly pyarrow ipykernel

# 3. Confirm the interpreter path baked into the test scripts:
cat /Users/wang/LLM_memory/baseline/.venv_python
# -> /Users/wang/LLM_memory/.venv/bin/python

# 4. The churn dataset (977,501 bytes) auto-downloads on first run of any
#    test script. Manual fetch is optional:
curl -L -o /Users/wang/LLM_memory/baseline/data/churn_data.csv \
  https://raw.githubusercontent.com/business-science/ai-data-science-team/master/data/churn_data.csv

# 5. Run any single test, offline (no key needed). The script auto-detects
#    OPENAI_API_KEY and falls back to a structural sanity check when absent.
/Users/wang/LLM_memory/.venv/bin/python /Users/wang/LLM_memory/baseline/test_loader.py
/Users/wang/LLM_memory/.venv/bin/python /Users/wang/LLM_memory/baseline/test_eda.py
/Users/wang/LLM_memory/.venv/bin/python /Users/wang/LLM_memory/baseline/test_viz.py
```

Idempotency: all three scripts cache the CSV at
`/Users/wang/LLM_memory/baseline/data/churn_data.csv`. Subsequent runs are
no-ops for data.

## Going live

Each script switches to live mode the moment `OPENAI_API_KEY` is present in
the environment.

```bash
export OPENAI_API_KEY=sk-...

/Users/wang/LLM_memory/.venv/bin/python /Users/wang/LLM_memory/baseline/test_loader.py \
  > /Users/wang/LLM_memory/baseline/results/loader.out \
  2> /Users/wang/LLM_memory/baseline/results/loader.err

/Users/wang/LLM_memory/.venv/bin/python /Users/wang/LLM_memory/baseline/test_eda.py \
  > /Users/wang/LLM_memory/baseline/results/eda.out \
  2> /Users/wang/LLM_memory/baseline/results/eda.err

/Users/wang/LLM_memory/.venv/bin/python /Users/wang/LLM_memory/baseline/test_viz.py \
  > /Users/wang/LLM_memory/baseline/results/viz.out \
  2> /Users/wang/LLM_memory/baseline/results/viz.err
```

Expected live outputs per script:

- `test_loader.py` — prints `df.shape`, `df.dtypes`, `df.head()` from
  `agent.get_artifacts(as_dataframe=True)`; the AI message; a compact dump
  of `agent.get_tool_calls()`.
- `test_eda.py` — prints the AI message, a tool-call name+count summary, and
  an artifact key/shape description.
- `test_viz.py` — prints `type(fig)`, first 30 lines of the generated code,
  recommended-steps markdown, log-summary markdown, **and writes**
  `/Users/wang/LLM_memory/baseline/results/viz_plot.html` plus
  `/Users/wang/LLM_memory/baseline/results/viz_generated.py`.

Each live block is wrapped in `try/except` so a mid-pipeline failure still
produces useful output in stdout/stderr.

## ADS public surface we depend on

This is the frozen contract the future `mads/adapters/ads.py` wraps. Anything
outside this list is fair game for ADS to break without breaking us. See
`SETUP_REPORT.md` §3 for the full constructor docstrings.

```python
# Loader -----------------------------------------------------------------
from ai_data_science_team.agents import DataLoaderToolsAgent
agent = DataLoaderToolsAgent(model=llm)                # llm: ChatOpenAI-like
agent.invoke_agent(user_instructions="...")            # no data_raw
df    = agent.get_artifacts(as_dataframe=True)
msg   = agent.get_ai_message(markdown=False)
tools = agent.get_tool_calls()

# EDA --------------------------------------------------------------------
from ai_data_science_team.ds_agents import EDAToolsAgent     # NOT .agents
agent = EDAToolsAgent(model=llm)
agent.invoke_agent(user_instructions="...", data_raw=df)     # data_raw kwarg
arts  = agent.get_artifacts(as_dataframe=False)
msg   = agent.get_ai_message(markdown=False)
tools = agent.get_tool_calls()

# Visualization ----------------------------------------------------------
from ai_data_science_team.agents import DataVisualizationAgent
agent = DataVisualizationAgent(model=llm, n_samples=30,
                               bypass_recommended_steps=False,
                               bypass_explain_code=False)
agent.invoke_agent(df, user_instructions="...", max_retries=3)  # df FIRST positional
fig      = agent.get_plotly_graph()
code_str = agent.get_data_visualization_function()
steps_md = agent.get_recommended_visualization_steps(markdown=True)
log_md   = agent.get_log_summary(markdown=True)
```

All three classes inherit from a langgraph `CompiledStateGraph`, so they
*also* expose the full langgraph runnable surface (`invoke`, `ainvoke`,
`stream`, `astream`, `batch`, `with_config`, `get_graph`). Treat that as an
escape hatch, not the primary interface — the ADS-specific methods are the
documented surface.

### Three known traps (from SETUP_REPORT §6, "What NOT to do")

1. **`EDAToolsAgent` lives in `ai_data_science_team.ds_agents`**, not
   `.agents`. Same for any other "tools agent" under `ds_agents/` or
   `ml_agents/`.
2. **`DataVisualizationAgent.invoke_agent(data_raw, ...)`** — `data_raw` is
   the first positional argument, *before* `user_instructions`. This is
   the opposite of `EDAToolsAgent`, which takes `user_instructions` first.
3. **`DataLoaderToolsAgent` does not accept `data_raw`.** Loading is the
   agent's job; pass the file path in `user_instructions` instead.

## What we observed

All three runs were OFFLINE-mode structural checks; live LLM behaviour
remains to be validated once `OPENAI_API_KEY` is provided.

- All three agents inherit from a langgraph `CompiledStateGraph` via ADS
  `BaseAgent` (MRO confirmed in `results/loader.out`).
- `DataLoaderToolsAgent` compiles to a **5-node linear graph**:
  `__start__ -> prepare_messages -> react_agent -> post_process -> __end__`.
  Confirms it is a langgraph ReAct agent with pre/post hooks.
- The churn dataset is `(7043, 21)`, mostly object dtypes; `TotalCharges`
  is `str` (whitespace for brand-new customers) — a real EDA run should
  surface this as a data-quality issue (a natural memory-layer test).
  `Contract` distribution: Month-to-month 3875, Two year 1695, One year
  1473. `Churn`: No 5174 / Yes 1869. Source: `results/eda.out`,
  `results/viz.out`.
- Construction is otherwise lazy — no network call happens at
  `DataLoaderToolsAgent(model=llm)` or `EDAToolsAgent(model=llm)` time.
  But see the next section for `ChatOpenAI`'s eager credential validation.
- `ai-data-science-team` does not bundle any sample data (see SETUP_REPORT
  §4); we fetch `churn_data.csv` from the upstream GitHub repo on first
  run and cache it.

## Known issues / gotchas

- **Eager credential validation in `openai==2.39.0`.** `ChatOpenAI(model=...)`
  now raises `openai.OpenAIError: Missing credentials` at construction time
  if no key / env var is set, because `openai-python` ≥ 2 builds an async
  client eagerly inside `_client.__init__`. Workaround in offline scripts:
  `ChatOpenAI(..., api_key="sk-offline-placeholder")`. Live mode leaves the
  constructor unparameterized so it picks up the real env var. See
  `results/eda.md` for the postmortem.
- **No bundled sample data.** ADS's `site-packages` tree contains no
  `*.csv` / `*.parquet` / `*.json` / `*.xlsx`. The baseline scripts
  download `churn_data.csv` themselves.
- **ADS version is `0.0.0.9017`** — pre-1.0, development-style version, no
  top-level `__version__` attribute (use `importlib.metadata.version(
  "ai-data-science-team")`). Expect API churn; the adapter is our shock
  absorber per `CLAUDE.md`.
- **Large transitive footprint.** ADS pulls streamlit, xgboost, scipy,
  altair, scikit-learn, etc. The venv is heavy. Acceptable for baseline;
  worth revisiting if we ever containerize.
- **`uv` is not on PATH** on this machine — stdlib `venv` + `pip` were
  used. `brew install uv` if we want faster reinstalls later.
- **Minor cross-report nit:** `results/loader.md` claims the CSV is *not*
  downloaded in offline mode, but `test_eda.py` and `test_viz.py` *do*
  fetch it during their own offline runs, so by the time anyone reads
  loader's offline report the file already exists. The CSV at
  `/Users/wang/LLM_memory/baseline/data/churn_data.csv` (977501 bytes) is
  the cached copy. Not a bug, just inconsistent prose.

## Next steps (for the memory layer)

Concrete hooks for `mads`. The naming below maps directly to categories in
`CLAUDE.md` §"The 8 memory categories" and to the adapter contract in
`mads/adapters/ads.py`.

- **Wrap point.** `invoke_agent` is the single, uniform call point across
  all three agents. The adapter should expose a function
  `run_ads(agent_kind, user_instructions, data_raw=None) -> AgentResult`
  that dispatches on `agent_kind` and handles the three positional/keyword
  quirks called out in `## ADS public surface we depend on`.
- **Curator probes.** After each `invoke_agent`, the three uniform "what
  just happened" probes are `get_ai_message()`, `get_tool_calls()`, and
  `get_artifacts()` (or `get_plotly_graph()` + `get_data_visualization_function()`
  for viz). These are what the curator consumes to mint memories:
  `data_quality_issue` and `schema_fact` from loader/EDA artifacts;
  `workflow_recipe.payload.code` from
  `DataVisualizationAgent.get_data_visualization_function()` verbatim;
  `analysis_conclusion` from the AI message.
- **Reference-not-value rule.** Per `CLAUDE.md`, dataframes never travel
  inline through state. The adapter must materialize `agent.get_artifacts(
  as_dataframe=True)` to the artifact store and emit a URI; subagents
  reload by URI. This contract is easiest to enforce *at the adapter
  boundary*, where we already translate ADS quirks.
- **Memory-recall canary.** Concrete sample question to validate memory
  recall against once the layer is in place:
  *"In churn_data.csv, `TotalCharges` is a string with whitespace for
  brand-new customers (tenure=0). Remember this as a data-quality issue
  and apply the same coercion next time you load this dataset."*
  This exercises requirement #4 (data-quality issues) and requirement #6
  (workflow recipes) in one shot, against a fact the offline EDA run
  already surfaced (`results/eda.out` shows `TotalCharges: str`).
- **API drift tripwire.** Convert the signatures in
  `## ADS public surface we depend on` into a doctest or pytest in
  `tests/test_ads_contract.py`. Any change in ADS that breaks one of
  those signatures should fail CI, not silently corrupt a downstream
  node.

## File map

```
/Users/wang/LLM_memory/baseline/
|-- README.md             # this file
|-- SETUP_REPORT.md       # initial env + ADS-contract report (source of truth for versions)
|-- .venv_python          # interpreter path: /Users/wang/LLM_memory/.venv/bin/python
|-- test_loader.py        # DataLoaderToolsAgent baseline test (offline + live)
|-- test_eda.py           # EDAToolsAgent baseline test (offline + live)
|-- test_viz.py           # DataVisualizationAgent baseline test (offline + live)
|-- data/
|   `-- churn_data.csv    # 977,501 B; cached from business-science upstream
`-- results/
    |-- loader.md         # loader run report (status, observations, repro)
    |-- loader.out        # loader stdout
    |-- loader.err        # loader stderr (empty)
    |-- eda.md            # eda run report
    |-- eda.out           # eda stdout
    |-- eda.err           # eda stderr (empty)
    |-- viz.md            # viz run report
    |-- viz.out           # viz stdout
    `-- viz.err           # viz stderr (empty)
                          # live runs will additionally create:
                          #   viz_plot.html, viz_generated.py
```
