# ADS Baseline Setup Report

## 1. Python interpreter and venv

- Chosen interpreter: **Python 3.13.2** (`/opt/homebrew/bin/python3.13`)
- Venv path: **`/Users/wang/LLM_memory/.venv`**
- Venv python: **`/Users/wang/LLM_memory/.venv/bin/python`** (also written to `/Users/wang/LLM_memory/baseline/.venv_python`)
- Created via stdlib `venv` (uv was not on PATH).
- No fallback to 3.12 / 3.11 needed — `ai-data-science-team` and all transitive deps installed cleanly under 3.13.2.

## 2. Installed package versions (pip freeze excerpt)

```
ai-data-science-team==0.0.0.9017
anthropic==0.105.2
ipykernel==7.2.0
langchain==1.3.2
langchain-anthropic==1.4.4
langchain-classic==1.0.7
langchain-community==0.4.2
langchain-core==1.4.0
langchain-experimental==0.4.2
langchain-ollama==1.1.0
langchain-openai==1.2.2
langchain-protocol==0.0.16
langchain-text-splitters==1.1.2
langgraph==1.2.3
langgraph-checkpoint==4.1.1
langgraph-prebuilt==1.1.0
langgraph-sdk==0.4.2
openai==2.39.0
pandas==3.0.3
plotly==6.7.0
pyarrow==24.0.0
```

Notable other transitive deps pulled in by ADS: `streamlit==1.58.0`, `xgboost==3.2.0`, `scikit-learn==1.8.0`, `numpy==2.4.6`, `pydantic==2.13.4`, `tiktoken==0.13.0`, `ollama==0.6.2`, `streamlit-flow-component==1.6.1`.

## 3. The three target agents (verified imports + signatures)

All three classes import cleanly under Python 3.13. The expected `DataLoaderToolsAgent` and `DataVisualizationAgent` live in `ai_data_science_team.agents`; `EDAToolsAgent` lives in `ai_data_science_team.ds_agents` (NOT in `agents`).

All three inherit from a langgraph `CompiledStateGraph`, so they expose the full langgraph runnable surface (`invoke`, `ainvoke`, `stream`, `astream`, `batch`, `with_config`, etc.) in addition to ADS-specific helpers. The ADS-specific public methods are what we care about for the baseline tests.

### 3a. DataLoaderToolsAgent

- Import: `from ai_data_science_team.agents import DataLoaderToolsAgent`
- Module: `ai_data_science_team.agents.data_loader_tools_agent`
- Constructor:
  ```python
  DataLoaderToolsAgent(
      model,
      create_react_agent_kwargs: Optional[Dict] = {},
      invoke_react_agent_kwargs: Optional[Dict] = {},
      checkpointer: Union[None, bool, langgraph.checkpoint.base.BaseCheckpointSaver] = None,
      log_tool_calls: bool = True,
  )
  ```
- Key ADS-specific public methods:
  - `invoke_agent(user_instructions: str = None, **kwargs)`
  - `ainvoke_agent(user_instructions: str = None, **kwargs)`
  - `invoke_messages(...)`, `ainvoke_messages(...)`
  - `get_artifacts(as_dataframe: bool = False)`  — returns the loaded data
  - `get_ai_message(markdown: bool = False)`
  - `get_tool_calls()`
  - `get_response()`
  - `get_internal_messages()`, `update_params(...)`, `show()`

### 3b. EDAToolsAgent

- Import: `from ai_data_science_team.ds_agents import EDAToolsAgent`
- Module: `ai_data_science_team.ds_agents.eda_tools_agent`
- Constructor: identical shape to `DataLoaderToolsAgent`:
  ```python
  EDAToolsAgent(
      model,
      create_react_agent_kwargs: Optional[Dict] = {},
      invoke_react_agent_kwargs: Optional[Dict] = {},
      checkpointer=None,
      log_tool_calls: bool = True,
  )
  ```
- Key ADS-specific public methods:
  - `invoke_agent(user_instructions: str = None, data_raw: pandas.DataFrame = None, **kwargs)` — note `data_raw` is required positionally-or-by-keyword.
  - `ainvoke_agent(user_instructions: str = None, data_raw: pandas.DataFrame = None, **kwargs)`
  - `get_artifacts(as_dataframe: bool = False)` — returns EDA artifacts (e.g. profile report path / dict)
  - `get_ai_message(markdown: bool = False)`
  - `get_tool_calls()`
  - `get_response()`
  - `get_internal_messages()`, `update_params(...)`, `show()`

### 3c. DataVisualizationAgent

- Import: `from ai_data_science_team.agents import DataVisualizationAgent`
- Module: `ai_data_science_team.agents.data_visualization_agent`
- Constructor (different shape — no `create_react_agent_kwargs`; this agent is a hand-built langgraph workflow with code-gen + explain steps):
  ```python
  DataVisualizationAgent(
      model,
      n_samples: int = 30,
      log: bool = False,
      log_path: Optional[str] = None,
      file_name: str = 'data_visualization.py',
      function_name: str = 'data_visualization',
      overwrite: bool = True,
      human_in_the_loop: bool = False,
      bypass_recommended_steps: bool = False,
      bypass_explain_code: bool = False,
      checkpointer=None,
  )
  ```
- Key ADS-specific public methods:
  - `invoke_agent(data_raw: pandas.DataFrame, user_instructions: str = None, max_retries: int = 3, retry_count: int = 0, **kwargs)` — `data_raw` is the first positional arg.
  - `ainvoke_agent(data_raw, user_instructions=None, max_retries=3, retry_count=0, **kwargs)`
  - `get_data_raw()` — returns the input dataframe (as dict / DataFrame)
  - `get_plotly_graph()` — returns the generated Plotly figure
  - `get_data_visualization_function(markdown=False)` — returns the generated Python code
  - `get_recommended_visualization_steps(markdown=False)`
  - `get_workflow_summary(markdown=False)`
  - `get_log_summary(markdown=False)`
  - `get_response()`
  - `update_params(...)`, `show()`

## 4. Bundled sample data

**The installed `ai_data_science_team` package does NOT ship any bundled sample data files.** A `find` over the package tree for `*.csv` / `*.parquet` / `*.json` / `*.xlsx` returned nothing, and there is no `data/` directory inside `site-packages/ai_data_science_team/`.

The package directory contains only python source dirs: `agents/`, `ds_agents/`, `ml_agents/`, `multiagents/`, `orchestration.py`, `parsers/`, `templates/`, `tools/`, `utils/`.

**Recommended dataset for the test phase** (small, public, well-known by ADS examples):

- **Churn (smallest, ~28 KB)** — `https://raw.githubusercontent.com/business-science/ai-data-science-team/master/data/churn_data.csv`
- Bike sales — `https://raw.githubusercontent.com/business-science/ai-data-science-team/master/data/bike_sales/bike_orderlines.csv` (larger)

Suggested local path the test agents should fetch-and-cache into: **`/Users/wang/LLM_memory/baseline/data/churn_data.csv`**.

(Alternatively, the test agents can use any small CSV they create on the fly — e.g. seaborn's `tips` or a tiny pandas-generated frame — since the goal is only to verify the agent pipeline runs.)

## 5. Install/import warnings, errors, workarounds

- No errors. Single `pip install` of all 10 requested packages succeeded.
- No deprecation warnings raised on import of the three target classes.
- `ai-data-science-team` version string is `0.0.0.9017` (development-style version) and the package does not expose `__version__` at top level — use `importlib.metadata.version("ai-data-science-team")` if needed.
- ADS pulls a fairly large transitive footprint (streamlit, xgboost, scikit-learn, scipy, altair). Install completed in well under 60 s on this machine but the venv is large — fine for baseline, something to keep in mind later.
- `uv` was not on PATH, so stdlib `venv` + `pip` were used. If we later want faster reinstalls we can `brew install uv`.

## 6. Next-phase contract (copy-paste for test agents)

The three baseline-test agents that come next MUST use exactly the following:

### Python / venv

- Interpreter: `/Users/wang/LLM_memory/.venv/bin/python` (also in `/Users/wang/LLM_memory/baseline/.venv_python`).
- Activate with `source /Users/wang/LLM_memory/.venv/bin/activate` or just call the python binary directly.

### LLM (OpenAI gpt-4o)

```python
import os
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0)  # requires OPENAI_API_KEY in env
```

### Data path

- Use `/Users/wang/LLM_memory/baseline/data/churn_data.csv`.
- If the file does not yet exist, download it once from
  `https://raw.githubusercontent.com/business-science/ai-data-science-team/master/data/churn_data.csv`
  into that path, then `pd.read_csv` it.

### DataLoaderToolsAgent test agent — exact API

```python
from ai_data_science_team.agents import DataLoaderToolsAgent

agent = DataLoaderToolsAgent(model=llm)
agent.invoke_agent(
    user_instructions="Load the CSV at /Users/wang/LLM_memory/baseline/data/churn_data.csv and return a small preview."
)
df = agent.get_artifacts(as_dataframe=True)   # the loaded dataframe
msg = agent.get_ai_message(markdown=False)    # the AI's final natural-language reply
tools = agent.get_tool_calls()                # list of tool invocations
```

### EDAToolsAgent test agent — exact API

```python
import pandas as pd
from ai_data_science_team.ds_agents import EDAToolsAgent  # NOTE: ds_agents, not agents

df = pd.read_csv("/Users/wang/LLM_memory/baseline/data/churn_data.csv")
agent = EDAToolsAgent(model=llm)
agent.invoke_agent(
    user_instructions="Run a quick EDA: shape, dtypes, missing-value summary, and notable correlations.",
    data_raw=df,
)
artifacts = agent.get_artifacts(as_dataframe=False)
msg = agent.get_ai_message(markdown=False)
tools = agent.get_tool_calls()
```

### DataVisualizationAgent test agent — exact API

```python
import pandas as pd
from ai_data_science_team.agents import DataVisualizationAgent

df = pd.read_csv("/Users/wang/LLM_memory/baseline/data/churn_data.csv")
agent = DataVisualizationAgent(
    model=llm,
    n_samples=30,
    log=False,
    bypass_recommended_steps=False,
    bypass_explain_code=False,
)
agent.invoke_agent(
    data_raw=df,                                              # first positional arg
    user_instructions="Plot churn rate by contract type as a bar chart.",
    max_retries=3,
)
fig        = agent.get_plotly_graph()                # plotly.graph_objs.Figure
code_str   = agent.get_data_visualization_function() # the generated Python code
steps_md   = agent.get_recommended_visualization_steps(markdown=True)
log_md     = agent.get_log_summary(markdown=True)
```

### What NOT to do

- Do not pass `data_raw` to `DataLoaderToolsAgent.invoke_agent` (it doesn't accept it; loading is the agent's job).
- Do not import `EDAToolsAgent` from `ai_data_science_team.agents` — it's not there. Use `ai_data_science_team.ds_agents`.
- Do not import any `DataLoaderToolsAgent` from `ds_agents` — it's only in `agents`.
- For `DataVisualizationAgent`, remember `data_raw` is the **first** positional argument to `invoke_agent`, before `user_instructions` — opposite of EDAToolsAgent's ordering.
