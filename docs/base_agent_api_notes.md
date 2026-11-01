# Base Agent API Notes

Reference for the six `ai_data_science_team` (ADS) sub-agents that `mads` either
already wraps or is considering wrapping. ADS version `0.0.0.9017` installed at
`.venv/lib/python3.13/site-packages/ai_data_science_team/`. All six extend
`ai_data_science_team.templates.BaseAgent` (a LangGraph `CompiledStateGraph`).
Shared pattern: build agent, call `invoke_agent(...)` (returns `None`), then
read `self.response` (a `dict`) via `get_*` accessors. All six also expose
`ainvoke_agent`, `invoke_messages`, `ainvoke_messages` variants.

## Wrapped in v1

`mads/adapters/ads.py` exposes `DataLoaderToolsAgent` (`LoaderHandle`,
`make_loader`), `EDAToolsAgent` (`EDAHandle`, `make_eda`), and
`DataVisualizationAgent` (`VizHandle`, `make_viz`) behind the uniform
`AgentResult` dataclass with token accounting. `PandasDataAnalyst`,
`SQLDataAnalyst`, `DataCleaningAgent` are NOT wrapped (candidates for v1.1+).

---

### PandasDataAnalyst

**Import**: `from ai_data_science_team import PandasDataAnalyst` (module `ai_data_science_team.multiagents.pandas_data_analyst`)
**Constructor**: `PandasDataAnalyst(model, data_wrangling_agent: DataWranglingAgent, data_visualization_agent: DataVisualizationAgent, checkpointer=None)`
**Invoke**: `invoke_agent(user_instructions: str, data_raw: pd.DataFrame | dict | list, max_retries=3, retry_count=0, **kwargs) -> None`
**Returns**: `None`. `self.response: dict` keys: `data_raw`, `data_wrangled`, `data_wrangler_function`, `data_visualization_function`, `plotly_graph`, `plotly_error`, `routing_preprocessor_decision`, `messages`.
**Accessors**:
- `.get_data_wrangled() -> pd.DataFrame | None`
- `.get_data_wrangler_function(markdown=False) -> str | None` — generated `data_wrangler(data_list)` source
- `.get_plotly_graph() -> plotly.Figure | None` — populated only on "chart" route
- `.get_data_visualization_function(markdown=False) -> str | None` — chart route only
- `.get_workflow_summary(markdown=False) -> str | None`
**Artifacts**: code + Plotly figure in-memory; no disk writes by default.
**Errors**: router failures swallowed (defaults to "table"); subagent errors surface in `response['plotly_error']` / `response['data_visualization_error']`. Bad `data_raw` type raises `ValueError`.

---

### SQLDataAnalyst

**Import**: `from ai_data_science_team import SQLDataAnalyst` (module `ai_data_science_team.multiagents.sql_data_analyst`)
**Constructor**: `SQLDataAnalyst(model, sql_database_agent: SQLDatabaseAgent, data_visualization_agent: DataVisualizationAgent, checkpointer=None)`
  - `SQLDatabaseAgent(model, connection, n_samples=1, log=False, log_path=None, file_name="sql_database.py", function_name="sql_database_pipeline", overwrite=True, human_in_the_loop=False, bypass_recommended_steps=False, bypass_explain_code=False, checkpointer=None, smart_schema_pruning=False, safe_mode=True)` — `connection` is a SQLAlchemy `Engine` or `Connection`.
**Invoke**: `invoke_agent(user_instructions: str, max_retries=3, retry_count=0, **kwargs) -> None` (no `data_raw` — DB is on the SQL sub-agent)
**Returns**: `None`. `self.response: dict` keys: `data_sql`, `sql_query_code`, `sql_database_function`, `data_visualization_function`, `plotly_graph`, `messages`.
**Accessors**:
- `.get_data_sql() -> pd.DataFrame | None`
- `.get_sql_query_code(markdown=False) -> str | None`
- `.get_sql_database_function(markdown=False) -> str | None`
- `.get_plotly_graph() -> plotly.Figure | None`
- `.get_data_visualization_function(markdown=False) -> str | None`
- `.get_workflow_summary(markdown=False) -> str | None`
**Artifacts**: SQL string, Python wrapper, Plotly figure; disk only if `log=True` on the SQL sub-agent.
**Errors**: SQL execution errors surface in `response['messages']`; retries up to `max_retries`. No exception for normal query failures.

---

### DataCleaningAgent

**Import**: `from ai_data_science_team import DataCleaningAgent` (module `ai_data_science_team.agents.data_cleaning_agent`)
**Constructor**: `DataCleaningAgent(model, n_samples=30, log=False, log_path=None, file_name="data_cleaner.py", function_name="data_cleaner", overwrite=True, human_in_the_loop=False, bypass_recommended_steps=False, bypass_explain_code=False, checkpointer=None)`
**Invoke**: `invoke_agent(data_raw: pd.DataFrame, user_instructions: str = None, max_retries=3, retry_count=0, **kwargs) -> None`
**Returns**: `None`. `self.response: dict` keys: `data_raw`, `data_cleaned`, `data_cleaner_function`, `data_cleaner_function_path`, `data_cleaner_function_name`, `recommended_steps`, `messages`.
**Accessors**:
- `.get_data_cleaned() -> pd.DataFrame | None`
- `.get_data_raw() -> pd.DataFrame | None`
- `.get_data_cleaner_function(markdown=False) -> str | None`
- `.get_recommended_cleaning_steps(markdown=False) -> str | None`
- `.get_workflow_summary(markdown=False) -> str | None`
- `.get_log_summary(markdown=False) -> str | None` — populated only when `log=True`
**Artifacts**: generated code in-memory; written to `log_path/file_name` only when `log=True`.
**Errors**: retries cleaning code on execution failure up to `max_retries`. Catastrophic LangGraph errors raise.

---

### DataLoaderToolsAgent (wrapped in v1)

**Import**: `from ai_data_science_team.agents import DataLoaderToolsAgent`
**Constructor**: `DataLoaderToolsAgent(model, create_react_agent_kwargs: dict = {}, invoke_react_agent_kwargs: dict = {}, checkpointer=None, log_tool_calls=True)`
**Invoke**: `invoke_agent(user_instructions: str = None, **kwargs) -> None`
**Returns**: `None`. `self.response: dict` keys: `messages`, `internal_messages`, `tool_calls`, `data_loader_artifacts`.
**Accessors**:
- `.get_ai_message(markdown=False) -> str | None` — last assistant message
- `.get_tool_calls() -> list`
- `.get_artifacts(as_dataframe=False) -> dict | pd.DataFrame | None` — loaded data (auto-unwraps single-artifact case)
- `.get_internal_messages(markdown=False) -> list | str`
**Artifacts**: loaded DataFrame(s) via `get_artifacts(as_dataframe=True)`; no disk writes by default.
**Errors**: tool errors captured by the react loop and reported in the AI message; no exception for normal load failures.
**mads wrapper**: `LoaderHandle` at `mads/adapters/ads.py:197`, factory `make_loader` at `mads/adapters/ads.py:336`. Memory tools appended to `ai_data_science_team.agents.data_loader_tools_agent.tools` during construction.

---

### EDAToolsAgent (wrapped in v1)

**Import**: `from ai_data_science_team.ds_agents import EDAToolsAgent`
**Constructor**: `EDAToolsAgent(model, create_react_agent_kwargs: dict = {}, invoke_react_agent_kwargs: dict = {}, checkpointer=None, log_tool_calls=True)`
**Invoke**: `invoke_agent(user_instructions: str = None, data_raw: pd.DataFrame = None, **kwargs) -> None`
**Returns**: `None`. `self.response: dict` keys: `messages`, `internal_messages`, `tool_calls`, `eda_artifacts`.
**Accessors**:
- `.get_ai_message(markdown=False) -> str | None`
- `.get_tool_calls() -> list`
- `.get_artifacts(as_dataframe=False) -> dict | pd.DataFrame | None` — describe / missing / correlation outputs
- `.get_internal_messages(markdown=False) -> list | str`
**Artifacts**: stats summaries + (optionally) saved HTML reports if an EDA tool writes them; otherwise in-memory only.
**Errors**: same react-loop convention as the loader.
**mads wrapper**: `EDAHandle` at `mads/adapters/ads.py:237`, factory `make_eda` at `mads/adapters/ads.py:345`. Memory tools appended to `ai_data_science_team.ds_agents.eda_tools_agent.EDA_TOOLS`.

---

### DataVisualizationAgent (wrapped in v1)

**Import**: `from ai_data_science_team.agents import DataVisualizationAgent`
**Constructor**: `DataVisualizationAgent(model, n_samples=30, log=False, log_path=None, file_name="data_visualization.py", function_name="data_visualization", overwrite=True, human_in_the_loop=False, bypass_recommended_steps=False, bypass_explain_code=False, checkpointer=None)`
**Invoke**: `invoke_agent(data_raw: pd.DataFrame, user_instructions: str = None, max_retries=3, retry_count=0, **kwargs) -> None`
**Returns**: `None`. `self.response: dict` keys: `data_raw`, `data_visualization_function`, `plotly_graph`, `recommended_steps`, `messages`; `data_visualization_function_path`/`_name` only when `log=True`.
**Accessors**:
- `.get_plotly_graph() -> plotly.Figure | None`
- `.get_data_visualization_function(markdown=False) -> str | None`
- `.get_data_raw() -> pd.DataFrame | None`
- `.get_recommended_visualization_steps(markdown=False) -> str | None`
- `.get_workflow_summary(markdown=False) -> str | None`
- `.get_log_summary(markdown=False) -> str | None`
- `.get_response() -> dict | None` — raw state dict
**Artifacts**: in-memory Plotly figure + generated code; disk only when `log=True`.
**Errors**: code-gen pipeline; retries on viz execution failure up to `max_retries`. Local regex patch in `_extract_missing_columns` — see `eval/install/apply_viz_patch.py`.
**mads wrapper**: `VizHandle` at `mads/adapters/ads.py:274`, factory `make_viz` at `mads/adapters/ads.py:355`. `extra_tools` is inert (code-gen pipeline, not a react tool-caller) — documented v1 limitation.

---

## AI Pipeline Studio

Not found. No Streamlit app (`streamlit_app.py`, `app/`, `apps/`, `*studio*`) ships
in the installed package. The only `streamlit` references in source are docstring
hints telling users to use `st.code()` when embedding generated code in their
own Streamlit apps. `ai_data_science_team/orchestration.py` is a stub
(`# TODO - add orchestration`). `streamlit` is in runtime deps for downstream user apps.

---

## Notes

- ADS version: `0.0.0.9017` (`pip show ai-data-science-team`).
- ADS source: `.venv/lib/python3.13/site-packages/ai_data_science_team/`.
- Smoke test confirmed for PandasDataAnalyst: 5x2 synthetic df + "What is the total revenue per product?". `invoke_agent` returned `None`; `response` was a dict (11 keys); `get_data_wrangled()` returned a 3x2 DataFrame; `get_data_wrangler_function()` returned 2082-char source; `get_plotly_graph()` and `get_data_visualization_function()` were `None` (router chose "table"); `get_workflow_summary()` returned a short header.
- Viz regex patch is applied locally; reapply via `python eval/install/apply_viz_patch.py` after any `pip install --force-reinstall`.
- For wrapped agents, the mads-side handle abstraction is in `mads/adapters/ads.py`; uniform return type `AgentResult` (`response`, `tool_calls`, `viz_code`, token counts).

## Open questions

- `PandasDataAnalyst.get_workflow_summary()` reports "0 agents" because the sub-agent messages aren't tagged with the expected `role`. Not a blocker; if mads surfaces this summary, synthesize from `response['messages']`.
- `DataLoaderToolsAgent.get_artifacts(as_dataframe=True)` may return `dict[str, DataFrame]` or a single `DataFrame` depending on how many tools fired; any wrapper needs to normalize.
