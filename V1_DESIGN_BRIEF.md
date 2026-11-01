# V1 Design Brief — `mads` memory-augmented data scientist

## 0. Scope recap (locked)

In scope for v1: cats **2** (`dataset_schema`), **3** (`metric_definition`), **5** (`analysis_episode`). Cats 1 (`user_preference`) and 4 (`data_quality_issue`) remain shipped but not surfaced through Memory Manager, not injected into wrapped agents, and not addressable via CLI `memory` subcommands. Their stores and tools stay importable from `mads.memory` so future v1.1 can wire them in by changing only Manager + wrapper + CLI registries.

No auto-extraction. No Validator. No second LLM pass on transcripts. Memory writes happen only when the wrapped agent's LLM emits a `record_*` / `register_*` / `add_*` tool call. Memory reads happen only when it emits a `recall_*` / `get_*` / `find_similar_*` / `list_*` / `search_*` tool call. The composer is templating; it never calls an LLM.

Vector store stays `sqlite-vec`. UI is CLI only. `python -m mads ask "..."` is the entry point.

---

## 1. File layout

New / modified files (all absolute paths in `/Users/wang/LLM_memory/`):

```
mads/
  __init__.py                       # bump __version__; re-export MemoryManager
  __main__.py                       # NEW — one-liner: from mads.cli import main; main()
  cli.py                            # NEW — argparse entrypoint (ask, memory ...)
  manager.py                        # NEW — MemoryManager facade
  router.py                         # NEW — Request Router (heuristic, LLM fallback)
  composer.py                       # NEW — Final Answer (pure templating)
  adapters/
    __init__.py                     # NEW
    ads.py                          # NEW — wraps ai_data_science_team
  agents/
    __init__.py                     # NEW
    wrapper.py                      # NEW — wrap_with_memory(base, manager, df=None)
    types.py                        # NEW — AgentResult, TurnArtifacts dataclasses
eval/
  run_with_memory.py                # NEW — 15 scenarios, A writes / B reads
  compare.py                        # NEW — baseline vs with-memory diff table
  agent_factories.py                # MODIFIED — import via mads.adapters.ads
tests/
  test_manager.py                   # NEW
  test_adapters_ads.py              # NEW
  test_wrapper.py                   # NEW
  test_router.py                    # NEW
  test_composer.py                  # NEW
  test_cli.py                       # NEW
  test_compare.py                   # NEW
  test_tenancy_e2e.py               # NEW — cross-layer tenancy contract
pyproject.toml                      # MODIFIED — add [project.scripts] + [eval] extra
mads/memory/__init__.py             # MODIFIED — re-export MemoryManager from mads.manager
```

Rationale for placement:

- `mads/manager.py` (top-level, not under `memory/`). The Manager is a *facade over* the memory package; placing it inside `mads/memory/` would invert the dependency direction. Top-level keeps `mads.memory` as a leaf package with no upward imports.
- `mads/adapters/ads.py` is the **only** file outside `eval/` that may `import ai_data_science_team`. CLAUDE.md flags the current `eval/agent_factories.py` import as the "exempt callsite"; v1 narrows that exemption to the adapter itself — eval will import the adapter, not ADS.
- `mads/agents/wrapper.py` plus a sibling `types.py` so the dataclasses (`AgentResult`, `TurnArtifacts`) don't induce a circular `wrapper ↔ composer` import.
- `__main__.py` is one line so `python -m mads ...` is symmetric with `python -m pytest`.

---

## 2. Memory Manager API — `mads/manager.py`

```python
# mads/manager.py
from __future__ import annotations

import os
import sqlite3
from typing import Optional

import pandas as pd
from langchain_core.tools import BaseTool

from mads.memory import (
    DatasetSchemaStore,
    MetricDefinitionStore,
    AnalysisEpisodeStore,
    MetricToolFactory,
    AnalysisEpisodeToolFactory,
    make_metric_factory,
    make_analysis_episode_factory,
    make_dataset_tools,
    Embedder,
)


class MemoryManager:
    """Facade owning one SQLite connection and the three in-scope stores.

    Constructor binds (user_id, project_id) into every factory closure; the
    LLM never sees these. ``db_path`` defaults to MADS_DB_PATH or
    ``./.mads/memory.sqlite`` (same default the stores already use).
    """

    def __init__(
        self,
        *,
        user_id: str,
        project_id: str,
        db_path: Optional[str] = None,
        embedder: Optional[Embedder] = None,
    ) -> None:
        if not isinstance(user_id, str) or not user_id:
            raise ValueError("user_id must be a non-empty string")
        if not isinstance(project_id, str) or not project_id:
            raise ValueError("project_id must be a non-empty string")
        self._user_id = user_id
        self._project_id = project_id

        if db_path is None:
            db_path = os.environ.get("MADS_DB_PATH") or "./.mads/memory.sqlite"
        self._db_path = db_path

        if db_path != ":memory:":
            parent = os.path.dirname(os.path.abspath(db_path))
            if parent:
                os.makedirs(parent, exist_ok=True)

        self._conn = sqlite3.connect(db_path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        # sqlite-vec must be loaded BEFORE AnalysisEpisodeStore touches it.
        self._conn.enable_load_extension(True)
        import sqlite_vec
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

        # Lazy attributes; constructed on first access.
        self._metric_store: Optional[MetricDefinitionStore] = None
        self._episode_store: Optional[AnalysisEpisodeStore] = None
        self._dataset_store: Optional[DatasetSchemaStore] = None
        self._metric_factory: Optional[MetricToolFactory] = None
        self._episode_factory: Optional[AnalysisEpisodeToolFactory] = None
        self._embedder = embedder
        # Late-bound dataframe (see Section 2.1).
        self._df: Optional[pd.DataFrame] = None
        self._dataset_name: Optional[str] = None
        self._table_name: Optional[str] = None

    # ------- tenancy accessors (read-only)
    @property
    def user_id(self) -> str: return self._user_id
    @property
    def project_id(self) -> str: return self._project_id
    @property
    def db_path(self) -> str: return self._db_path

    # ------- factory accessors
    @property
    def metric(self) -> MetricToolFactory: ...
    @property
    def episode(self) -> AnalysisEpisodeToolFactory: ...

    def dataset_tools(self) -> list[BaseTool]:
        """Return the 6 dataset @tool callables bound to the currently-bound df.
        Raises RuntimeError if no df has been bound via ``bind_dataframe``."""
        ...

    def bind_dataframe(
        self, df: pd.DataFrame, *, dataset_name: str, table_name: str
    ) -> None:
        """Bind the df for the *current analysis turn*. The Dataset factory
        re-emits tools that fingerprint THIS df. Idempotent within a turn —
        re-call to swap the df."""
        ...

    def all_tools(self, *, include_dataset: bool = True) -> list[BaseTool]:
        """Union of metric + episode + (optionally) dataset tools.
        ``include_dataset=False`` is the safe default when no df is bound yet
        (e.g. loader's first turn)."""
        ...

    def close(self) -> None:
        """Close the shared SQLite connection. Idempotent."""
        ...
```

### 2.1 Single-connection / df-binding decisions

**One shared connection** for all three stores: confirmed. `MetricDefinitionStore` and `AnalysisEpisodeStore` already accept `conn=` (see their constructors at line 173 and 177 respectively). `DatasetSchemaStore` does **not** accept `conn=` today (line 210: `__init__(self, db_path=...)` only). One required change: extend `DatasetSchemaStore.__init__` to accept an optional `conn=` kwarg following the exact pattern of the other two stores. This is the only modification to a shipped store; preserves the existing public surface.

**Why one connection.** Cross-store transactions are not needed in v1, but a single connection means a single `sqlite-vec` extension load, a single PRAGMA setup, and clean teardown semantics. Two connections to the same SQLite file under WAL would also work, but cost a second extension load (~ms) and complicate the `close()` contract.

**df-binding.** The Dataset factory pre-computes a fingerprint at construction (`dataset_tools.py` line 76). It cannot exist before the analyst loads data. v1 resolves this with **late binding via `bind_dataframe`**:

- After loader returns a df, the wrapper calls `manager.bind_dataframe(df, dataset_name=..., table_name=...)`.
- `manager.dataset_tools()` constructs (and caches per df-identity) the 6 dataset tools.
- A second call with a different df replaces the cached tools.
- `manager.all_tools(include_dataset=False)` is what the loader's *first* call gets, since at that moment no df exists yet. The loader's *post-load* turn would, in v1.1, get dataset tools — v1 does not loop the loader.

**Why not a `with_df` setter on a sub-namespace?** Adding `manager.dataset` as a factory property is misleading because there is no `DatasetToolFactory` class today — `make_dataset_tools` returns a list directly. Wrapping that in a stub factory just to mirror metric/episode adds a layer without value. The explicit `bind_dataframe` + `dataset_tools()` pair makes the late-binding obvious in the call site.

---

## 3. ADS Adapter — `mads/adapters/ads.py`

The first concrete implementation of what CLAUDE.md called "aspirational." Spec:

```python
# mads/adapters/ads.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd
from langchain_core.tools import BaseTool

# Only this module imports ai_data_science_team (plus eval/, transitionally).


@dataclass
class AgentResult:
    """Uniform return for every ADS sub-agent invocation."""
    response: Optional[str]               # natural-language reply
    tool_calls: list = field(default_factory=list)   # raw ADS tool-call records
    viz_code: Optional[str] = None        # only set for viz
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


def make_loader(
    *, extra_tools: Optional[list[BaseTool]] = None, model: Optional[Any] = None
) -> "LoaderHandle": ...
def make_eda(
    df: pd.DataFrame, *, extra_tools=None, model=None
) -> "EDAHandle": ...
def make_viz(
    df: pd.DataFrame, *, extra_tools=None, model=None
) -> "VizHandle": ...


class LoaderHandle:
    """Thin wrapper around DataLoaderToolsAgent.

    ``extra_tools`` is appended to the agent's tool list (memory @tools).
    Calling ``.invoke(prompt)`` returns an AgentResult.
    """
    def invoke(self, prompt: str) -> AgentResult: ...

class EDAHandle:
    def invoke(self, prompt: str) -> AgentResult: ...

class VizHandle:
    def invoke(self, prompt: str) -> AgentResult: ...
```

**`model=None` default**: build a fresh `ChatAnthropic(model="claude-sonnet-4-6", temperature=0)` + a `_UsageCallback`. The callback logic in `eval/agent_factories.py` (lines 46–88) moves into `mads/adapters/ads.py` verbatim — eval will then re-import it from the adapter rather than redefining it.

**Tool injection**: this is the load-bearing piece. The ADS agents construct their tool list internally. Investigate at code-time whether `DataLoaderToolsAgent(model=llm, tools=...)` accepts an `extra_tools` kwarg; if not, the handle constructs the agent, then mutates `agent._tools` (or whichever private attribute the agent uses) before `invoke_agent`. Document the mechanism in the adapter docstring and flag in section 12 as an open question — the patch approach depends on the ADS internals.

**`eval/agent_factories.py` refactor**: replace `from ai_data_science_team.agents import ...` with `from mads.adapters.ads import make_loader, make_eda, make_viz`. Keep `make_loader_factory()` / `make_eda_factory(df)` / `make_viz_factory(df)` signatures unchanged — they now build `LoaderHandle` etc. and adapt the return shape if needed for `invoke_for_agent`. Tests that import these functions are untouched.

---

## 4. Memory-Aware Wrapper — `mads/agents/wrapper.py`

```python
# mads/agents/wrapper.py
from __future__ import annotations

from typing import Optional
import pandas as pd

from mads.adapters.ads import LoaderHandle, EDAHandle, VizHandle, AgentResult
from mads.manager import MemoryManager
from mads.agents.types import TurnArtifacts


def wrap_loader(manager: MemoryManager) -> "WrappedLoader": ...
def wrap_eda(
    manager: MemoryManager, df: pd.DataFrame, *, dataset_name: str, table_name: str
) -> "WrappedEDA": ...
def wrap_viz(
    manager: MemoryManager, df: pd.DataFrame, *, dataset_name: str, table_name: str
) -> "WrappedViz": ...


class WrappedLoader:
    def invoke(self, prompt: str) -> TurnArtifacts: ...
class WrappedEDA:
    def invoke(self, prompt: str) -> TurnArtifacts: ...
class WrappedViz:
    def invoke(self, prompt: str) -> TurnArtifacts: ...
```

### Tool curation per agent

Recommendation: **all three categories' tools, every agent.** Reasons:

1. The agent's prompt template already drives tool selection; giving Viz access to `record_analysis_episode` is harmless and lets it log "I made this chart for question X" — exactly the kind of memory we want.
2. Curation per agent would force a 3×3 tool-injection matrix the implementer has to maintain; the LLM-side cost of seeing 5 extra unused tools is negligible (≈ a few hundred input tokens of tool-spec).
3. The exception: loader's *first* invocation has no df yet. `WrappedLoader` injects metric + episode tools only. After load (in v1.1, when we re-enter), it'd gain dataset tools. For v1, loader does one turn and is done.

| Agent | Metric tools (5) | Dataset tools (6) | Episode tools (6) | Total injected |
|---|---:|---:|---:|---:|
| WrappedLoader | yes | **no** (no df yet) | yes | 11 |
| WrappedEDA | yes | yes | yes | 17 |
| WrappedViz | yes | yes | yes | 17 |

### df-binding inside the wrapper

`wrap_eda(manager, df, dataset_name=, table_name=)`:
1. Calls `manager.bind_dataframe(df, dataset_name=dataset_name, table_name=table_name)`.
2. Builds `extra_tools = manager.all_tools(include_dataset=True)`.
3. Constructs `eda_handle = make_eda(df, extra_tools=extra_tools)`.
4. Stashes a reference to the handle and to the manager for later teardown.

### TurnArtifacts capture

The wrapper observes which memory tools were called during `eda_handle.invoke(prompt)` by inspecting `result.tool_calls`. Memory tool names are a known set (the union of names returned by `manager.metric.tools()`, `manager.dataset_tools()`, `manager.episode.tools()`); the wrapper filters tool-call records by name to populate `TurnArtifacts.memory_reads` and `TurnArtifacts.memory_writes`. This is observation only — no second SQL query, no re-fetch. The composer uses this list to cite memories in the final answer.

```python
# mads/agents/types.py
@dataclass
class TurnArtifacts:
    question: str
    agent_name: str                         # "loader" | "eda" | "viz"
    agent_result: AgentResult               # raw adapter output
    memory_reads: list[dict]                # tool_call records that READ memory
    memory_writes: list[dict]               # tool_call records that WROTE memory
```

---

## 5. Request Router — `mads/router.py`

**Recommendation: heuristic-first with no LLM fallback in v1.** The 15 baseline scenarios are unambiguous — load/preview/columns/dtypes → loader; missing-values/describe/correlation/distribution → eda; chart/plot/visualize/scatter/hist → viz. Adding an LLM call per turn doubles cost on the most common path. If the heuristic misses, the router returns `"eda"` as a generic default (EDA tools are a strict superset of loader for follow-on questions) and the wrapped agent still works because all three categories' tools are injected.

```python
# mads/router.py
from __future__ import annotations
import re
from typing import Literal

AgentName = Literal["loader", "eda", "viz"]


_LOADER_PATTERNS = [
    r"\bload\b", r"\bread\s+csv\b", r"\bimport\b.*\bdata\b",
    r"\bingest\b", r"\bpreview\b", r"\bfirst\s+\d+\s+rows?\b",
    r"\bdtype\b", r"\bcolumn\s+list\b", r"\blist\s+(?:the\s+)?columns\b",
]
_VIZ_PATTERNS = [
    r"\bplot\b", r"\bchart\b", r"\bvisuali[sz]e\b", r"\bscatter\b",
    r"\bhistogram\b", r"\bhist\b", r"\bbar\s+chart\b", r"\bbox\s*plot\b",
    r"\bheatmap\b", r"\bdraw\b", r"\bgraph\b",
]
# EDA is the residual.


def route(question: str) -> AgentName:
    """Classify ``question`` to one of loader/eda/viz. Heuristic only."""
    q = question.lower()
    for p in _VIZ_PATTERNS:               # check viz FIRST: "plot the missing values..."
        if re.search(p, q): return "viz"
    for p in _LOADER_PATTERNS:
        if re.search(p, q): return "loader"
    return "eda"
```

Why viz patterns are checked before loader: "plot the columns of x.csv" mentions both "plot" and "columns"; viz wins. Order is deliberate.

---

## 6. Final Answer composer — `mads/composer.py`

Pure templating. No LLM call.

```python
# mads/composer.py
from __future__ import annotations
from mads.agents.types import TurnArtifacts


def compose(turn: TurnArtifacts) -> str:
    """Format the wrapped agent's output + memory citations into a CLI-ready string."""
    parts: list[str] = []
    parts.append(f"# Answer ({turn.agent_name})")
    parts.append("")
    parts.append(turn.agent_result.response or "_(no response)_")

    if turn.agent_result.viz_code:
        parts.append("")
        parts.append("## Generated code")
        parts.append("```python")
        parts.append(turn.agent_result.viz_code)
        parts.append("```")

    if turn.memory_reads:
        parts.append("")
        parts.append(f"## Memory consulted ({len(turn.memory_reads)})")
        for rec in turn.memory_reads:
            parts.append(f"- {rec.get('name', '?')}({_short_args(rec)})")

    if turn.memory_writes:
        parts.append("")
        parts.append(f"## Memory written ({len(turn.memory_writes)})")
        for rec in turn.memory_writes:
            parts.append(f"- {rec.get('name', '?')}({_short_args(rec)})")

    parts.append("")
    parts.append(
        f"_tokens: {turn.agent_result.total_tokens} "
        f"(in={turn.agent_result.prompt_tokens}, out={turn.agent_result.completion_tokens})_"
    )
    return "\n".join(parts)


def _short_args(rec: dict, *, width: int = 80) -> str:
    """Render tool-call args as a one-liner, truncated."""
    ...
```

---

## 7. CLI command surface — `mads/cli.py` + `mads/__main__.py`

stdlib argparse. No Click/Typer.

```
mads ask "<question>" [--user USER] [--project PROJECT] [--data PATH] [--db PATH]

mads memory list metric                     [--project PROJECT]
mads memory list dataset                    [--user USER] [--project PROJECT]
mads memory list episode                    [--project PROJECT] [--limit N]

mads memory show metric NAME                [--project PROJECT]
mads memory show episode ID                 [--project PROJECT]
mads memory show dataset TABLE              [--user USER] [--project PROJECT]

mads memory history metric NAME             [--project PROJECT]

mads memory similar episode "QUERY"         [--project PROJECT] [--k N]
```

**Defaults**:
- `--user`: `$MADS_USER` env var; else `os.getenv("USER")`; else error with exit 2.
- `--project`: `$MADS_PROJECT`; else `"default"`.
- `--db`: `$MADS_DB_PATH`; else `./.mads/memory.sqlite`.
- `--limit`: 10 for `list episode`.
- `--k`: 5 for `similar episode`.

**Behavior**:
- `mads ask` route → wrap → invoke → compose → stdout. If `--data PATH` provided AND the router selects eda/viz, the CLI loads the CSV with `pandas.read_csv(PATH)` and passes the df into `wrap_eda` / `wrap_viz`. If router selects loader, `--data` is ignored (loader's job is to load).
- `mads memory list metric`: calls `manager.metric.tools()` → invokes `list_metric_definitions` → prints its string output.
- Every `memory` subcommand uses the same MemoryManager but with `include_dataset=False` (no df binding needed for browsing).

**Exit codes**:
- 0: success.
- 1: caught domain error (no metric found, invalid args from the tool layer).
- 2: argparse/usage error (missing `--user`, unknown subcommand). argparse's default.
- 3: unexpected exception (uncaught). Print traceback to stderr.

Also add a `[project.scripts]` entry in `pyproject.toml` so `mads ask ...` works in addition to `python -m mads ask ...`:

```toml
[project.scripts]
mads = "mads.cli:main"
```

---

## 8. Eval harness extension — `eval/run_with_memory.py` + `eval/compare.py`

### `eval/run_with_memory.py`

- Same 15 scenarios from `eval/scenarios.json`.
- Same `run_scenario_pair` harness from `eval/harness.py`.
- Factory difference: instead of `make_loader_factory()`, use a `make_memory_loader_factory(*, user_id, project_id, db_path)` that returns a zero-arg callable producing a *wrapped* loader handle. Same for eda and viz.
- Session A and Session B share the same `db_path` — that's the whole point. A's writes (via the LLM emitting `add_metric_definition` / `register_dataset` / `record_analysis_episode`) are visible to B's wrapped agent on the next invocation.
- Per-session JSON+MD into `eval/results/<agent>__with_memory/`.

Required prompt nudge per scenario A: the baseline A prompts state preferences in passing. Without auto-extraction, the wrapped loader's LLM must elect to call `register_dataset` or `add_metric_definition` to persist that. Options:

- **A1 (no prompt change)**: rely on the agent's system prompt + tool descriptions to nudge it to record. Risk: low write rate.
- **A2 (append a system-prompt nudge inside the wrapper)**: when wrapping, append "When the user states a domain fact, schema rule, metric definition, or analysis preference, record it via the appropriate memory tool before answering."

**Recommend A2**, gated by a `--nudge-memory/--no-nudge-memory` flag (default on).

Output:
```
eval/results/loader__with_memory/runs.jsonl
eval/results/loader__with_memory/SUMMARY.md
eval/results/eda__with_memory/...
eval/results/viz__with_memory/...
```

Per scenario, use a fresh DB file (or wipe between scenarios) so cross-scenario leakage doesn't muddle the signal.

### `eval/compare.py`

Reads `eval/results/<agent>/runs.jsonl` (baseline) AND `eval/results/<agent>__with_memory/runs.jsonl` (v1) → emits a diff table:

```
scenario_id              | baseline B_refs_A | v1 B_refs_A | delta
loader_01_snake_case     | no                | yes         | gap-closed
loader_02_no_phone...    | no                | yes         | gap-closed
loader_03_drop_pii       | no                | no          | unchanged
```

Output:
- `eval/results/COMPARE.md` — markdown table + summary (`gap-closed: N/15`).
- Exit 0 if at least 1 gap closed; exit 1 if zero gaps closed.

---

## 9. Dependency additions

Modify `pyproject.toml`:

```toml
[project]
dependencies = [
    "pydantic>=2.0",
    "langchain-core>=0.3",
    "sqlite-vec>=0.1.5",
    "sentence-transformers>=2.7",
    "pandas>=2.0",
    "numpy>=1.26",
    "langchain-anthropic>=0.3",        # NEW
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
]
eval = [
    "ai-data-science-team",            # NEW — only needed when running mads/eval
]

[project.scripts]
mads = "mads.cli:main"                  # NEW
```

The adapter imports ADS lazily inside function bodies so `import mads.memory` and `pytest` continue to work without ADS installed.

---

## 10. Test plan

| File | ~Count | Coverage |
|---|---:|---|
| `tests/test_manager.py` | 18 | shared conn; bind_dataframe swap; tool counts (11 vs 17); close idempotent; tenancy isolation; db_path precedence (arg > env > default); rejects empty user_id/project_id; embedder injection; cached factory |
| `tests/test_adapters_ads.py` | 8 | AgentResult shape; handles return AgentResult; extra_tools present; token usage; viz_code only for viz; ADS import error surfaces cleanly |
| `tests/test_wrapper.py` | 12 | wrap_loader injects 11 tools; wrap_eda injects 17; TurnArtifacts capture; nudge applied/skipped per flag; df=None to wrap_eda raises |
| `tests/test_router.py` | 12 | unambiguous classifications; viz wins over loader on "plot the columns"; case-insensitive; deterministic; pure (no I/O) |
| `tests/test_composer.py` | 10 | header; agent_name; viz_code block conditional; memory sections conditional; tokens line; truncation; idempotent; None response handling |
| `tests/test_cli.py` | 16 | each subcommand exits 0 on happy path with mocked wrapper; missing --user → exit 2; env var precedence; --db wins over env; --help exits 0 |
| `tests/test_compare.py` | 6 | diff table; labels (gap-closed/unchanged/regressed); exit 0/1 semantics; handles missing files; stable byte output |
| `tests/test_tenancy_e2e.py` | 8 | two managers (same db, different project) isolated for metrics + episodes; column rows shared by design; same-project cross-user can see metrics; args_schema excludes tenancy for all factories |

**Total new tests**: ~90; full suite ~268.

Use HashEmbedder for all episode-touching tests — sentence-transformers loads ~80MB on first call.

---

## 11. Build order

1. **Modify `DatasetSchemaStore.__init__`** to accept `conn=None` like the other two stores. `pytest tests/test_dataset_schema.py -q` → 24/24 still pass.
2. **`mads/manager.py` skeleton** (constructor + properties; no factories wired yet). `tests/test_manager.py` minimum 5 cases. Green.
3. **Wire metric + episode factories** into Manager; dataset path stubbed. Extend test to ~10. Green.
4. **Add `bind_dataframe` + `dataset_tools()`** to Manager. Finalize test_manager at ~18. Green.
5. **`mads/adapters/`** with `AgentResult` + 3 handle classes. Move `_UsageCallback` + `_make_llm_with_usage` from `eval/agent_factories.py`. `tests/test_adapters_ads.py` (mock ADS imports). **Pause**: run `cd eval && python run_loader.py` against 1 scenario to confirm refactor is transparent.
6. **Refactor `eval/agent_factories.py`** to import from `mads.adapters.ads`. Run loader eval on 1 scenario. **Pause**.
7. **`mads/agents/types.py`** + `mads/agents/wrapper.py`. `tests/test_wrapper.py` ~12. Green.
8. **`mads/router.py`**. `tests/test_router.py` ~12. Green.
9. **`mads/composer.py`**. `tests/test_composer.py` ~10. Green.
10. **`mads/cli.py` + `mads/__main__.py`**. `tests/test_cli.py` ~16 (mocked wrapper for `ask`). Green.
11. **Add `mads = "mads.cli:main"`** to pyproject.toml. `pip install -e ".[dev]"` confirms `mads --help` works.
12. **Cross-layer tenancy tests** `tests/test_tenancy_e2e.py`. Green.
13. **`eval/run_with_memory.py`**. Smoke-test with **1 scenario only** (DO NOT run all 15 — user will approve full run separately).
14. **`eval/compare.py`** + `tests/test_compare.py`. Green.
15. **Update CLAUDE.md** to remove "(aspirational; `mads/adapters/ads.py` not yet built)" caveat; add a Common-commands section for `mads ask` / `mads memory ...`.
16. **Full suite green**: `pytest -q` → 268-ish passes.

---

## 12. Open questions for the implementer

1. **ADS tool injection mechanism.** Does `DataLoaderToolsAgent.__init__` accept a `tools=` kwarg? Investigate at step 5.
2. **LLM provider key at CLI invocation.** `mads ask` requires `ANTHROPIC_API_KEY`. If unset, fail with a clear stderr message at the top of `cmd_ask`.
3. **ADS regex patch on fresh install.** Should the adapter's `make_viz()` validate the patch is applied, or trust the user?
4. **Embedder selection for production `mads ask`.** Default is `SentenceTransformerEmbedder` (~80MB on first run). Document in `--help` epilog.
5. **Cross-scenario DB isolation in `run_with_memory.py`.** Wipe between scenarios (clean signal) or accumulate (realistic). Default wipe + `--accumulate` flag.
