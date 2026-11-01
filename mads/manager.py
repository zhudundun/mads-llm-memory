"""MemoryManager: facade over the four in-scope memory stores.

Owns one shared :mod:`sqlite3` connection and binds ``(user_id, project_id)``
into every factory closure so the LLM never sees tenancy fields.

In-scope categories for v1 (matching V1_DESIGN_BRIEF):

* :class:`MetricDefinitionStore`         — ``metric`` factory (5 tools)
* :class:`PreferenceStore`               — ``preference`` factory (4 tools)
* :class:`DatasetSchemaStore`            — late-bound via ``bind_dataframe``
                                            (6 tools, only available after a df
                                            is bound)
* :class:`AnalysisEpisodeStore`          — ``episode`` factory (6 tools)
"""

from __future__ import annotations

import os
import sqlite3
import threading
from typing import Optional

import pandas as pd
from langchain_core.tools import BaseTool

from mads.dataset.fingerprint import semantic_fingerprint
from mads.memory import (
    AnalysisEpisodeStore,
    AnalysisEpisodeToolFactory,
    DataQualityStore,
    DataQualityToolFactory,
    DatasetSchemaStore,
    Embedder,
    HashEmbedder,
    MetricDefinitionStore,
    MetricToolFactory,
    PreferenceStore,
    make_analysis_episode_factory,
    make_data_quality_factory,
    make_dataset_tools,
    make_metric_factory,
    make_preference_tools,
)
from mads.memory.retrievers import (
    DatasetRetriever,
    EpisodeRetriever,
    HybridRetriever,
    MetricRetriever,
    PreferenceRetriever,
    RetrievedMemory,
)
from mads.tools.pandas_exec import make_pandas_exec_factory


# v1.4: prompt-budget cap for the bound-df column listing in the
# APPLICABLE annotation. 10 names is enough to disambiguate a Telco-shaped
# (~20 col) dataset without blowing the prompt on a wide warehouse table.
_AVAIL_COLS_DISPLAY_LIMIT = 10


def _format_applicable_false(
    *, missing: list[str], available: list[str]
) -> str:
    """Render the one-line ``APPLICABLE: false`` annotation for the prompt.

    Shape::

        APPLICABLE: false (requires `<missing_a>`, `<missing_b>`; bound df has [a, b, c ... +N more])

    Truncates the available column list to keep the prompt budget bounded.
    """
    missing_part = ", ".join(f"`{c}`" for c in missing) if missing else "?"
    if len(available) <= _AVAIL_COLS_DISPLAY_LIMIT:
        cols_part = ", ".join(available)
    else:
        head = ", ".join(available[:_AVAIL_COLS_DISPLAY_LIMIT])
        extra = len(available) - _AVAIL_COLS_DISPLAY_LIMIT
        cols_part = f"{head} ... +{extra} more"
    return f"APPLICABLE: false (requires {missing_part}; bound df has [{cols_part}])"


class MemoryManager:
    """Facade owning one SQLite connection and the three in-scope stores.

    Constructor binds (user_id, project_id) into every factory closure; the
    LLM never sees these. ``db_path`` defaults to ``MADS_DB_PATH`` or
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

        # db_path precedence: explicit arg > env var > default.
        if db_path is None:
            db_path = os.environ.get("MADS_DB_PATH") or "./.mads/memory.sqlite"
        self._db_path = db_path

        if db_path != ":memory:":
            parent = os.path.dirname(os.path.abspath(db_path))
            if parent:
                os.makedirs(parent, exist_ok=True)

        # ``check_same_thread=False`` is required because the eval harness
        # runs each session inside a daemonic worker thread (see
        # ``eval/harness.py::_run_with_timeout``) and langgraph's tool node
        # internally also dispatches tool execution through a ThreadPool —
        # so multiple tool calls can land on the same connection from
        # different threads. ``self._lock`` (re-entrant) is injected into
        # every store so they share one serialiser; without that, two
        # stores could BEGIN IMMEDIATE on the same conn at once and SQLite
        # raises ``cannot start a transaction within a transaction``.
        self._conn = sqlite3.connect(
            db_path, isolation_level=None, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        if db_path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL;")

        self._lock = threading.RLock()

        # sqlite-vec must be loaded BEFORE AnalysisEpisodeStore touches the db.
        self._conn.enable_load_extension(True)
        import sqlite_vec

        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

        # Lazy attributes; constructed on first access.
        self._metric_store: Optional[MetricDefinitionStore] = None
        self._episode_store: Optional[AnalysisEpisodeStore] = None
        self._dataset_store: Optional[DatasetSchemaStore] = None
        self._preference_store: Optional[PreferenceStore] = None
        self._data_quality_store: Optional[DataQualityStore] = None
        self._metric_factory: Optional[MetricToolFactory] = None
        self._episode_factory: Optional[AnalysisEpisodeToolFactory] = None
        self._preference_tools_cache: Optional[list[BaseTool]] = None
        self._embedder: Optional[Embedder] = embedder

        # Late-bound dataframe; populated by ``bind_dataframe``.
        self._df: Optional[pd.DataFrame] = None
        self._dataset_name: Optional[str] = None
        self._table_name: Optional[str] = None
        # Stash the dataset's semantic fingerprint at bind time so
        # data-quality tools (dataset-scoped, IDENTITY_REQUIRED) can be
        # built with the correct closure-bound fingerprint.
        self._dataset_fingerprint: Optional[str] = None
        # Cache the dataset @tool list keyed by python ``id(df)`` so a re-bind
        # with the same df reuses the same closures.
        self._cached_dataset_tools: Optional[list[BaseTool]] = None
        self._cached_dataset_tools_df_id: Optional[int] = None
        # Same for the data-quality tools (they depend on the fingerprint).
        self._cached_data_quality_tools: Optional[list[BaseTool]] = None
        self._cached_data_quality_tools_fp: Optional[str] = None

        # Lazy retriever — instantiated on first ``retrieve`` call.
        self._hybrid_retriever: Optional[HybridRetriever] = None

        self._closed = False

    # ------------------------------------------------------------------ accessors
    @property
    def user_id(self) -> str:
        return self._user_id

    @property
    def project_id(self) -> str:
        return self._project_id

    @property
    def db_path(self) -> str:
        return self._db_path

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    # ------------------------------------------------------------------ stores
    @property
    def _metric_store_(self) -> MetricDefinitionStore:
        # Lazy + thread-safe: two concurrent first-accesses (e.g. langgraph
        # dispatching two tool calls before the store has been touched)
        # must not both run init_schema on the same shared connection.
        if self._metric_store is None:
            with self._lock:
                if self._metric_store is None:
                    # B3: share the SAME Embedder instance with cat 5 so
                    # the sentence-transformers model is loaded once per
                    # manager. The episode store property lazily defaults
                    # to HashEmbedder when nothing was passed; mirror
                    # that contract here so the two stores agree.
                    embedder = self._embedder or HashEmbedder()
                    # Cache so `_episode_store_` picks up the same one.
                    self._embedder = embedder
                    self._metric_store = MetricDefinitionStore(
                        conn=self._conn,
                        lock=self._lock,
                        embedder=embedder,
                    )
        return self._metric_store

    @property
    def _episode_store_(self) -> AnalysisEpisodeStore:
        if self._episode_store is None:
            with self._lock:
                if self._episode_store is None:
                    # Default to HashEmbedder for cheap deterministic tests and
                    # interactive CLI runs; production callers can pass a
                    # SentenceTransformerEmbedder explicitly.
                    embedder = self._embedder or HashEmbedder()
                    self._episode_store = AnalysisEpisodeStore(
                        conn=self._conn, embedder=embedder, lock=self._lock
                    )
        return self._episode_store

    @property
    def _dataset_store_(self) -> DatasetSchemaStore:
        if self._dataset_store is None:
            with self._lock:
                if self._dataset_store is None:
                    self._dataset_store = DatasetSchemaStore(
                        conn=self._conn, lock=self._lock
                    )
        return self._dataset_store

    @property
    def _data_quality_store_(self) -> DataQualityStore:
        """Lazy DataQualityStore sharing the manager's connection + lock."""
        if self._data_quality_store is None:
            with self._lock:
                if self._data_quality_store is None:
                    self._data_quality_store = DataQualityStore(
                        conn=self._conn, lock=self._lock
                    )
        return self._data_quality_store

    @property
    def _preference_store_(self) -> PreferenceStore:
        # Double-checked locking: matches the other lazy store getters so that
        # two concurrent first-accesses (e.g. via langgraph's ToolNode pool)
        # don't both run init_schema on the shared connection.
        if self._preference_store is None:
            with self._lock:
                if self._preference_store is None:
                    self._preference_store = PreferenceStore(
                        conn=self._conn, lock=self._lock
                    )
        return self._preference_store

    # ------------------------------------------------------------------ factories
    @property
    def metric(self) -> MetricToolFactory:
        """Return a cached :class:`MetricToolFactory` bound to (user, project).

        B3: thread the SAME MetricRetriever used by HybridRetriever into
        the tool factory so the reactive ``search_metric_definitions``
        tool and the proactive injection path share one hybrid
        (string + semantic max-of) scoring impl.
        """
        if self._metric_factory is None:
            # Touch the hybrid retriever first so its MetricRetriever
            # closes over the same embedder/store. We pass that
            # sub-retriever into the tool factory.
            metric_retriever = self._hybrid_retriever_._metric  # type: ignore[attr-defined]
            self._metric_factory = make_metric_factory(
                self._metric_store_,
                self._user_id,
                self._project_id,
                retriever=metric_retriever,
            )
        return self._metric_factory

    @property
    def episode(self) -> AnalysisEpisodeToolFactory:
        """Return a cached :class:`AnalysisEpisodeToolFactory` bound to
        (user, project)."""
        if self._episode_factory is None:
            self._episode_factory = make_analysis_episode_factory(
                self._episode_store_, self._user_id, self._project_id
            )
        return self._episode_factory

    class _PreferenceNamespace:
        """Tiny namespace exposing ``.tools()`` so callers can write
        ``manager.preference.tools()`` symmetrically with metric/episode.

        :func:`make_preference_tools` is a plain function (not a factory class)
        — this wrapper caches its 4 tools on first call to match the
        metric/episode factories' "build once per manager" behaviour.
        """

        def __init__(self, manager: "MemoryManager") -> None:
            self._manager = manager

        def tools(self) -> list[BaseTool]:
            mgr = self._manager
            if mgr._preference_tools_cache is None:
                # Thread a LIVE accessor for the dataset fingerprint so
                # ``dataset_scoped=True`` reflects any mid-conversation
                # ``bind_dataframe`` call (snapshotting at factory-build
                # time would lock in whatever fingerprint was current then,
                # potentially None for loader's first turn).
                def _live_fp() -> Optional[str]:
                    return mgr._dataset_fingerprint

                mgr._preference_tools_cache = make_preference_tools(
                    mgr._preference_store_,
                    mgr._user_id,
                    project_id=mgr._project_id,
                    get_dataset_fingerprint=_live_fp,
                )
            return list(mgr._preference_tools_cache)

    @property
    def preference(self) -> "MemoryManager._PreferenceNamespace":
        """Return a namespace whose ``.tools()`` yields the 4 preference tools
        bound to ``(store, user_id, project_id)`` via closure.

        Per the axis model (v1.2): ``user_id`` is IDENTITY_REQUIRED on
        preferences; ``project_id`` is IDENTITY_OPTIONAL — the LLM
        controls scope per call by passing ``project_scoped: bool`` to
        ``remember_preference``/``forget_preference``."""
        return MemoryManager._PreferenceNamespace(self)

    # ------------------------------------------------------------------ dataset
    def bind_dataframe(
        self,
        df: pd.DataFrame,
        *,
        dataset_name: str,
        table_name: str,
    ) -> None:
        """Bind the df for the *current analysis turn*. The dataset factory
        re-emits tools that fingerprint THIS df. Idempotent within a turn —
        re-call to swap the df.
        """
        if df is None:
            raise ValueError("bind_dataframe requires a non-None DataFrame")
        if not isinstance(dataset_name, str) or not dataset_name:
            raise ValueError("dataset_name must be a non-empty string")
        if not isinstance(table_name, str) or not table_name:
            raise ValueError("table_name must be a non-empty string")
        self._df = df
        self._dataset_name = dataset_name
        self._table_name = table_name
        # Compute and stash the semantic fingerprint so data-quality tools
        # (dataset-scoped, IDENTITY_REQUIRED) can be built with the
        # correct identity. We use the SAME semantic fingerprint helper the
        # profiler uses, so a single df always yields a single identity
        # across calls.
        self._dataset_fingerprint = semantic_fingerprint(df)
        # Invalidate cached tool lists.
        self._cached_dataset_tools = None
        self._cached_dataset_tools_df_id = None
        self._cached_data_quality_tools = None
        self._cached_data_quality_tools_fp = None

    @property
    def dataset_fingerprint(self) -> Optional[str]:
        """Semantic fingerprint of the currently-bound df, or ``None`` if no
        df has been bound. Populated by :meth:`bind_dataframe`."""
        return self._dataset_fingerprint

    def data_quality_tools(self) -> list[BaseTool]:
        """Return the 6 data-quality ``@tool`` callables bound to the
        currently-bound df's fingerprint.

        Raises ``RuntimeError`` if no df has been bound — data-quality
        memory is dataset-scoped under the axis model, so its tools
        can't be built without a dataset_fingerprint. The recommended
        pattern is ``all_tools(include_dataset=False)`` for loader's
        pre-bind turn (excludes dataset + dq tools), then
        ``all_tools(include_dataset=True)`` after :meth:`bind_dataframe`.
        """
        if self._dataset_fingerprint is None:
            raise RuntimeError(
                "data_quality_tools() requires a DataFrame to be bound first via "
                "MemoryManager.bind_dataframe(df, dataset_name=..., table_name=...)"
            )
        fp = self._dataset_fingerprint
        if (
            self._cached_data_quality_tools is not None
            and self._cached_data_quality_tools_fp == fp
        ):
            return list(self._cached_data_quality_tools)
        factory = make_data_quality_factory(
            self._data_quality_store_,
            self._user_id,
            self._project_id,
            fp,
        )
        tools = factory.tools()
        self._cached_data_quality_tools = tools
        self._cached_data_quality_tools_fp = fp
        return list(tools)

    def pandas_exec_tools(self) -> list[BaseTool]:
        """Return the 1 ``run_pandas_code`` tool bound (by closure) to the
        currently-bound df.

        The closure reads ``self._df`` LIVE on every tool invocation, so a
        ``bind_dataframe`` swap mid-conversation is honored. Raises
        ``RuntimeError`` if no df has been bound — the tool is meaningless
        without a target df (its first behavior would be to return an
        ERROR string on every call).

        Per axis-model contract: the tool's args expose only ``code: str``;
        neither the df nor any tenancy is reachable from the LLM-visible
        surface.
        """
        if self._df is None:
            raise RuntimeError(
                "pandas_exec_tools() requires a DataFrame to be bound first via "
                "MemoryManager.bind_dataframe(df, dataset_name=..., table_name=...)"
            )
        factory = make_pandas_exec_factory(lambda: self._df)
        return list(factory.tools())

    def dataset_tools(self) -> list[BaseTool]:
        """Return the 6 dataset ``@tool`` callables bound to the currently-bound
        df. Raises ``RuntimeError`` if no df has been bound via
        :meth:`bind_dataframe`.
        """
        if self._df is None:
            raise RuntimeError(
                "dataset_tools() requires a DataFrame to be bound first via "
                "MemoryManager.bind_dataframe(df, dataset_name=..., table_name=...)"
            )
        df_identity = id(self._df)
        if (
            self._cached_dataset_tools is not None
            and self._cached_dataset_tools_df_id == df_identity
        ):
            return list(self._cached_dataset_tools)
        tools = make_dataset_tools(
            self._dataset_store_,
            self._df,
            dataset_name=self._dataset_name,  # type: ignore[arg-type]
            table_name=self._table_name,  # type: ignore[arg-type]
            user_id=self._user_id,
            project_id=self._project_id,
        )
        self._cached_dataset_tools = tools
        self._cached_dataset_tools_df_id = df_identity
        return list(tools)

    # ------------------------------------------------------------------ unions
    def all_tools(self, *, include_dataset: bool = True) -> list[BaseTool]:
        """Union of metric + preference + episode + (optionally) dataset +
        pandas-exec tools.

        Counts:

        * ``include_dataset=False`` → 5 metric + 4 preference + 6 episode = 15.
          This is the safe default when no df is bound yet (loader's first
          turn). ``run_pandas_code`` is NOT included — it needs a bound df.
        * ``include_dataset=True`` → 15 + 6 dataset + 1 pandas-exec = 22.

        Preference tools have no df dependency, so they're included
        unconditionally (just like metric tools). The ``run_pandas_code``
        tool, like dataset tools, requires a bound df and therefore only
        appears when ``include_dataset=True``.
        """
        tools: list[BaseTool] = []
        tools.extend(self.metric.tools())
        tools.extend(self.preference.tools())
        tools.extend(self.episode.tools())
        if include_dataset:
            tools.extend(self.dataset_tools())
            tools.extend(self.pandas_exec_tools())
        return tools

    # ------------------------------------------------------------------ retrieval
    @property
    def _hybrid_retriever_(self) -> HybridRetriever:
        if self._hybrid_retriever is None:
            # B3: pass the manager's shared embedder into MetricRetriever
            # (the store may also carry it, but make it explicit so the
            # retriever's behavior is independent of construction order
            # — e.g. tests that build the manager without touching the
            # metric store first).
            self._hybrid_retriever = HybridRetriever(
                metric=MetricRetriever(
                    self._metric_store_, embedder=self._embedder
                ),
                dataset=DatasetRetriever(self._dataset_store_),
                episode=EpisodeRetriever(self._episode_store_),
                preference=PreferenceRetriever(self._preference_store_),
            )
        return self._hybrid_retriever

    def _available_columns_for_session(self) -> Optional[set[str]]:
        """Return the set of column names visible for applicability checks.

        Resolution order (v1.4 schema coupling):

        1. If a df is currently bound, return ``set(df.columns)``. This is
           the most accurate signal — the LLM is about to operate on this
           exact df, so its columns are the ones that matter.
        2. Else, if a ``dataset_fingerprint`` is stashed, look up the
           ``column_memory`` rows for that fingerprint via the dataset
           store. Returns the set of registered column names. This is
           the fallback for pre-bind reads (e.g. loader's first turn
           after a session A registration).
        3. Else, return ``None`` — caller treats this as "can't check;
           assume applicable". Returning an empty set would falsely
           mark every column-dependent metric inapplicable.
        """
        if self._df is not None:
            return set(self._df.columns)
        if self._dataset_fingerprint is not None and self._table_name is not None:
            cols = self._dataset_store_.list_columns(
                fingerprint=self._dataset_fingerprint,
                table_name=self._table_name,
            )
            if cols:
                return {c["column_name"] for c in cols}
        return None

    def retrieve(
        self,
        *,
        query: str,
        user_id: str,
        project_id: str,
        dataset_fingerprint: Optional[str] = None,
        k: int = 5,
    ) -> list[RetrievedMemory]:
        """Hybrid retrieval across metric / dataset / episode / preference categories.

        Lazily instantiates a :class:`HybridRetriever` and reuses it across
        calls. ``dataset_fingerprint`` (when provided) is threaded into the
        episode retriever as a ``datasets_used_contains`` post-filter so
        only episodes that referenced the same logical dataset surface.

        The dataset sub-retriever requires a df+table_name to fire; we use
        the currently-bound df (if any) so the lookup is grounded in the
        same artifact the EDA agent will see. If no df is bound, the
        dataset retriever silently returns no rows.

        Preferences are always returned for the user — they are user-state,
        not query-context, and are NOT capped by ``k`` (which only bounds
        the relevance-ranked metric/dataset/episode rows). So the returned
        list may contain up to ``k`` relevance rows PLUS every active
        preference row for the user.

        ``dataset_fingerprint`` (v1.3) is threaded into the preference
        sub-retriever so dataset-scoped prefs surface when the session
        has a df bound; when None, only project/global prefs surface.
        Callers typically pass the dataset's semantic fingerprint here
        (the same one the episode side uses for ``datasets_used``).
        """
        # If the caller did not supply a dataset_fingerprint but the
        # manager has one bound (typical for the EDA wrapper), prefer
        # that so preference scoping stays consistent across retrieve
        # calls within the same turn.
        effective_fp = dataset_fingerprint
        if effective_fp is None and self._dataset_fingerprint is not None:
            effective_fp = self._dataset_fingerprint
        return self._hybrid_retriever_.retrieve(
            query,
            user_id=user_id,
            project_id=project_id,
            k=k,
            datasets_used_contains=effective_fp,
            df=self._df,
            table_name=self._table_name,
            available_columns=self._available_columns_for_session(),
        )

    @staticmethod
    def format_context(memories: list[RetrievedMemory]) -> str:
        """Render retrieved memories as citation-tagged blocks for prompt injection.

        Block shape::

            [memory:CATEGORY:ID] (score=S.SS)
            <key>: <value>
            ...
            <blank line>

        where ``CATEGORY`` ∈ ``{metric, dataset, episode, preference}`` and
        ``ID`` is the category's natural identifier (``normalized_name`` for
        metric, ``semantic_fingerprint`` for dataset, ``id`` for episode,
        ``preference_key`` for preference).
        """
        if not memories:
            return "(no relevant memories found)"
        blocks: list[str] = []
        for rm in memories:
            cat = rm.category
            mem = rm.memory or {}
            # v1.4: pulled out so it survives whatever per-category dict
            # mutation the caller did and so we can append a uniform
            # APPLICABLE annotation block at the bottom for metrics.
            applicable_annotation: Optional[str] = None
            if cat == "metric":
                mid = mem.get("normalized_name") or mem.get("metric_name") or "?"
                rows = [
                    ("name", mem.get("metric_name")),
                    ("business", mem.get("business_definition")),
                    ("sql", mem.get("sql_definition")),
                ]
                # Only annotate when the retriever decorated the row AND
                # found the metric inapplicable. ``applicable=True`` (the
                # default) needs no extra line — silence is the loud signal.
                if mem.get("applicable") is False:
                    missing = mem.get("missing_columns") or []
                    # ``required_columns`` came back through model_dump on
                    # the metric model, so use it for the display message
                    # when available; fall back to missing list otherwise.
                    req = mem.get("required_columns") or missing
                    available = mem.get("_available_columns_for_display") or []
                    applicable_annotation = _format_applicable_false(
                        missing=missing if missing else list(req),
                        available=available,
                    )
            elif cat == "dataset":
                mid = (
                    mem.get("semantic_fingerprint")
                    or mem.get("dataset_fingerprint")
                    or "?"
                )
                mid = (mid[:12] + "…") if isinstance(mid, str) and len(mid) > 12 else mid
                # Dataset rows from lookup_dataset don't carry row counts —
                # they come from the active dataset_memory table; surface
                # the high-signal fields we *do* have.
                rows = [
                    ("name", mem.get("dataset_name") or mem.get("table_name")),
                    ("grain", mem.get("grain")),
                    ("known_issues", mem.get("known_issues")),
                ]
            elif cat == "episode":
                mid = mem.get("id") or "?"
                rows = [
                    ("question", mem.get("question")),
                    ("conclusion", mem.get("conclusion")),
                    ("assumptions", mem.get("assumptions")),
                ]
            elif cat == "preference":
                # ID is the preference key — short, stable, and the
                # natural human-readable handle. The "value" row carries
                # the actual preference text the LLM needs to respect.
                mid = mem.get("preference_key") or "?"
                rows = [
                    ("value", mem.get("preference_value")),
                ]
            else:
                mid = "?"
                rows = []
            header = f"[memory:{cat}:{mid}] (score={rm.score:.2f})"
            lines = [header]
            for k, v in rows:
                if v is None or v == "":
                    continue
                # Coerce non-strings (lists/dicts) to repr; truncate long values
                # so a single fat row doesn't blow the prompt.
                if not isinstance(v, str):
                    v = repr(v)
                if len(v) > 400:
                    v = v[:399] + "…"
                lines.append(f"{k}: {v}")
            if applicable_annotation:
                lines.append(applicable_annotation)
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks) + "\n"

    def validate(
        self,
        agent_result,
        *,
        memories_used: Optional[list[RetrievedMemory]] = None,
    ) -> list[str]:
        """Fan out to the three :mod:`mads.validation` validators.

        All three are v1.1 STUBS that raise ``NotImplementedError`` —
        catch and skip. This method exists so callers can wire it in now;
        the moment any validator gets a real implementation, the wrapper
        starts surfacing its warnings automatically. Never raises.
        """
        from mads.validation.memory_consistency_validator import (
            MemoryConsistencyValidator,
        )
        from mads.validation.metric_usage_validator import MetricUsageValidator
        from mads.validation.result_validator import ResultValidator

        warnings: list[str] = []

        for validator_cls in (
            ResultValidator,
            MetricUsageValidator,
        ):
            try:
                v = validator_cls()
                out = v.validate(agent_result)
                if out:
                    warnings.extend(str(x) for x in out)
            except NotImplementedError:
                continue
            except Exception as e:  # defensive: never let a validator crash the turn
                warnings.append(f"{validator_cls.__name__} raised: {e!r}")

        # MemoryConsistencyValidator has a ``sweep(namespace=...)`` shape —
        # it's a periodic sweep, not a per-turn check, but the spec wants
        # all three wired. Call it with this turn's tenancy.
        try:
            mcv = MemoryConsistencyValidator()
            out = mcv.sweep(namespace=(self._user_id, self._project_id))
            if out:
                # ``sweep`` returns dicts; render them as one-liners.
                for finding in out:
                    if isinstance(finding, dict):
                        warnings.append(
                            f"{finding.get('category', '?')}/{finding.get('row_id', '?')}: "
                            f"{finding.get('issue', '?')}"
                        )
                    else:
                        warnings.append(str(finding))
        except NotImplementedError:
            pass
        except Exception as e:
            warnings.append(f"MemoryConsistencyValidator raised: {e!r}")

        return warnings

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        """Close the shared SQLite connection. Idempotent."""
        if self._closed:
            return
        try:
            if self._conn is not None:
                self._conn.close()
        finally:
            self._closed = True
            self._conn = None  # type: ignore[assignment]


__all__ = ["MemoryManager"]
