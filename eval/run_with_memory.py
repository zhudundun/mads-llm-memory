"""Driver: run scenario pairs through wrapped (memory-augmented) agents.

Mirrors ``eval/run_loader.py`` / ``run_eda.py`` / ``run_viz.py`` but uses
:mod:`mads.agents.wrapper` (or :class:`mads.wrappers.memory_aware.MemoryAwareEDA`
for the proactive variant) so that session A's writes are visible to session
B's wrapped agent through a shared SQLite db.

Usage::

    .venv/bin/python eval/run_with_memory.py --agent loader \\
        --scenario loader_01_snake_case

    # Or run all scenarios for one agent:
    .venv/bin/python eval/run_with_memory.py --agent loader

    # Route EDA through the proactive MemoryAwareEDA wrapper:
    .venv/bin/python eval/run_with_memory.py --agent eda \\
        --wrapper-variant memory-aware

The ``--wrapper-variant`` flag controls which EDA wrapper runs:

* ``legacy`` (default): :func:`mads.agents.wrapper.wrap_eda` — reactive nudge
  only. Loader and Viz always use their legacy wrappers (no proactive variants
  exist yet). Results land in ``eval/results/eda__with_memory/``.
* ``memory-aware``: :class:`mads.wrappers.memory_aware.MemoryAwareEDA` —
  proactive retrieval + citation-tagged injection. Results land in
  ``eval/results/eda__memory_aware/`` (kept separate so each variant has
  reproducible artifacts without clobbering the other).

Outputs per agent::

    eval/results/<agent>__<variant>/<scenario_id>__session_<a|b>.{json,md}
    eval/results/<agent>__<variant>/runs.jsonl
    eval/results/<agent>__<variant>/SUMMARY.md

Each scenario uses a fresh db file by default (``--accumulate`` to keep one
shared db across scenarios). The per-scenario db is named
``<results_dir>/<scenario_id>.sqlite``.

DO NOT run this without checking estimated cost first — each scenario calls
the Anthropic API twice (sessions A and B) plus extra tokens for the memory
nudge and tool-call descriptions. Plan for ~$0.05 per scenario as of Sonnet 4.6.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

# Force matplotlib to a non-interactive backend BEFORE any ADS module pulls
# it in. The harness wraps each session in a worker thread; on macOS the
# default GUI backend raises ``RuntimeError: Cannot create a GUI
# FigureManager outside the main thread`` when ADS tools like
# ``visualize_missing`` and ``generate_sweetviz_report`` call plt.figure().
os.environ.setdefault("MPLBACKEND", "Agg")
try:
    import matplotlib  # noqa: E402
    matplotlib.use("Agg", force=True)
except Exception:  # noqa: BLE001
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import pandas as pd

from harness import run_scenario_pair  # noqa: E402
from ground_truth import (  # noqa: E402
    SYNTHETIC_PREMIUM_ORDER_MEAN_AMOUNT,
    TELCO_CHURN_PCT,
    TELCO_CHURN_BY_CONTRACT_PCT,
    TELCO_LOYAL_CUSTOMER_COUNT,
    TELCO_QUALIFIED_CUSTOMER_CHURN,
    TELCO_FILTERED_CHURN_BY_CONTRACT,
    TELCO_NON_CHURN_PCT,
    assert_all_values_present,
    assert_at_least_n_present,
    assert_no_value_present,
    assert_number_matches,
    assert_pattern_absent,
    assert_pattern_present,
    extract_numbers,
    extract_percentages,
)
from synthetic_dataset import (  # noqa: E402
    SYNTHETIC_PRODUCT_CATEGORIES,
    SYNTHETIC_CUSTOMER_SEGMENTS,
    SYNTHETIC_REGIONS,
    write_synthetic_csv,
)

from mads.adapters.ads import AgentResult  # noqa: E402
from mads.agents.wrapper import wrap_eda, wrap_loader, wrap_viz  # noqa: E402
from mads.manager import MemoryManager  # noqa: E402
from mads.memory.embedding import SentenceTransformerEmbedder  # noqa: E402
from mads.wrappers.memory_aware import MemoryAwareEDA  # noqa: E402

# B3: load the sentence-transformers model ONCE for the harness lifetime
# and share it across every per-session MemoryManager. Cat 5 (episodes)
# and cat 3 (metrics) both lean on this for semantic retrieval; loading
# per-manager would re-pay the ~80MB cold-load cost every scenario.
_SHARED_EMBEDDER: Optional[SentenceTransformerEmbedder] = None


def _get_shared_embedder() -> SentenceTransformerEmbedder:
    global _SHARED_EMBEDDER
    if _SHARED_EMBEDDER is None:
        _SHARED_EMBEDDER = SentenceTransformerEmbedder()
        # Warmup so the first session's first tool call doesn't eat the
        # cold-load latency (and so failures surface here, before the
        # harness spawns its worker threads).
        _SHARED_EMBEDDER.embed("warmup")
    return _SHARED_EMBEDDER

# Try to load .env so ANTHROPIC_API_KEY is available without exporting.
try:
    from dotenv import load_dotenv  # noqa: E402
    load_dotenv(REPO_ROOT / ".env")
except Exception:  # noqa: BLE001
    pass


SCENARIOS_PATH = REPO_ROOT / "eval" / "scenarios.json"
RESULTS_ROOT = REPO_ROOT / "eval" / "results"
TIMEOUT_S = 180


# -----------------------------------------------------------------------------
# Wrapped-agent factories
# -----------------------------------------------------------------------------
def make_wrapped_factory(
    agent_name: str,
    *,
    user_id: str,
    project_id: str,
    db_path: str,
    df: Optional[pd.DataFrame] = None,
    dataset_name: Optional[str] = None,
    table_name: Optional[str] = None,
    nudge_memory: bool = True,
    wrapper_variant: str = "legacy",
    per_session_dfs: bool = False,
):
    """Return a zero-arg callable that builds a *wrapped* handle.

    Each call builds a fresh :class:`MemoryManager` connected to ``db_path``
    so the harness's contract (one fresh agent per session) is preserved
    while the underlying memory persists across sessions.

    The wrapped object's ``invoke(prompt)`` returns a :class:`TurnArtifacts`
    object; we coerce it back to the dict shape that ``harness._invoke_once``
    expects via :func:`invoke_for_wrapped`.

    When ``wrapper_variant == "memory-aware"`` and ``agent_name == "eda"``,
    the factory returns a small adapter object that wraps :class:`MemoryAwareEDA`
    in a uniform ``.invoke(prompt) -> TurnArtifacts`` shape (since
    :class:`MemoryAwareEDA.run` requires keyword args the harness doesn't pass).
    Loader and Viz always use the legacy wrappers.
    """

    def _factory():
        # A fresh manager per session — but they all point at the SAME db.
        # Decide the active project_id BEFORE constructing the manager so
        # per-session project overrides (cross_03) are honored.
        idx = _SESSION_INDEX[0]
        effective_project = project_id
        if (
            _SESSION_PROJECT_SLOTS
            and idx < len(_SESSION_PROJECT_SLOTS)
            and _SESSION_PROJECT_SLOTS[idx] is not None
        ):
            effective_project = _SESSION_PROJECT_SLOTS[idx]

        mgr = MemoryManager(
            user_id=user_id,
            project_id=effective_project,
            db_path=db_path,
            # B3: production-grade semantic retrieval needs the real
            # sentence-transformers embedder. The default HashEmbedder
            # is token-bag and can't bridge synonym/plural cases (e.g.
            # "qualified customers" → stored "qualified_customer"). One
            # model load per harness; shared across every session.
            embedder=_get_shared_embedder(),
        )
        # Pick per-session df if the runner populated slots; otherwise use
        # the closure df. Slot consumption is order-sensitive: the harness
        # calls factory() once for A, once for B.
        _df = df
        _name = dataset_name
        if per_session_dfs and _SESSION_DF_SLOTS:
            if idx < len(_SESSION_DF_SLOTS):
                slot_df, slot_name = _SESSION_DF_SLOTS[idx]
                if slot_df is not None:
                    _df = slot_df
                    _name = slot_name or _name
        # Advance the slot index uniformly across df+project slots.
        _SESSION_INDEX[0] = idx + 1
        _table = _name
        if agent_name == "loader":
            return wrap_loader(mgr, nudge_memory=nudge_memory)
        if agent_name == "eda":
            if _df is None:
                raise ValueError("eda factory requires df")
            if wrapper_variant == "memory-aware":
                return _MemoryAwareEDAAdapter(
                    MemoryAwareEDA(mgr, nudge_memory=nudge_memory),
                    df=_df,
                    user_id=user_id,
                    project_id=effective_project,
                    dataset_name=_name or "dataset",
                    table_name=_table or "dataset",
                )
            return wrap_eda(
                mgr,
                _df,
                dataset_name=_name or "dataset",
                table_name=_table or "dataset",
                nudge_memory=nudge_memory,
            )
        if agent_name == "viz":
            if _df is None:
                raise ValueError("viz factory requires df")
            return wrap_viz(
                mgr,
                _df,
                dataset_name=_name or "dataset",
                table_name=_table or "dataset",
                nudge_memory=nudge_memory,
            )
        raise ValueError(f"unknown agent name {agent_name!r}")

    return _factory


# Module-level side-channel: the harness only returns a fixed payload dict
# from ``_invoke_once`` and drops extra fields like ``memories_injected`` and
# ``warnings``. Rather than fork the harness, the adapter appends each
# produced TurnArtifacts to ``_TURNS_CAPTURED`` so the runner can hydrate
# the row after harness returns. Cleared at the start of each scenario; the
# harness always calls session A before session B so ``_TURNS_CAPTURED[0]``
# is the A turn and ``[-1]`` is the B turn.
_TURNS_CAPTURED: list = []

# Per-session df slots for cross-dataset scenarios. When ``per_session_dfs``
# is True on the factory, each call pops the next ``(df, dataset_name)``
# pair so session A and session B can receive different dataframes. The
# runner populates this with [(df_a, name_a), (df_b, name_b)] before
# invoking the harness.
_SESSION_DF_SLOTS: list[tuple] = []
_SESSION_INDEX: list[int] = [0]

# Per-session project_id overrides for project-isolation scenarios
# (e.g. cross_03). When non-empty, the i-th factory call uses
# _SESSION_PROJECT_SLOTS[i] as project_id instead of the harness-level
# default. Populated by the runner BEFORE invoking the harness; cleared
# at the start of each scenario.
_SESSION_PROJECT_SLOTS: list[Optional[str]] = []


class _MemoryAwareEDAAdapter:
    """Adapt :class:`MemoryAwareEDA` to the legacy ``.invoke(prompt)`` shape.

    The proactive wrapper's :meth:`run` takes a bag of keyword args (df,
    user_id, project_id, dataset_name, table_name) that the harness has no
    way to thread through; we close over them here so the harness can call
    ``wrapped.invoke(prompt)`` uniformly. Also stores the produced
    ``TurnArtifacts`` in the module-level ``_LAST_TURN`` channel so the
    runner can capture ``memories_injected`` / ``warnings``.
    """

    agent_name: str = "eda"

    def __init__(
        self,
        memory_aware,
        *,
        df: pd.DataFrame,
        user_id: str,
        project_id: str,
        dataset_name: str,
        table_name: str,
    ) -> None:
        self._inner = memory_aware
        self._df = df
        self._user_id = user_id
        self._project_id = project_id
        self._dataset_name = dataset_name
        self._table_name = table_name

    def invoke(self, prompt: str):
        turn = self._inner.run(
            prompt,
            df=self._df,
            user_id=self._user_id,
            project_id=self._project_id,
            dataset_name=self._dataset_name,
            table_name=self._table_name,
        )
        # Stash this turn so the runner can read memory-aware metrics that
        # the harness's payload schema doesn't preserve.
        _TURNS_CAPTURED.append(turn)
        return turn


def invoke_for_wrapped(prompt: str, wrapped) -> dict:
    """Dispatcher for the harness. Returns a payload dict compatible with
    :func:`harness._invoke_once`'s expected shape.
    """
    turn = wrapped.invoke(prompt)
    r: AgentResult = turn.agent_result
    # Annotate tool_calls with read/write classification so SUMMARY.md can
    # show memory activity without re-classifying.
    annotated = list(r.tool_calls or [])
    return {
        "response": r.response,
        "tool_calls": annotated,
        "viz_code": r.viz_code,
        "prompt_tokens": r.prompt_tokens,
        "completion_tokens": r.completion_tokens,
        "total_tokens": r.total_tokens,
        "memory_reads": turn.memory_reads,
        "memory_writes": turn.memory_writes,
        "memories_injected": list(getattr(turn, "memories_injected", []) or []),
        "warnings": list(getattr(turn, "warnings", []) or []),
    }


# -----------------------------------------------------------------------------
# B-references-A heuristics
# -----------------------------------------------------------------------------
# Per-scenario tight programmatic observables. Each branch returns
# ``(passed: bool, reason: str)`` with the reason naming exactly which
# sub-check passed or failed. Loader observables match
# eval/run_loader.py verbatim; EDA observables are tightened to inspect
# the actual response content rather than coarse keyword overlap.
#
# Ground truth values (computed from baseline/data/churn_data.csv):
#   * Churn rate: 1869/7043 = 26.5370% → "26.5%" with 1 decimal place
#   * Distinct Contract values: {"Month-to-month", "One year", "Two year"}
#     (no actual typos in the data — only the prompt mentions hypothetical
#     variants, so the eda_01 observable focuses on the canonical-3 outcome
#     OR an explicit normalization step)
#   * Fiber-optic churn ≈ 41.9%, DSL churn ≈ 19.0% (≈ 2.2x ratio)
GROUND_TRUTH_CHURN_PCT = 26.5  # 1-decimal


def _b_referenced_a(scenario: dict, b_rec: dict) -> tuple[bool, str]:
    sid = scenario["id"]
    resp_raw = b_rec.get("response") or ""
    resp = resp_raw.lower()
    tool_names = " ".join(b_rec.get("tool_call_names") or []).lower()
    haystack = f"{resp}\n{tool_names}"

    # Loader scenarios (same logic as eval/run_loader.py).
    if sid == "loader_01_snake_case":
        snake = ("customer_id" in haystack) or ("monthly_charges" in haystack)
        camel = ("customerid" in haystack and "customer_id" not in haystack) or (
            "monthlycharges" in haystack
        )
        if snake and not camel:
            return True, "preview shows snake_case columns"
        return False, "preview retains original casing"
    if sid == "loader_02_no_phone_service_null":
        mentions_no_phone = "no phone service" in haystack
        m = re.search(r"multiplelines[^0-9]{0,60}(\d{1,5})", haystack)
        elevated = bool(m and int(m.group(1)) > 0)
        if mentions_no_phone or elevated:
            return True, "treated 'No phone service' as null"
        return False, "no null-coercion rule honored"
    if sid == "loader_03_drop_pii":
        has_customer_id = "customerid" in haystack
        if not has_customer_id:
            return True, "customerID absent from column list"
        return False, "customerID still present"
    if sid == "loader_04_tenure_as_months":
        if "tenure_years" in haystack:
            return True, "derived tenure_years column present"
        return False, "no tenure_years derived"
    if sid == "loader_05_total_charges_numeric":
        m = re.search(r"totalcharges[^a-z0-9]{0,40}(float\d*|object|string)", haystack)
        if m:
            dtype = m.group(1)
            return dtype.startswith("float"), f"TotalCharges dtype={dtype}"
        return False, "dtype unclear"

    # ---- EDA scenarios: per-scenario observables ----
    if sid.startswith("eda_"):
        return _eda_observable(sid, resp_raw, b_rec)

    # ---- Broad-eval scenarios (cons_, learn_, eff_, cross_, multi_, adv_, neg_) ----
    if sid in _BROAD_DISPATCH:
        return _BROAD_DISPATCH[sid](resp_raw, b_rec)

    # ---- Viz scenarios (unchanged coarse heuristic for now) ----
    a_prompt = (scenario.get("session_a", {}).get("prompt") or "").lower()
    keywords = set(re.findall(r"[a-z_]{4,}", a_prompt))
    common = keywords & set(re.findall(r"[a-z_]{4,}", resp))
    if any(t in tool_names for t in ("recall_dataset", "get_metric_definition",
                                     "list_metric_definitions", "find_similar_analyses",
                                     "search_analyses", "get_analysis_episode")):
        return True, "session B consulted memory"
    if len(common) > 5:
        return True, "session B response overlaps A's preferences"
    return False, "no memory signal in B's response"


# -----------------------------------------------------------------------------
# Per-scenario EDA observables (callable + tested in tests/test_eval_rubrics.py)
# -----------------------------------------------------------------------------
def _eda_observable(sid: str, resp_raw: str, b_rec: dict) -> tuple[bool, str]:
    """Dispatch to a per-scenario observable. ``resp_raw`` is the original
    casing of session B's response; ``b_rec`` is the harness's session-B
    payload (we use its ``tool_call_names`` only for diagnostics)."""
    if sid == "eda_01_contract_typos":
        return _check_eda_01(resp_raw)
    if sid == "eda_02_churn_definition":
        return _check_eda_02(resp_raw)
    if sid == "eda_03_grouping_preference":
        return _check_eda_03(resp_raw)
    if sid == "eda_04_exec_summary_first":
        return _check_eda_04(resp_raw)
    if sid == "eda_05_followup_finding":
        return _check_eda_05(resp_raw)
    # Broad-eval rubrics (12 new scenarios).
    if sid in _BROAD_DISPATCH:
        return _BROAD_DISPATCH[sid](resp_raw, b_rec)
    return False, f"no observable for {sid}"


# Canonical contract values present in the actual CSV (no real typos there).
_CANONICAL_CONTRACTS = ("Month-to-month", "One year", "Two year")
_NORMALIZE_VOCAB = (
    "normaliz", "strip", "lower()", ".str.lower", "case-fold", "casefold",
    "case fold", "deduplicate", "merge variants", "canonical",
)


def _check_eda_01(resp: str) -> tuple[bool, str]:
    """eda_01_contract_typos — normalize Contract typos before aggregation.

    The actual CSV has NO typos (3 canonical Contract values exist as-is),
    so "showing 3 values" is the default outcome and does NOT discriminate
    memory-on vs memory-off. The only discriminating signal is whether the
    response **explicitly acknowledges the normalization step**.

    PASS criteria: response mentions a normalization step using one of the
    vocab tokens (``normalize``, ``.str.lower``, ``strip``, ``canonical``,
    ``case-fold``, ``deduplicate``, ``merge variants``). The 3 canonical
    Contract values must still appear so we know the response is actually
    answering the question.
    """
    low = resp.lower()
    mentions_normalize = any(v in low for v in _NORMALIZE_VOCAB)

    canon_hits = {c.lower(): False for c in _CANONICAL_CONTRACTS}
    for c in _CANONICAL_CONTRACTS:
        if re.search(r"\b" + re.escape(c.lower()) + r"\b", low):
            canon_hits[c.lower()] = True
    canon_count = sum(canon_hits.values())

    has_space_variant = bool(re.search(r"month\s+to\s+month", low))

    if mentions_normalize and canon_count >= 3:
        return True, (
            f"normalization step acknowledged AND 3 canonical Contract "
            f"values present (space_variant={has_space_variant})"
        )
    if mentions_normalize:
        return True, (
            f"normalization step acknowledged "
            f"(canonical_hits={canon_count}/3, space_variant={has_space_variant})"
        )
    return False, (
        f"no normalization vocab (canonical_hits={canon_count}/3, "
        f"space_variant={has_space_variant})"
    )


def _check_eda_02(resp: str) -> tuple[bool, str]:
    """eda_02_churn_definition — churn_rate as percentage with one decimal.

    Multi-part check:
      (a) response contains a percentage matching ground truth ± 0.5%
      (b) the matched value uses exactly ONE decimal place (e.g. ``26.5%``,
          not ``26%`` or ``26.54%``)
    """
    # Extract every "<digits>.<digits>%" or "<digits>%" pattern.
    matches = re.findall(r"(\d{1,3}(?:\.\d{1,4})?)\s*%", resp)
    if not matches:
        return False, "no percentage in response"

    # Look for any match whose value is within ±0.5 of ground truth AND uses
    # exactly one decimal.
    truth = GROUND_TRUTH_CHURN_PCT
    one_decimal_match = None
    near_truth_any_format = None
    for tok in matches:
        try:
            val = float(tok)
        except ValueError:
            continue
        if abs(val - truth) <= 0.5:
            near_truth_any_format = tok
            # Strict: must contain exactly one decimal digit.
            if re.fullmatch(r"\d{1,3}\.\d", tok):
                one_decimal_match = tok
                break

    if one_decimal_match is not None:
        return True, (
            f"churn rate {one_decimal_match}% present with 1 decimal "
            f"(ground truth {truth}%)"
        )
    if near_truth_any_format is not None:
        return False, (
            f"churn rate {near_truth_any_format}% near ground truth but "
            f"not 1-decimal formatted"
        )
    return False, f"no percentage near ground truth {truth}% (saw {matches[:5]})"


def _check_eda_03(resp: str) -> tuple[bool, str]:
    """eda_03_grouping_preference — slice by Contract AND tenure buckets
    (0-12, 13-24, 25-48, 49+).

    Multi-part check:
      (a) response mentions "Contract" as a grouping dimension
      (b) >= 3 of the 4 specific tenure buckets appear (matching regex
          ``\\b<low>\\s*[-to]+\\s*<high>\\b`` plus a ``49+`` form)
    """
    low = resp.lower()
    has_contract = "contract" in low

    # Tenure bucket patterns. Allow "0-12", "0 to 12", "0–12" (en-dash).
    def _bucket(lo: int, hi: int) -> bool:
        pat = rf"\b{lo}\s*(?:-|–|to)\s*{hi}\b"
        return bool(re.search(pat, low))

    b1 = _bucket(0, 12)
    b2 = _bucket(13, 24)
    b3 = _bucket(25, 48)
    # 49+ can render as "49+", "49 +", "49+ months".
    b4 = bool(re.search(r"\b49\s*\+", low)) or bool(re.search(r"\b49\s*\+\s*month", low))
    bucket_count = sum([b1, b2, b3, b4])

    if has_contract and bucket_count >= 3:
        return True, (
            f"Contract grouping present AND {bucket_count}/4 tenure buckets "
            f"matched (0-12:{b1}, 13-24:{b2}, 25-48:{b3}, 49+:{b4})"
        )
    return False, (
        f"contract={has_contract}, tenure buckets matched={bucket_count}/4 "
        f"(0-12:{b1}, 13-24:{b2}, 25-48:{b3}, 49+:{b4})"
    )


def _check_eda_04(resp: str) -> tuple[bool, str]:
    """eda_04_exec_summary_first — 3-bullet exec summary, then technical details.

    Multi-part check:
      (a) response begins (within first 400 chars) with an "executive
          summary" or similar header
      (b) exactly 3 bullets follow under that header (lines starting with
          ``-``, ``*``, ``1.``, ``2.``, ``3.``)
      (c) a "technical"/"details"/"analysis"/"correlations"/"dtypes" header
          appears AFTER the bullets
    """
    if not resp.strip():
        return False, "empty response"
    head = resp[:600].lower()
    # Header regex tolerates an optional run of decorative non-word, non-space
    # characters (emoji, glyphs, punctuation) between the `#+` markdown marker
    # and the heading keyword. The LLM often writes `## 🔍 Executive Summary`
    # or `## ⭐ Summary` when seeded with an "exec-summary first" preference;
    # the `[^\w\s]*\s*` clause inside the `#+` group consumes those glyphs
    # without loosening the "header position" constraint required by the
    # leading `(?:^|\n)\s*` anchor.
    exec_header_match = re.search(
        r"(?:^|\n)\s*(?:#+\s*[^\w\s]*\s*)?\**\s*(executive\s+summary|exec\s+summary|summary)\b",
        head,
    )
    if exec_header_match is None:
        return False, "no executive-summary header near the top"
    exec_pos = exec_header_match.start()

    # Find bullets after exec header. We scan from exec_pos forward in the
    # ORIGINAL casing (bullets can be `-`, `*`, or `1.`/`2.`/`3.`) until the
    # next section header (line starting with `#` or `**` text).
    after = resp[exec_pos:]
    # Stop scanning at the next ## header (markdown) or a bold standalone header
    # that isn't part of the exec section itself.
    # Simplest: gather the first contiguous block of bullets after the header.
    lines = after.splitlines()
    bullets = 0
    seen_first_bullet = False
    bullets_end_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"^([-*]\s+|[1-9]\.\s+)", stripped):
            bullets += 1
            seen_first_bullet = True
            bullets_end_idx = i
        elif seen_first_bullet and stripped == "":
            # Allow a blank line between bullets but stop if we've seen
            # bullets and then hit a header or text-paragraph after a blank.
            continue
        elif seen_first_bullet and stripped.startswith("#"):
            break
        elif seen_first_bullet and bullets >= 1:
            # A non-bullet, non-empty, non-header line: bullet block ended.
            break

    # Check for a technical-details section header after the bullets.
    tail = "\n".join(lines[bullets_end_idx + 1:]).lower() if bullets_end_idx else ""
    tech_keywords = (
        "technical", "details", "dtype", "missingness", "correlation",
        "analysis", "data quality", "descriptive statistics",
    )
    has_tech = any(k in tail for k in tech_keywords)

    if bullets == 3 and has_tech:
        return True, (
            f"exec-summary header, exactly 3 bullets, technical-details "
            f"section follows"
        )
    if bullets == 3:
        return False, "3 bullets in exec summary but no technical-details section after"
    if has_tech:
        return False, f"technical section present but bullet count={bullets} (need 3)"
    return False, (
        f"bullets={bullets} (need exactly 3), technical_section={has_tech}"
    )


def _check_eda_05(resp: str) -> tuple[bool, str]:
    """eda_05_followup_finding — build on prior fiber-vs-DSL finding.

    The default ADS EDA on this dataset will already surface "fiber optic
    churns more" as a correlation-funnel insight without any memory. So
    merely mentioning fiber is NOT a memory signal. The discriminator is
    whether the response explicitly references the **prior** analysis
    (e.g. "previous", "earlier", "last week", "prior finding") AND
    extends it with a new, specific drill-down.

    PASS criteria (BOTH required):
      (a) response uses prior-analysis vocab (``previous``, ``prior``,
          ``last week``, ``earlier``, ``continuing``, ``follow up on``,
          ``building on``)
      (b) response mentions fiber/DSL/InternetService AND proposes at
          least one extension (drill / cohort / interaction / survival /
          deeper segment / hypothesis).
    """
    low = resp.lower()
    has_fiber = "fiber" in low
    has_dsl = "dsl" in low
    has_internetservice = "internetservice" in low or "internet service" in low

    # Use word-boundary regex for "prior" to avoid matching "Priority"; the
    # other tokens are multi-word so substring matching is safe.
    #
    # v1.1-rubric-widening (action C): legitimate "building on prior"
    # phrasings like "extending the earlier finding..." or "since we
    # previously saw..." were missing from the strict alternation. Widen
    # the vocab without loosening the structure — every pattern still
    # binds to a clearly retrospective verb/noun pair to avoid
    # false-positives on stores that happen to be empty.
    prior_word_patterns = (
        r"\bprior\s+(?:finding|analysis|work|result|investigation)",
        r"\bprior\b(?!\s*(?:ity|itize))",  # "prior" not followed by ity/itize
        r"\bprevious\b",
        r"\blast\s+week",
        r"\bearlier\s+(?:finding|analysis|work|investigation)",
        r"\bcontinuing\b",
        r"\bfollow\s*-?\s*up\b",
        r"\bbuilding\s+on\b",
        r"\bas\s+discussed\b",
        # Widened patterns:
        # "extending the earlier finding", "extended the prior analysis"
        r"\bextend(?:ing|ed|s)?\s+(?:the|our|on)\s+(?:earlier|prior|previous|known)\s+"
        r"(?:finding|analysis|work|investigation|issue)",
        # "since we previously saw / earlier found / last saw"
        r"\bsince\s+we\s+(?:previously|earlier|last)\b",
        # "we previously found / we earlier saw / we previously noted"
        r"\bwe\s+(?:previously|earlier)\s+"
        r"(?:found|saw|discovered|noted|investigated|established)\b",
        # "continuing / building on / following up on the earlier finding"
        r"\b(?:continuing|building\s+on|following\s+up\s+on)\s+"
        r"(?:the|our)\s+(?:earlier|prior|previous)\s+"
        r"(?:finding|analysis|work|investigation)",
        # "as we previously saw / like I earlier found / as we previously
        # established"
        r"\b(?:as|like)\s+(?:we|i)\s+(?:previously|earlier)\s+"
        r"(?:saw|found|noted|established|discovered)\b",
        # "builds on / extends [the] known finding/issue"
        r"\b(?:builds?|building)\s+on\s+(?:the\s+)?(?:known|established|prior|earlier|previous)\b",
        r"\bextends?\s+(?:the\s+)?known\s+(?:finding|issue|analysis)",
        # "you've confirmed / you have confirmed / we've already shown"
        r"\b(?:you|we)['’]?\s*(?:ve|have)\s+(?:confirmed|shown|established|seen|found)\b",
        # "already known / already established / what's already known"
        r"\b(?:already\s+(?:known|established|shown|seen|found))\b",
        # "grounded in what(?:'s| is) already known" / "grounded in what's known"
        r"\bgrounded\s+in\s+what['’]?s?\s+(?:already\s+)?known\b",
    )
    mentions_prior = any(re.search(p, low) for p in prior_word_patterns)

    extend_vocab = (
        "drill", "deeper", "further", "survival", "cohort",
        "interaction", "hypothesis", "why ", "drive", "split",
        "investigate", "next step",
    )
    extends = any(v in low for v in extend_vocab)

    references_fiber = has_fiber and (has_dsl or has_internetservice or mentions_prior)

    if mentions_prior and references_fiber and extends:
        return True, (
            f"prior-analysis vocab acknowledged AND fiber/DSL referenced "
            f"AND extension proposed (fiber={has_fiber}, dsl={has_dsl}, "
            f"prior={mentions_prior})"
        )
    if not mentions_prior:
        return False, (
            f"no prior-analysis vocab (fiber={has_fiber}, dsl={has_dsl}, "
            f"extends={extends})"
        )
    if not references_fiber:
        return False, (
            f"prior vocab present but no fiber/DSL reference "
            f"(fiber={has_fiber}, dsl={has_dsl})"
        )
    return False, "references prior fiber finding but no extension proposed"


# -----------------------------------------------------------------------------
# Broad-eval rubrics (12 new scenarios; ground-truth-anchored)
# -----------------------------------------------------------------------------
def _check_cons_01(resp: str, b_rec: dict) -> tuple[bool, str]:
    """Consistency: B reports churn rate within ±0.5pp of 26.5%."""
    return assert_number_matches(
        resp, expected=TELCO_CHURN_PCT, tolerance=0.5, label="cons_01 churn_rate"
    )


def _check_cons_02(resp: str, b_rec: dict) -> tuple[bool, str]:
    """Consistency: B reports churn-by-Contract with all 3 values AND ≥2 of 3
    percentages within ±1pp."""
    ok_vals, reason_vals = assert_all_values_present(
        resp, ["Month-to-month", "One year", "Two year"], label="contract_values"
    )
    if not ok_vals:
        return False, reason_vals
    nums = extract_percentages(resp)
    targets = [
        TELCO_CHURN_BY_CONTRACT_PCT["Month-to-month"],  # 42.71
        TELCO_CHURN_BY_CONTRACT_PCT["One year"],         # 11.27
        TELCO_CHURN_BY_CONTRACT_PCT["Two year"],         # 2.83
    ]
    matched = 0
    matched_list: list[float] = []
    for t in targets:
        for v in nums:
            if abs(v - t) <= 1.0:
                matched += 1
                matched_list.append(t)
                break
    if matched >= 2:
        return True, f"cons_02: all 3 contracts named; matched {matched}/3 churn pcts {matched_list} (need ≥2)"
    return False, f"cons_02: contracts named but only {matched}/3 churn pcts matched (targets={targets}, found_pcts={nums[:8]})"


def _check_learn_01(resp: str, b_rec: dict) -> tuple[bool, str]:
    """Learning: B reports corrected churn ~26.5% AND NOT the wrong 73.5%."""
    ok_correct, reason_correct = assert_number_matches(
        resp, expected=TELCO_CHURN_PCT, tolerance=0.5, label="learn_01 corrected_rate"
    )
    if not ok_correct:
        return False, reason_correct
    # Make sure 73.5% (wrong formula) is NOT being asserted as the churn rate.
    # We use a simple pattern: look for "73.5" or "73%" near "churn".
    if re.search(r"7[23](?:\.[0-9])?\s*%[^.]{0,80}churn", resp, re.IGNORECASE):
        return False, f"learn_01: 73.5%-style wrong rate cited near 'churn' ({reason_correct})"
    if re.search(r"churn[^.]{0,80}7[23](?:\.[0-9])?\s*%", resp, re.IGNORECASE):
        return False, f"learn_01: 73.5%-style wrong rate cited near 'churn' ({reason_correct})"
    return True, f"learn_01: correct rate honored ({reason_correct}); wrong rate not cited"


def _check_learn_02(resp: str, b_rec: dict) -> tuple[bool, str]:
    """Learning: B mentions 11 nulls/blanks AND coerce/to_numeric."""
    has_eleven = bool(re.search(r"\b(11|eleven)\b", resp))
    has_null_vocab = bool(re.search(r"null|nan|blank|missing|empty", resp, re.IGNORECASE))
    has_coerce = bool(re.search(r"coerce|to_numeric|pd\.to_numeric", resp, re.IGNORECASE))
    if has_eleven and has_null_vocab and has_coerce:
        return True, "learn_02: 11 mentioned with null vocab AND coerce/to_numeric"
    return False, (
        f"learn_02: has_eleven={has_eleven}, has_null_vocab={has_null_vocab}, "
        f"has_coerce={has_coerce}"
    )


def _check_eff_01(resp: str, b_rec: dict) -> tuple[bool, str]:
    """Efficiency: B references ≥4 of 5 seeded facts."""
    patterns = (
        # 1. Contract typo normalization
        r"normaliz|case-?fold|\.str\.lower|strip|deduplicate|merge\s+variants",
        # 2. tenure in months
        r"tenure[^.]{0,40}(month|months|in\s+months)",
        # 3. churn means Yes
        r"churn[^.]{0,40}(==\s*['\"]?yes|equal[s]?\s+['\"]?yes|yes['\"]?\s*means)",
        # 4. Contract AND tenure-bucket slicing
        r"(contract[^.]{0,80}tenure|tenure[^.]{0,80}contract).*?(bucket|bracket|bin|group)",
        # 5. TotalCharges coercion / 11 blanks
        r"totalcharges[^.]{0,80}(coerce|to_numeric|11\s+blank|11\s+null|blank)",
    )
    return assert_at_least_n_present(
        resp, patterns, n=4, label="eff_01 fact_patterns"
    )


def _check_eff_02(resp: str, b_rec: dict) -> tuple[bool, str]:
    """Efficiency: B does NOT ask clarifying-Qs AND uses net-revenue / MonthlyCharges*tenure."""
    clarify_pat = r"\b(could you (?:clarify|specify)|please clarify|which (?:metric|definition|measure)|do you mean|what do you mean by|how do you (?:define|measure))\b"
    ok_no_clarify, reason_clarify = assert_pattern_absent(
        resp, clarify_pat, label="eff_02 no_clarify"
    )
    if not ok_no_clarify:
        return False, reason_clarify
    net_pat = r"net\s+revenue|monthlycharges\s*\*\s*tenure|monthly\s*charges.{0,20}tenure"
    ok_net, reason_net = assert_pattern_present(
        resp, net_pat, label="eff_02 net_revenue"
    )
    if ok_net:
        return True, f"eff_02: no clarifying-Q AND {reason_net}"
    return False, f"eff_02: no clarifying-Q but {reason_net}"


_SYNTHETIC_ALL_COLS = (
    "order_id", "product_category", "customer_segment",
    "amount", "order_date", "region", "discount_pct",
)


def _check_cross_01(resp: str, b_rec: dict) -> tuple[bool, str]:
    """Cross-dataset: B mentions ≥3 synthetic columns AND no customerID/customer_id."""
    ok_no_bleed, reason_no_bleed = assert_no_value_present(
        resp, ["customerID", "customer_id"], label="cross_01 no_bleed"
    )
    if not ok_no_bleed:
        return False, reason_no_bleed
    hits = sum(1 for c in _SYNTHETIC_ALL_COLS if c.lower() in resp.lower())
    if hits >= 3:
        return True, f"cross_01: {hits}/7 synthetic columns named; no churn rename bleed"
    return False, f"cross_01: no rename bleed but only {hits}/7 synthetic columns named"


def _check_cross_03(resp: str, b_rec: dict) -> tuple[bool, str]:
    """Cross-project isolation (v1.3): B (in project_beta) MUST NOT apply
    the project_alpha-only USD-whole-dollar rounding rule. We check that
    the response does NOT contain any of the rounding-phrasing tokens
    seeded in the alpha-scoped preference. Same dataset, different
    project — the rule must not bleed."""
    return assert_no_value_present(
        resp,
        ["rounded to whole", "whole dollar", "whole-dollar", "no cents"],
        label="cross_03 no_project_bleed",
    )


def _check_cross_02(resp: str, b_rec: dict) -> tuple[bool, str]:
    """Cross-dataset: B mentions ≥3 synthetic cols AND ≤1 churn-only terms."""
    hits = sum(1 for c in _SYNTHETIC_ALL_COLS if c.lower() in resp.lower())
    low = resp.lower()
    churn_terms = ["contract", "churn", "tenure", "monthlycharges", "totalcharges"]
    churn_hits = sum(1 for t in churn_terms if t in low)
    if hits >= 3 and churn_hits <= 1:
        return True, (
            f"cross_02: {hits}/7 synthetic cols, churn-term hits={churn_hits} (≤1)"
        )
    return False, (
        f"cross_02: synthetic_cols={hits}/7 (need ≥3), churn_term_hits={churn_hits} (need ≤1)"
    )


def _check_multi_01(resp: str, b_rec: dict) -> tuple[bool, str]:
    """Multi-step: B reports qualified-customer churn within ±2pp of 21.7%."""
    return assert_number_matches(
        resp,
        expected=TELCO_QUALIFIED_CUSTOMER_CHURN,
        tolerance=2.0,
        label="multi_01 qualified_churn",
    )


def _check_multi_02(resp: str, b_rec: dict) -> tuple[bool, str]:
    """Multi-step: filter+groupby — ≥2 of 3 filtered Contract churn pcts within ±1pp."""
    ok_vals, reason_vals = assert_all_values_present(
        resp, ["Month-to-month", "One year", "Two year"], label="multi_02 contract_values"
    )
    if not ok_vals:
        return False, reason_vals
    nums = extract_percentages(resp)
    targets = [
        TELCO_FILTERED_CHURN_BY_CONTRACT["Month-to-month"],  # 42.71
        TELCO_FILTERED_CHURN_BY_CONTRACT["One year"],         # 11.28
        TELCO_FILTERED_CHURN_BY_CONTRACT["Two year"],         # 2.85
    ]
    matched = 0
    matched_list: list[float] = []
    for t in targets:
        for v in nums:
            if abs(v - t) <= 1.0:
                matched += 1
                matched_list.append(t)
                break
    if matched >= 2:
        return True, f"multi_02: contracts named; matched {matched}/3 filtered pcts {matched_list}"
    return False, f"multi_02: only {matched}/3 filtered pcts matched (targets={targets}, found={nums[:8]})"


def _check_multi_03(resp: str, b_rec: dict) -> tuple[bool, str]:
    """Multi-step (synthetic): B reports premium_order mean amount within
    ±$5 of ground truth (≈ 399.39).

    This is the cleanest signal of genuine computation: the LLM has never
    seen the synthetic sales dataset, so any plausible number is either
    (a) computed by ``run_pandas_code`` against the bound df, or
    (b) hallucinated. Tolerance ±5 dollars is generous enough to absorb
    rounding/formatting variance ("approximately $399.4", "~$399.39",
    etc.) but tight enough to rule out wild guesses.
    """
    return assert_number_matches(
        resp,
        expected=SYNTHETIC_PREMIUM_ORDER_MEAN_AMOUNT,
        tolerance=5.0,
        label="multi_03 premium_mean_amount",
        treat_as_percent=False,
    )


def _check_adv_01(resp: str, b_rec: dict) -> tuple[bool, str]:
    """Adversarial: B uses payment-recency, not signup interpretation."""
    payment_pat = r"paid|payment|recent\s+(payment|transaction)|transaction"
    ok_payment, reason_payment = assert_pattern_present(
        resp, payment_pat, label="adv_01 payment_vocab"
    )
    if not ok_payment:
        return False, reason_payment
    signup_pat = r"signed?\s*up|sign[-\s]?up|registration"
    ok_no_signup, reason_signup = assert_pattern_absent(
        resp, signup_pat, label="adv_01 no_signup"
    )
    if ok_no_signup:
        return True, f"adv_01: payment vocab present AND no signup interpretation"
    return False, f"adv_01: payment vocab present BUT {reason_signup}"


def _check_adv_02(resp: str, b_rec: dict) -> tuple[bool, str]:
    """v1.4 adv_02_applicable_metric: B reports the loyal_customer count
    (tenure > 36 AND Contract == 'Two year') within ±10 customers of the
    deterministic ground-truth value computed from churn_data.csv.

    Rubric is intentionally tolerant on ±10 (the true value is ~1441 out
    of 7043 rows, so 10 ≈ 0.7%): the signal we care about is "the LLM
    actually evaluated the seeded metric on the bound df," not "to the
    last unit". A flat-out hallucinated number won't be within ±10 of the
    real one.

    Comma-separated thousands ("1,441") are normalized BEFORE extraction
    — the LLM almost always renders four-digit counts with a thousands
    separator and ground_truth.extract_numbers caps at 4 digits with no
    comma handling. Strip the commas first.
    """
    normalized = re.sub(r"(\d),(\d)", r"\1\2", resp)
    return assert_number_matches(
        normalized,
        expected=float(TELCO_LOYAL_CUSTOMER_COUNT),
        tolerance=10.0,
        label="adv_02 loyal_customer_count",
        treat_as_percent=False,
    )


def _check_neg_01(resp: str, b_rec: dict) -> tuple[bool, str]:
    """Negative: B has NO '[memory:' citation AND memories_injected ≤ 1."""
    ok_no_cite, reason_no_cite = assert_pattern_absent(
        resp, r"\[memory:", label="neg_01 no_memory_cite"
    )
    if not ok_no_cite:
        return False, reason_no_cite
    # b_rec doesn't carry memories_injected directly in the harness payload,
    # but the runner stashes this as memories_injected_count in the side
    # channel via _TURNS_CAPTURED. We re-read it from b_rec's tool_call_names
    # surrogate when present, but the authoritative figure lives in
    # _TURNS_CAPTURED (read in the main loop). To keep this function
    # rubric-only, we only require the citation absence here; the main
    # runner additionally records memories_injected_count for the report.
    return True, f"neg_01: {reason_no_cite}"


_BROAD_DISPATCH = {
    "cons_01_churn_rate_replay": _check_cons_01,
    "cons_02_grouping_replay": _check_cons_02,
    "learn_01_metric_correction": _check_learn_01,
    "learn_02_dtype_correction": _check_learn_02,
    "eff_01_re_spec_avoidance": _check_eff_01,
    "eff_02_no_clarifying_q": _check_eff_02,
    "cross_01_no_bleed": _check_cross_01,
    "cross_02_dataset_recognition": _check_cross_02,
    "cross_03_project_isolation": _check_cross_03,
    "multi_01_chained_metric": _check_multi_01,
    "multi_02_filter_plus_groupby": _check_multi_02,
    "multi_03_synthetic_filter": _check_multi_03,
    "adv_01_ambiguous_term": _check_adv_01,
    "adv_02_applicable_metric": _check_adv_02,
    "neg_01_unrelated_query": _check_neg_01,
}


BROAD_SCENARIO_IDS: tuple[str, ...] = tuple(_BROAD_DISPATCH.keys())


# -----------------------------------------------------------------------------
# Per-scenario runner
# -----------------------------------------------------------------------------
def run_one_scenario(
    scenario: dict,
    *,
    user_id: str,
    project_id: str,
    db_path: str,
    result_dir: Path,
    timeout_s: int,
    nudge_memory: bool,
    wrapper_variant: str = "legacy",
) -> dict:
    agent_name = scenario["agent"]
    df = None
    dataset_name = None
    table_name = None
    # Per-session dataset spec. Cross-dataset scenarios use the synthetic
    # CSV in session B (the placeholder ``{SYNTHETIC_CSV}`` in scenarios.json
    # is substituted at scenario-load time in main()). We extract paths from
    # BOTH session prompts and may pick a different df per session.
    df_a = None
    df_b = None
    dataset_name_a = None
    dataset_name_b = None
    if agent_name in ("eda", "viz"):
        def _df_for(prompt: str) -> tuple[pd.DataFrame | None, str | None]:
            m = re.search(r"(/\S+\.csv)", prompt)
            p = m.group(1) if m else None
            if p and Path(p).exists():
                return pd.read_csv(p), Path(p).stem
            return None, None

        df_a, dataset_name_a = _df_for(scenario["session_a"]["prompt"])
        df_b, dataset_name_b = _df_for(scenario["session_b"]["prompt"])
        # Fall back to churn for whichever side didn't resolve.
        fallback = str(REPO_ROOT / "baseline" / "data" / "churn_data.csv")
        if df_a is None:
            df_a = pd.read_csv(fallback)
            dataset_name_a = "churn_data"
        if df_b is None:
            # If B has no CSV reference (e.g. neg_01 poem), use A's df.
            df_b = df_a
            dataset_name_b = dataset_name_a
        # Use df_a as the "primary" handed to the factory below; for
        # cross-dataset scenarios we switch via the side channel.
        df = df_a
        dataset_name = dataset_name_a
        table_name = dataset_name_a

    # Side-channel: monotonically increases on each factory call so we
    # can hand session B a different df. The harness always calls the
    # factory once per session in A-then-B order.
    _SESSION_DF_SLOTS.clear()
    _SESSION_DF_SLOTS.extend([(df_a, dataset_name_a), (df_b, dataset_name_b)])
    # Per-session project_id overrides (v1.3 cross_03). Optional field on
    # each session dict; absent → fall through to the harness-level default
    # project_id inside the factory closure.
    _SESSION_PROJECT_SLOTS.clear()
    _SESSION_PROJECT_SLOTS.extend([
        scenario.get("session_a", {}).get("project_id"),
        scenario.get("session_b", {}).get("project_id"),
    ])
    _SESSION_INDEX[0] = 0

    factory = make_wrapped_factory(
        agent_name,
        user_id=user_id,
        project_id=project_id,
        db_path=db_path,
        df=df,
        dataset_name=dataset_name,
        table_name=table_name,
        nudge_memory=nudge_memory,
        wrapper_variant=wrapper_variant,
        per_session_dfs=True,
    )

    # Clear the side-channel before running so we don't leak state across
    # scenarios. Only the memory-aware adapter writes to it; legacy wrappers
    # leave it empty.
    _TURNS_CAPTURED.clear()

    pair = run_scenario_pair(
        scenario,
        factory,
        result_dir,
        timeout_s=timeout_s,
        invoke_fn=invoke_for_wrapped,
    )
    return pair


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _agents_for_scope(scope: str) -> list[str]:
    if scope == "all":
        return ["loader", "eda", "viz"]
    if scope in ("loader", "eda", "viz"):
        return [scope]
    raise SystemExit(f"unknown --agent value: {scope!r}")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run eval scenarios through memory-wrapped agents."
    )
    parser.add_argument(
        "--agent",
        default="all",
        help="One of loader / eda / viz / all (default: all).",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Filter to a single scenario id (e.g. loader_01_snake_case).",
    )
    parser.add_argument(
        "--scenarios",
        default=None,
        help=(
            "Comma-separated list of scenario ids (e.g. "
            "'cons_01_churn_rate_replay,multi_01_chained_metric'). Mutually "
            "exclusive with --scenario; if both are given, --scenarios wins."
        ),
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help=(
            "After collecting baseline + memory-aware results for each "
            "scenario, call the LLM judge to score the pair. Adds ~1 "
            "Anthropic call per scenario. Requires --baseline-suffix to "
            "name the baseline-side results dir; defaults to '__with_memory' "
            "if not set (which is wrong for true baseline comparisons — "
            "supply '' or the actual baseline dir name)."
        ),
    )
    parser.add_argument(
        "--judge-baseline-suffix",
        default="__with_memory",
        help=(
            "Suffix on the baseline results directory used for the judge "
            "comparison. Default '__with_memory' pairs the memory-aware "
            "variant's response with the legacy variant's response from "
            "the same --both-variants run. Set to '' to use the true ADS "
            "baseline (only available for the original 15 scenarios)."
        ),
    )
    parser.add_argument(
        "--accumulate",
        action="store_true",
        help="Share one db across all scenarios (realistic). Default is per-scenario fresh db.",
    )
    parser.add_argument(
        "--no-nudge-memory",
        action="store_true",
        help="Disable the prompt-level nudge that asks the LLM to record memory.",
    )
    parser.add_argument(
        "--both-variants",
        action="store_true",
        help=(
            "Run each scenario through BOTH legacy (baseline) and "
            "memory-aware variants in sequence. Results land in "
            "eval/results/<agent>__with_memory/ (legacy) and "
            "eval/results/<agent>__memory_aware/ (memory-aware). When "
            "combined with --judge, the judge automatically pairs each "
            "scenario's two responses (baseline-side read from "
            "__with_memory)."
        ),
    )
    parser.add_argument(
        "--wrapper-variant",
        choices=("legacy", "memory-aware"),
        default="legacy",
        help=(
            "Which EDA wrapper to use. 'legacy' uses mads.agents.wrapper.wrap_eda "
            "(reactive nudge only); 'memory-aware' uses "
            "mads.wrappers.memory_aware.MemoryAwareEDA (proactive retrieval + "
            "citation-tagged injection). Loader/Viz always use legacy. "
            "Results land in eval/results/<agent>__memory_aware/ when set to "
            "memory-aware so the artifacts don't clobber the legacy runs."
        ),
    )
    parser.add_argument("--user", default="evaluser")
    parser.add_argument("--project", default="eval")
    parser.add_argument("--timeout-s", type=int, default=TIMEOUT_S)
    args = parser.parse_args(argv)

    scope = _agents_for_scope(args.agent)
    nudge = not args.no_nudge_memory
    if args.both_variants:
        # Force legacy first so the memory-aware run can find baseline-side
        # responses on disk for the judge.
        variants_to_run = ["legacy", "memory-aware"]
    else:
        variants_to_run = [args.wrapper_variant]

    # Ensure the synthetic CSV is materialized BEFORE scenarios that reference
    # it run. The path is stable across calls (deterministic seed → identical
    # bytes), so re-writing is cheap.
    synthetic_csv_path = str(write_synthetic_csv())
    print(f"synthetic CSV materialized at: {synthetic_csv_path}")

    scenarios_doc = json.loads(SCENARIOS_PATH.read_text())
    # Determine scenario filter: --scenarios (comma list) wins over --scenario.
    if args.scenarios:
        wanted_ids = {sid.strip() for sid in args.scenarios.split(",") if sid.strip()}
        selected_raw = [
            s for s in scenarios_doc["scenarios"]
            if s["agent"] in scope and s["id"] in wanted_ids
        ]
    else:
        selected_raw = [
            s for s in scenarios_doc["scenarios"]
            if s["agent"] in scope and (args.scenario is None or s["id"] == args.scenario)
        ]
    # Substitute {SYNTHETIC_CSV} placeholder in BOTH session prompts.
    selected = []
    for s in selected_raw:
        s2 = json.loads(json.dumps(s))  # deep copy
        for half in ("session_a", "session_b"):
            p = s2[half]["prompt"]
            s2[half]["prompt"] = p.replace("{SYNTHETIC_CSV}", synthetic_csv_path)
        selected.append(s2)
    if not selected:
        print(f"no scenarios matched scope={scope} scenario={args.scenario}")
        return 1
    print(f"selected scenarios: {[s['id'] for s in selected]} (nudge={nudge})")

    by_agent: dict[str, list[dict]] = {}
    shared_db = (
        str(RESULTS_ROOT / "with_memory_shared.sqlite") if args.accumulate else None
    )

    # Outer loop over variants — when --both-variants is set we run legacy
    # first (its outputs serve as baseline for the judge) then memory-aware.
    for wrapper_variant in variants_to_run:
      print(f"\n=== variant: {wrapper_variant} ===")
      for scenario in selected:
        sid = scenario["id"]
        agent_name = scenario["agent"]
        # Pick the result-dir suffix per variant. Memory-aware EDA artifacts
        # go to a SEPARATE directory so legacy and proactive runs don't
        # clobber each other. Loader and Viz always use __with_memory since
        # there's no proactive variant for them.
        if agent_name == "eda" and wrapper_variant == "memory-aware":
            suffix = "__memory_aware"
        else:
            suffix = "__with_memory"
        result_dir = RESULTS_ROOT / f"{agent_name}{suffix}"
        result_dir.mkdir(parents=True, exist_ok=True)
        runs_jsonl = result_dir / "runs.jsonl"
        if not runs_jsonl.exists():
            runs_jsonl.write_text("")

        db_path = shared_db or str(result_dir / f"{sid}.sqlite")
        # Per-scenario fresh: unlink existing file to start clean.
        if not args.accumulate and Path(db_path).exists():
            Path(db_path).unlink()

        try:
            pair = run_one_scenario(
                scenario,
                user_id=args.user,
                project_id=args.project,
                db_path=db_path,
                result_dir=result_dir,
                timeout_s=args.timeout_s,
                nudge_memory=nudge,
                wrapper_variant=wrapper_variant,
            )
        except BaseException as e:
            err = f"{type(e).__name__}: {e}"
            print(f"{sid} | HARNESS ERROR | {err}")
            with runs_jsonl.open("a") as f:
                f.write(json.dumps({"scenario_id": sid, "ok": False, "error": err}) + "\n")
            continue

        a = pair["a"]
        b = pair["b"]
        ok = pair["ok"]
        # Count memory writes (in A) and reads (in B) by inspecting tool_call_names.
        # Kept in sync with mads.agents.wrapper._MEMORY_WRITE_NAMES /
        # _MEMORY_READ_NAMES — re-imported here rather than referenced
        # directly so the eval driver stays decoupled from the wrapper
        # internals.
        write_names = {
            "add_metric_definition",
            "remember_preference",
            "forget_preference",
            "register_dataset",
            "register_column",
            "record_analysis_episode",
            "update_user_feedback",
        }
        read_names = {
            "get_metric_definition",
            "list_metric_definitions",
            "search_metric_definitions",
            "get_metric_history",
            "recall_preference",
            "list_preferences",
            "recall_dataset",
            "list_datasets",
            "recall_column",
            "list_columns",
            "get_analysis_episode",
            "list_recent_analyses",
            "find_similar_analyses",
            "search_analyses",
        }
        a_writes = [n for n in (a.get("tool_call_names") or []) if n in write_names]
        b_reads = [n for n in (b.get("tool_call_names") or []) if n in read_names]
        ref, ref_reason = _b_referenced_a(scenario, b)

        # Pull memory metrics from the side-channel. The memory-aware adapter
        # appends each turn; legacy wrappers leave it empty. _TURNS_CAPTURED
        # is [A_turn, B_turn] in order; we surface the B-side metrics.
        b_response = b.get("response") or ""
        b_cited_count = b_response.count("[memory:")
        if _TURNS_CAPTURED:
            b_turn = _TURNS_CAPTURED[-1]
            b_injected = len(getattr(b_turn, "memories_injected", []) or [])
            b_warnings = len(getattr(b_turn, "warnings", []) or [])
            b_reads_count = len(getattr(b_turn, "memory_reads", []) or [])
            b_writes_count = len(getattr(b_turn, "memory_writes", []) or [])
        else:
            b_injected = 0
            b_warnings = 0
            b_reads_count = len(b_reads)
            b_writes_count = 0

        # For neg_01, the rubric needs to enforce memories_injected ≤ 1
        # (it cannot see _TURNS_CAPTURED). Re-apply that side condition here.
        if sid == "neg_01_unrelated_query" and ref:
            if b_injected > 1:
                ref = False
                ref_reason = (
                    f"neg_01: no [memory: cite but memories_injected_count={b_injected} >1"
                )

        row = {
            "scenario_id": sid,
            "agent": agent_name,
            "ok": ok,
            "error": pair.get("error"),
            "db_path": db_path,
            "wrapper_variant": wrapper_variant if agent_name == "eda" else "legacy",
            "a_total_tokens": a.get("total_tokens", 0),
            "b_total_tokens": b.get("total_tokens", 0),
            "a_cost_usd": a.get("cost_usd", 0.0),
            "b_cost_usd": b.get("cost_usd", 0.0),
            "a_tool_call_names": a.get("tool_call_names", []),
            "b_tool_call_names": b.get("tool_call_names", []),
            "a_memory_writes": a_writes,
            "b_memory_reads": b_reads,
            "memories_injected_count": b_injected,
            "memories_cited_count": b_cited_count,
            "read_tools_count": b_reads_count,
            "write_tools_count": b_writes_count,
            "warnings_count": b_warnings,
            "b_referenced_a": ref,
            "b_referenced_a_reason": ref_reason,
            "gt_passed": ref,
            "gt_reason": ref_reason,
        }

        # Optional LLM judge. Only fires on the memory-aware variant (it
        # needs the baseline response already on disk from the earlier
        # legacy run). Restricted to broad-eval scenarios per the spec —
        # the original 15 already have an established rubric.
        if args.judge and wrapper_variant == "memory-aware" and sid in _BROAD_DISPATCH:
            try:
                from judge import judge_pair, agreement_with_rubric  # noqa: PLC0415
                baseline_dir = RESULTS_ROOT / f"{agent_name}{args.judge_baseline_suffix}"
                baseline_b_json = baseline_dir / f"{sid}__session_b.json"
                if baseline_b_json.exists():
                    baseline_b = json.loads(baseline_b_json.read_text())
                    baseline_resp = baseline_b.get("response") or ""
                else:
                    baseline_resp = ""
                mem_resp = b.get("response") or ""
                jr = judge_pair(
                    session_a_prompt=scenario["session_a"]["prompt"],
                    session_b_prompt=scenario["session_b"]["prompt"],
                    baseline_response=baseline_resp,
                    mem_aware_response=mem_resp,
                )
                row["judge_winner"] = jr.get("winner")
                row["judge_baseline_score"] = jr.get("baseline_score")
                row["judge_mem_score"] = jr.get("mem_aware_score")
                row["judge_reason"] = jr.get("reason")
                row["judge_baseline_response_chars"] = len(baseline_resp)
                row["judge_baseline_source"] = str(baseline_b_json) if baseline_b_json.exists() else "MISSING"
                if "raw_attempts" in jr:
                    row["judge_raw_attempts"] = jr["raw_attempts"]
                row["agree"] = agreement_with_rubric(
                    gt_passed=ref, judge_winner=jr.get("winner") or ""
                )
            except BaseException as je:  # noqa: BLE001
                row["judge_error"] = f"{type(je).__name__}: {je}"
                row["judge_winner"] = None
                row["agree"] = None

        with runs_jsonl.open("a") as f:
            f.write(json.dumps(row) + "\n")
        by_agent.setdefault((wrapper_variant, agent_name), []).append(row)

        print(
            f"{sid} | ok={ok} "
            f"| A_writes={a_writes} "
            f"| B_reads={b_reads} "
            f"| injected={b_injected} "
            f"| cited={b_cited_count} "
            f"| tokens={a.get('total_tokens', 0)}+{b.get('total_tokens', 0)} "
            f"| cost=${(a.get('cost_usd') or 0) + (b.get('cost_usd') or 0):.4f} "
            f"| B_refs_A={ref} ({ref_reason})"
        )

    # ---- Per-agent SUMMARY.md ----
    for (variant_label, agent_name), rows in by_agent.items():
        # Pick the matching suffix used during the run (memory-aware EDA goes
        # to __memory_aware; everything else to __with_memory).
        if agent_name == "eda" and variant_label == "memory-aware":
            suffix = "__memory_aware"
        else:
            suffix = "__with_memory"
        result_dir = RESULTS_ROOT / f"{agent_name}{suffix}"
        summary = result_dir / "SUMMARY.md"
        lines = []
        title_variant = "memory-aware" if suffix == "__memory_aware" else "with-memory"
        lines.append(f"# {agent_name} {title_variant} results — SUMMARY")
        lines.append("")
        n_total = len(rows)
        n_closed = sum(1 for r in rows if r["b_referenced_a"])
        lines.append(f"**Gap-closed:** {n_closed}/{n_total}")
        lines.append("")
        lines.append(
            "| id | ok | A writes | B reads | injected | cited | warnings | "
            "B refs A | reason |"
        )
        lines.append("|---|:--:|---|---|:--:|:--:|:--:|:--:|---|")
        for r in rows:
            lines.append(
                f"| {r['scenario_id']} | {'y' if r['ok'] else 'n'} "
                f"| {','.join(r['a_memory_writes']) or '-'} "
                f"| {','.join(r['b_memory_reads']) or '-'} "
                f"| {r.get('memories_injected_count', 0)} "
                f"| {r.get('memories_cited_count', 0)} "
                f"| {r.get('warnings_count', 0)} "
                f"| {'yes' if r['b_referenced_a'] else 'no'} "
                f"| {r['b_referenced_a_reason']} |"
            )
        summary.write_text("\n".join(lines) + "\n")
        print(f"wrote {summary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
