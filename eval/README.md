# ADS Live Baseline (Sonnet 4.6)

This directory holds the live, behavioral baseline for the `ai_data_science_team`
(ADS) loader / EDA / viz agents running WITHOUT any memory layer. Every
scenario is a paired prompt: session A states a fact or preference, session B
is issued to a literally fresh agent instance and — by design — has no
mechanism to recall A. The point is to record exactly what B does today so a
later A/B against a memory-augmented stack has a fixed, reproducible
counterfactual on the same 15 scenarios.

Cross-session-pair design only. No single-turn smoke tests; no multi-turn
within a single `invoke_agent` call. The single thing under test is whether
information from a prior session can flow into a fresh agent invocation.

## Headline result

- 11 / 15 pairs ran clean end-to-end (loader 5/5, eda 5/5, viz 1/5).
- The 4 viz failures are ALL the same upstream ADS bug: a malformed regex in
  `ai_data_science_team/agents/data_visualization_agent.py:775` raises
  `re.PatternError: unbalanced parenthesis at position 10` on the retry path.
  Not a defect in this harness, scenarios, or the Anthropic adapter.
- ~164.6k total tokens, ~$0.72 USD across all 30 live invocations
  (Sonnet 4.6 list pricing, `$3/M` in + `$15/M` out).
- ALL 11 clean pairs show the expected baseline state: `B_referenced_A = no`
  (loader 5/5, eda 4/5 with one heuristic false-positive on `eda_05`, viz 1/1
  on `viz_04`). I.e. memory-applied rate at baseline is effectively 0/15 —
  exactly the gap an A/B with memory should close.

## Results table

One row per scenario. `B ref A?` column: `no` = expected baseline gap (memory
should later flip this to `yes`); `yes` = weak signal (B coincidentally
honored A); `n/a` = blocked by upstream bug, no comparable B artifact.

| id                              | agent  | tokens (a+b)        | cost (a+b)              | status            | B ref A? | one-line observation                                                                                                  |
|---------------------------------|--------|---------------------|-------------------------|-------------------|----------|-----------------------------------------------------------------------------------------------------------------------|
| loader_01_snake_case            | loader | 4923+5234=10157     | $0.0185+$0.0235=$0.0420 | ok                | no       | preview retains original casing (customerID/MonthlyCharges)                                                           |
| loader_02_no_phone_service_null | loader | 5027+4951=9978      | $0.0198+$0.0193=$0.0391 | ok                | no       | MultipleLines reported as 0 missing (no null-coercion rule)                                                           |
| loader_03_drop_pii              | loader | 4997+5031=10028     | $0.0194+$0.0205=$0.0399 | ok                | no       | customerID still present in column list                                                                               |
| loader_04_tenure_as_months      | loader | 5054+5054=10108     | $0.0199+$0.0208=$0.0407 | ok                | no       | no tenure_years column derived                                                                                        |
| loader_05_total_charges_numeric | loader | 5046+5123=10169     | $0.0199+$0.0218=$0.0417 | ok                | no       | TotalCharges dtype=object (default, no coercion)                                                                      |
| eda_01_contract_typos           | eda    | 16813+8922=25735    | $0.0737+$0.0318=$0.1055 | ok                | no       | raw groupby, no typo-normalization signal                                                                             |
| eda_02_churn_definition         | eda    | 2420+2383=4803      | $0.0080+$0.0078=$0.0158 | ok                | no       | no one-decimal pct format detected                                                                                    |
| eda_03_grouping_preference      | eda    | 18086+9603=27689    | $0.0902+$0.0428=$0.1330 | ok                | no       | no Contract x tenure-bucket cross-cut with edges 12/24/48                                                             |
| eda_04_exec_summary_first       | eda    | 2418+2387=4805      | $0.0079+$0.0078=$0.0158 | ok                | no       | no exec-summary-first / technical-details template                                                                    |
| eda_05_followup_finding         | eda    | 16208+15772=31980   | $0.0656+$0.0619=$0.1275 | ok                | yes\*    | "references fiber/DSL/InternetService" — heuristic FP; B independently rediscovered the pattern via its own EDA tools |
| viz_01_horizontal_viridis       | viz    | 1880+1727=3607      | $0.0103+$0.0106=$0.0209 | blocked-upstream  | n/a      | A produced horizontal viridis bar; B crashed on PatternError before code/figure persisted                             |
| viz_02_log_y_axis               | viz    | 0+0=0               | $0.0000+$0.0000=$0.0000 | blocked-upstream  | n/a      | both sessions hit PatternError before any LLM completion was recorded (token cb saw 0)                                |
| viz_03_exec_minimal             | viz    | 0+4026=4026         | $0.0000+$0.0254=$0.0254 | blocked-upstream  | n/a      | B's code has title + showlegend=False but NO insight subtitle — fails 3-part exec rule (partial gap evidence)         |
| viz_04_contract_color_map       | viz    | 3522+3867=7389      | $0.0192+$0.0239=$0.0431 | ok                | no       | "B used `color_discrete_sequence=px.colors.qualitative.Set2` — totally different palette. Strongest memory-gap evidence." |
| viz_05_tenure_bucket_grain      | viz    | 4090+0=4090         | $0.0255+$0.0000=$0.0255 | blocked-upstream  | n/a      | A bucketed 0-12/13-24/25-48/49+ correctly; B crashed before producing code                                            |

\* `eda_05` flagged "yes" by the keyword detector because B's own response
mentioned `InternetService = Fiber optic` as a churn driver. But B reached
that finding via its own correlation-funnel tool call, not by recalling A's
prior context — see `## Known issues` for the false-positive note.

## What the baseline proves

For every clean pair, B's observable diverges from A's stated preference in a
specific, scriptable way. A memory layer that closes the gap should flip the
`B_referenced_A` column for these IDs:

- `loader_03_drop_pii` — PII column `customerID` survives into B's column
  listing. With memory it should be dropped at load.
- `loader_05_total_charges_numeric` — `TotalCharges` stays `object` dtype in
  B (the well-known 11-blank gotcha). Memory would carry the
  `errors='coerce'` numeric coercion forward.
- `loader_01_snake_case` — B's preview shows `customerID`, `MonthlyCharges`.
  Memory would carry the snake_case normalization preference.
- `eda_03_grouping_preference` — B uses a flat Contract-only-ish breakdown.
  Memory would reproduce the Contract × tenure-bucket-with-edges-(12,24,48)
  cross-cut.
- `eda_04_exec_summary_first` — B's "summary of this dataset" is an
  un-templated narrative with no 3-bullet exec / technical-details structure.
  Memory would carry the template forward.
- `viz_04_contract_color_map` — B reaches for
  `color_discrete_sequence=px.colors.qualitative.Set2`. Memory would reuse
  the explicit Month-to-month=red / One year=blue / Two year=green mapping
  from A.

These six are the cleanest "memory will help" signals in the set. The
remaining clean pairs (`loader_02`, `loader_04`, `eda_01`, `eda_02`) are
weaker but still concretely observable; see the per-scenario rows above.

## Cost

| agent  | pairs | tokens (sum a+b) | cost USD |
|--------|------:|-----------------:|---------:|
| loader |     5 |           50,440 |  $0.2035 |
| eda    |     5 |           95,012 |  $0.3976 |
| viz    |     5 |           19,112 |  $0.1149 |
| TOTAL  |    15 |          164,564 |  $0.7160 |

Pricing constants (defined in `eval/harness.py`):

```
PRICE_INPUT_USD_PER_M  = 3.00
PRICE_OUTPUT_USD_PER_M = 15.00
```

Model id (defined in `eval/agent_factories.py`):

```
ANTHROPIC_MODEL = "claude-sonnet-4-6"
```

If you swap models, update BOTH constants in lockstep.

## Repro from scratch

Prerequisites: Python venv at `/Users/wang/LLM_memory/.venv` with
`langchain-anthropic`, `ai-data-science-team`, `pandas`, `plotly` already
installed (these come with the project venv).

1. Make `.env` at the repo root with at minimum:

   ```
   ANTHROPIC_API_KEY=sk-ant-...
   ```

2. Install the EDA-side extras Sonnet reaches for during EDA tool calls
   (sweetviz / missingno / pytimetk are imported lazily by the ADS EDA tools):

   ```bash
   /Users/wang/LLM_memory/.venv/bin/pip install sweetviz missingno pytimetk
   ```

3. Run each driver. Each one truncates its own `runs.jsonl`, writes
   `SUMMARY.md`, and per-session `.json`/`.md` artifacts under
   `eval/results/<agent>/`:

   ```bash
   cd /Users/wang/LLM_memory && set -a && source .env && set +a && .venv/bin/python eval/run_loader.py 2>&1 | tee eval/results/loader/run.log
   cd /Users/wang/LLM_memory && set -a && source .env && set +a && MPLBACKEND=Agg .venv/bin/python eval/run_eda.py 2>&1 | tee eval/results/eda/run.log
   cd /Users/wang/LLM_memory && set -a && source .env && set +a && .venv/bin/python eval/run_viz.py 2>&1 | tee eval/results/viz/run.log
   ```

   The `MPLBACKEND=Agg` prefix on the EDA driver is REQUIRED. The default
   matplotlib backend on macOS is not thread-safe, and the harness runs each
   session inside a worker thread (for wallclock timeout). Without `Agg`,
   matplotlib-based EDA tool calls (`visualize_missing`, etc.) segfault or
   hang. The loader and viz drivers don't import matplotlib in the hot path
   and don't need the prefix.

4. (Optional) sanity-check tokens + cost match the table above:

   ```bash
   wc -l /Users/wang/LLM_memory/eval/results/{loader,eda,viz}/runs.jsonl
   ```

   Each `runs.jsonl` should have 5 lines (one per pair, written incrementally
   as each pair finishes so a crash mid-batch doesn't lose earlier rows).

## Known issues

- **Upstream ADS regex bug (4/5 viz pairs fail).** File:
  `/Users/wang/LLM_memory/.venv/lib/python3.13/site-packages/ai_data_science_team/agents/data_visualization_agent.py:775`.
  Offending line:

  ```python
  for match in re.findall(r"\\['([^']+)'\\]\\s+not in index", error_text):
  ```

  The pattern has escaped square brackets in a way Python's `re` engine
  reports as `unbalanced parenthesis at position 10`. This fires inside the
  retry-on-missing-column path; it triggers whenever generated viz code
  raises a column-not-found KeyError and the agent tries to extract the
  missing column name. Suggested remediation paths (NOT applied in this
  pass): (a) file an upstream issue against
  `business-science/ai-data-science-team` with a one-line patch, or
  (b) add a local `eval/wrappers/` shim that monkey-patches
  `_extract_missing_columns` before constructing the viz agent. Do NOT patch
  the installed package directly.

- **`eda_05_followup_finding` is a heuristic FP.** The detector in
  `run_eda.py::_detect_b_references_a` flags any mention of "fiber" / "dsl"
  / "internetservice" in B's response as "B referenced A." In this run B
  produced those strings on its own — it called `generate_correlation_funnel`
  and surfaced `InternetService = Fiber optic` as a top churn driver
  organically. The per-agent SUMMARY documents this as a heuristic FP. When
  interpreting the `B referenced A?` column, treat `eda_05` "yes" as a known
  false-positive against the baseline state. The proper grader for this
  scenario is an LLM judge that distinguishes "extending a prior finding"
  from "rediscovering a feature importance."

- **Anthropic rate limits.** Sonnet 4.6 input-token rate limit is ~30k
  TPM. The EDA driver's `eda_03` and `eda_05` pairs each draw ~28k tokens in
  session A alone; a back-to-back rerun has brushed the limit once during
  this baseline (first attempt 429'd, retry passed). If you re-run inside a
  minute, add a short sleep between driver invocations or expect a one-time
  blip.

- **viz_02 token cb shows 0.** Both sessions of `viz_02_log_y_axis` recorded
  0 tokens. Cause: the upstream PatternError fires inside the agent's first
  internal step (before any LLM completion reaches the
  `BaseCallbackHandler.on_llm_end` hook), so `_UsageCallback` never sees a
  generation. Cost row reads $0.0000 — accurate for accounting, but the
  per-session latency row (25,572 / 13,836 ms) is real wallclock spent
  inside the agent before the crash.

## How to A/B against a memory-augmented version (later)

1. **Treat `eval/scenarios.json` as the contract.** Don't edit it for the
   A/B run. The fairness of the comparison rests on B getting the exact
   same prompt with no preamble.
2. **Re-use the harness.** A memory-augmented runner should call the SAME
   `eval.harness.run_scenario_pair(scenario, agent_factory, result_dir,
   timeout_s, invoke_fn=...)` API. The only thing that changes is the
   `agent_factory` and (optionally) `invoke_fn`: build a factory that wraps
   the ADS agent in your memory layer (e.g. attach a session id, hydrate a
   prelude from a memory store before `invoke_agent`).
3. **Persist into a sibling results dir.** E.g.
   `eval/results/loader_mem/`, `eval/results/eda_mem/`,
   `eval/results/viz_mem/`. Don't overwrite the baseline.
4. **Per-scenario diff.** For each scenario id, compare baseline
   `runs.jsonl` row vs memory-augmented `runs.jsonl` row. The headline lift
   is: did `B_referenced_A` (or the viz `b_followed_preference` verdict)
   flip from `no` → `yes`?
5. **Suggested aggregate metric.**
   `memory_applied_rate = #(pairs where B_referenced_A == "yes") / #(pairs total)`.
   Baseline = effectively 0/15 (1/15 if you count the `eda_05` heuristic
   FP). Anything materially above 0/15 is real signal; for an A/B writeup
   the table to publish is: scenario id | baseline verdict | memory verdict
   | flipped? | one-line evidence.
6. **Don't conflate.** Single-turn smoke tests (`baseline/test_*.py`) and
   in-session multi-turn (langgraph runtime state) are deliberately out of
   scope. If a memory-augmented run regresses one of those, that's a
   separate bug, not an A/B finding.

## File map

```
eval/
├── README.md                          ← this file
├── SCENARIOS.md                       ← human-readable 15-scenario catalogue
├── SCENARIO_DESIGN_NOTES.md           ← design rationale + axis matrix
├── scenarios.json                     ← machine-readable scenarios (the contract)
├── harness.py                         ← agent-agnostic paired-session runner
├── agent_factories.py                 ← Anthropic adapter + per-agent dispatcher
├── run_loader.py                      ← driver: loader 5 scenarios
├── run_eda.py                         ← driver: eda 5 scenarios (needs MPLBACKEND=Agg)
├── run_viz.py                         ← driver: viz 5 scenarios (4/5 hit upstream bug)
└── results/
    ├── loader/
    │   ├── SUMMARY.md                 ← one-page per-agent rollup
    │   ├── runs.jsonl                 ← one line per scenario pair
    │   ├── run.log                    ← stdout from the driver
    │   └── <id>__session_<a|b>.{json,md}   (10 per agent)
    ├── eda/                           ← same layout as loader/
    └── viz/                           ← same layout PLUS:
        ├── figures/<id>__session_<a|b>.html   (only for sessions that produced a fig)
        └── code/<id>__session_<a|b>.py        (only for sessions that produced code)
```

External dependencies referenced by this directory but living elsewhere:

```
/Users/wang/LLM_memory/baseline/data/churn_data.csv   ← dataset (7043 × 21)
/Users/wang/LLM_memory/baseline/SETUP_REPORT.md       ← original ADS API contract
/Users/wang/LLM_memory/.env                           ← ANTHROPIC_API_KEY
/Users/wang/LLM_memory/.venv                          ← project venv
/Users/wang/LLM_memory/CLAUDE.md                      ← project context
```
