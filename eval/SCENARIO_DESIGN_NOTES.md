# Scenario design notes

## Why cross-session-only

The whole point of this baseline is to make the comparison against a
memory-augmented version *clean*. To do that, every scenario tests exactly
**one thing**: whether information from a prior session can flow into a fresh
agent invocation. We achieve this by:

1. Spinning up a *new* agent instance for session B (a literal new
   `DataLoaderToolsAgent(...)` / `EDAToolsAgent(...)` / `DataVisualizationAgent(...)`).
2. Never passing any conversation history, checkpointer, or shared state
   between A and B.

The baseline result for every scenario should therefore be: B behaves
identically to a cold-start request, oblivious to A. When memory is wired in,
B should reflect the A-session fact. The observable is defined narrowly
enough to scriptably classify each B output as "honored" vs "did not honor."

We deliberately exclude two adjacent test classes:

- **Single-turn smoke**: already covered by `baseline/test_*.py`. We don't
  re-test that the agents run end-to-end.
- **Multi-turn within a session**: the langgraph runtime within a single
  `invoke_agent` call already passes intermediate state. That's plumbing, not
  memory. It would conflate the experiment.

## Why these 5 categories per agent

Each agent's five scenarios probe a different *axis* of what a memory layer
should preserve, in rough order of stakes:

| Axis                           | Loader scenario | EDA scenario        | Viz scenario           |
|--------------------------------|-----------------|---------------------|------------------------|
| stylistic / format             | snake_case      | exec_summary_first  | exec_minimal           |
| structural / shape             | drop_pii        | grouping_preference | tenure_bucket_grain    |
| dtype / parsing / definition   | total_charges_num + no_phone_service_null | churn_definition | log_y_axis (conditional rule) |
| derived feature / transformation | tenure_as_months | contract_typos    | contract_color_map     |
| longitudinal / continuation    | (n/a — loader rarely continues threads) | followup_finding | horizontal_viridis (cross-chart consistency) |

The matrix isn't perfectly orthogonal (loader has two "dtype" rows because
that's where loader memory bites hardest; viz has consistency-across-charts
instead of follow-up because viz threads are rarely "continue last
investigation"). It's good enough to surface failure modes broadly.

## Why this concrete dataset

`churn_data.csv` (telecom churn, 7043×21) has:
- camelCase + PascalCase column names → casing preferences make sense,
- a known dtype gotcha (`TotalCharges` has 11 blanks that force object dtype),
- a known sentinel string (`'No phone service'`) used in 6 columns,
- both numerics (tenure, charges) and categoricals (contract, payment) so the
  viz scenarios have natural targets,
- a clear KPI (churn rate) that organizations actually define and re-define.

It's also small enough that none of these scenarios will hit the 90 s
per-session timeout in practice.

## What's deliberately NOT tested (this pass)

- **Memory eviction / contradiction handling.** If session A says "always
  snake_case" and session C says "always camelCase", which wins? Out of scope.
- **Cross-agent memory.** A loader-A preference (snake_case) carrying into an
  EDA-B run. Plausible-future, but conflates two memory mechanisms.
- **Tool-call-level observables.** We capture tool names/counts for forensics
  but no scenario *asserts* on a tool-call pattern. The semantics live in the
  AI message / artifacts, which is where users actually see them.
- **Multi-step session-A.** Every A is a single prompt. Real users build up
  preferences across multiple exchanges; we save that for v2.

## Acceptable variance / known soft observables

A few observables are inherently fuzzy ("AI message mentions the fiber
finding"). For those, Phase 3 grading will need either an LLM-judge pass or a
regex/keyword check. The `observable` field in `scenarios.json` is written to
be greppable in the simple cases (`'customer_id'`, `'float64'`,
`'showlegend=False'`) and as a natural-language assertion for the soft ones —
explicit so the grader knows which lane it's in.

## Cost estimate

30 live invocations at gpt-4o ($2.50/M in + $10/M out, ≈ $6/M blended at a
1:1 in:out ratio). Per-invocation token spend for these tool-using agents is
empirically in the 8–15k range (system prompts + tool schemas dominate;
responses are short). Rough total: 30 × 12k × $6/M ≈ **$2.16**, with most
spend on the EDA agent (longest tool chains) and least on the loader.
