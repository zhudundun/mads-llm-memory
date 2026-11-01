"""Live smoke test for :class:`mads.wrappers.memory_aware.MemoryAwareEDA`.

What it proves
--------------
The proactive-injection wrapper actually retrieves a relevant metric
definition and threads it into the prompt that hits the LLM. We run two
queries sequentially against the same wrapper instance:

1. **Run 1**: bare token ``"conversion"`` — regression check that the
   legacy substring path still finds the metric.
2. **Run 2**: natural language ``"Show conversion by channel"`` — proves
   the new tokenized-match path now finds the metric where the
   substring-only path used to return ``[]``.

For each run we:

* capture the prompt that was sent to ADS
* print the LLM response
* assert ``len(memories_injected) >= 1``
* assert the metric's business_definition substring shows up in either
  the prompt or the response
* print per-run cost

Cost: ~$0.06 per run, $0.12 total.

Run from the repo root::

    .venv/bin/python eval/smoke_memory_aware_eda.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path("/Users/wang/LLM_memory")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load ANTHROPIC_API_KEY from .env if not already set.
try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except Exception:  # noqa: BLE001
    pass


import pandas as pd  # noqa: E402

from mads.manager import MemoryManager  # noqa: E402
from mads.wrappers.memory_aware import MemoryAwareEDA  # noqa: E402


CONVERSION_BIZDEF = "signups / sessions, attributed to first-touch channel"
CONVERSION_SQLDEF = (
    "SELECT channel, "
    "CAST(SUM(signups) AS FLOAT) / NULLIF(SUM(sessions), 0) AS conversion "
    "FROM sessions_fact GROUP BY channel"
)

# Anthropic Sonnet 4.6 list pricing (approx): $3/MTok in, $15/MTok out.
_PRICE_IN = 3.0 / 1_000_000
_PRICE_OUT = 15.0 / 1_000_000


def _build_df() -> pd.DataFrame:
    """10-row toy frame with three distinct channels."""
    return pd.DataFrame(
        {
            "channel": [
                "organic", "paid", "referral",
                "organic", "paid", "referral",
                "organic", "paid", "referral",
                "organic",
            ],
            "sessions": [120, 80, 40, 130, 90, 35, 110, 95, 50, 125],
            "signups": [5, 12, 1, 7, 14, 2, 6, 15, 1, 8],
        }
    )


def _seed_metric(manager: MemoryManager, user_id: str, project_id: str) -> None:
    """Insert one active conversion metric definition."""
    store = manager._metric_store_  # internal but stable; the public path
    # is `add_metric_definition` tool — we go through the store directly
    # for unambiguous setup.
    definition, action = store.upsert(
        user_id=user_id,
        project_id=project_id,
        metric_name="conversion",
        business_definition=CONVERSION_BIZDEF,
        sql_definition=CONVERSION_SQLDEF,
        source="smoke_seed",
    )
    print(f"  seeded metric `{definition.metric_name}` (action={action})")


def _install_prompt_spy(captured: dict) -> None:
    """Patch ``make_eda`` to spy on the prompt before delegating to the real one.

    ``captured["prompts"]`` accumulates one entry per ADS invocation; this
    lets the same wrapper handle both smoke runs without clobbering the
    earlier prompt.
    """
    import mads.wrappers.memory_aware as mod

    real_make_eda = mod.make_eda
    captured.setdefault("prompts", [])

    def spying_make_eda(df, *, extra_tools=None, model=None):
        handle = real_make_eda(df, extra_tools=extra_tools, model=model)
        real_invoke = handle.invoke

        def spy_invoke(prompt):
            captured["prompts"].append(prompt)
            return real_invoke(prompt)

        handle.invoke = spy_invoke  # type: ignore[method-assign]
        return handle

    mod.make_eda = spying_make_eda  # type: ignore[assignment]


def _run_query(
    *,
    label: str,
    query: str,
    wrapper: MemoryAwareEDA,
    df: pd.DataFrame,
    user_id: str,
    project_id: str,
    captured: dict,
) -> tuple[bool, float, str]:
    """Run one query and return ``(passed, cost_usd, failure_reason)``."""
    print(f"\n{'=' * 60}")
    print(f"{label}: {query!r}")
    print("=" * 60)

    prompts_before = len(captured.get("prompts", []))
    ta = wrapper.run(
        query,
        df,
        user_id=user_id,
        project_id=project_id,
        dataset_name="channels",
        table_name="channels",
    )
    prompts_after = captured.get("prompts", [])
    prompt = prompts_after[-1] if len(prompts_after) > prompts_before else None

    print("\n--- ENRICHED PROMPT SENT TO ADS EDA ----------------------")
    print(prompt or "(prompt not captured)")
    print("----------------------------------------------------------\n")

    print("--- LLM RESPONSE -----------------------------------------")
    print(ta.agent_result.response or "(no response)")
    print("----------------------------------------------------------\n")

    print(f"memories_injected: {len(ta.memories_injected)}")
    for rm in ta.memories_injected:
        mid = (
            rm.memory.get("normalized_name")
            or rm.memory.get("id")
            or rm.memory.get("semantic_fingerprint")
        )
        print(f"  - {rm.category}:{mid} (score={rm.score:.2f}) reason={rm.reason}")
    print(f"warnings: {ta.warnings}")

    r = ta.agent_result
    cost_usd = r.prompt_tokens * _PRICE_IN + r.completion_tokens * _PRICE_OUT
    print(
        f"tokens: total={r.total_tokens} "
        f"(in={r.prompt_tokens}, out={r.completion_tokens}) "
        f"approx_cost=${cost_usd:.4f} (~{cost_usd * 100:.2f}¢)"
    )

    # Assertions
    print(f"\n[{label}] assertions:")
    fail_reason = ""
    if len(ta.memories_injected) < 1:
        fail_reason = (
            f"expected >=1 retrieved memory; got {len(ta.memories_injected)}"
        )
        print(f"  FAIL: {fail_reason}")
        return False, cost_usd, fail_reason
    print(f"  ok: len(memories_injected) == {len(ta.memories_injected)} >= 1")

    needle = "signups / sessions"
    in_prompt = needle in (prompt or "")
    in_response = needle in (ta.agent_result.response or "")
    print(f"  '{needle}' in prompt? {in_prompt}; in response? {in_response}")
    if not (in_prompt or in_response):
        fail_reason = (
            "expected the metric's business_definition fragment to appear "
            "in the prompt or the response"
        )
        print(f"  FAIL: {fail_reason}")
        return False, cost_usd, fail_reason
    print("  ok: bizdef fragment landed in prompt or response")

    return True, cost_usd, ""


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "error: ANTHROPIC_API_KEY not set (looked in env and .env). "
            "Smoke test requires Anthropic creds.",
            file=sys.stderr,
        )
        return 2

    user_id, project_id = "alice", "default"

    with tempfile.TemporaryDirectory() as td:
        db_path = str(Path(td) / "smoke.sqlite")
        manager = MemoryManager(
            user_id=user_id, project_id=project_id, db_path=db_path
        )
        try:
            print("Step 1: seed conversion metric")
            _seed_metric(manager, user_id, project_id)

            print("\nStep 2: build df")
            df = _build_df()
            print(f"  df shape={df.shape}, columns={list(df.columns)}")

            print("\nStep 3: install prompt spy + build wrapper")
            captured: dict = {}
            _install_prompt_spy(captured)
            wrapper = MemoryAwareEDA(manager)

            # ----- Run 1: bare token (regression check)
            pass1, cost1, fail1 = _run_query(
                label="Run 1 (bare token)",
                query="conversion",
                wrapper=wrapper,
                df=df,
                user_id=user_id,
                project_id=project_id,
                captured=captured,
            )

            # ----- Run 2: natural language (proves tokenized fix)
            pass2, cost2, fail2 = _run_query(
                label="Run 2 (natural language)",
                query="Show conversion by channel",
                wrapper=wrapper,
                df=df,
                user_id=user_id,
                project_id=project_id,
                captured=captured,
            )

            total_cost = cost1 + cost2
            print(f"\n{'=' * 60}")
            print("SMOKE SUMMARY")
            print("=" * 60)
            print(
                f"  Run 1 (bare token):        "
                f"{'PASS' if pass1 else 'FAIL'} (cost ~{cost1 * 100:.2f}¢)"
            )
            if not pass1:
                print(f"    failure: {fail1}")
            print(
                f"  Run 2 (natural language):  "
                f"{'PASS' if pass2 else 'FAIL'} (cost ~{cost2 * 100:.2f}¢)"
            )
            if not pass2:
                print(f"    failure: {fail2}")
            print(f"  Combined cost: ~${total_cost:.4f} (~{total_cost * 100:.2f}¢)")

            if pass1 and pass2:
                print("\nSMOKE PASSED")
                return 0
            print("\nSMOKE FAILED")
            return 1
        finally:
            manager.close()


if __name__ == "__main__":
    raise SystemExit(main())
