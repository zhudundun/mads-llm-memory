"""Unit tests for the per-scenario EDA observables in eval/run_with_memory.py.

Each EDA scenario has a tight programmatic observable that returns
``(passed: bool, reason: str)``. We test PASS and FAIL cases per scenario
to lock down the rubric semantics — the eval headline numbers depend on
these checks being stable.

These tests do NOT call any LLM; they pass synthetic ``response`` strings
to the observable functions and assert the verdict.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make eval/ importable.
REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import run_with_memory as rwm  # noqa: E402


# -----------------------------------------------------------------------------
# eda_01_contract_typos — normalize typos before aggregation.
# -----------------------------------------------------------------------------
def test_eda_01_pass_with_explicit_normalization_and_3_canonical():
    resp = (
        "I normalized the Contract column with .str.lower().str.strip() before "
        "any aggregation. The 3 canonical values are: Month-to-month (3875), "
        "One year (1473), Two year (1695)."
    )
    ok, reason = rwm._check_eda_01(resp)
    assert ok, reason
    assert "normalization" in reason.lower() or "canonical" in reason.lower()


def test_eda_01_fail_with_canonical_only_no_normalization():
    # The CSV has no real typos, so simply showing 3 canonical values is the
    # default behaviour — it does NOT discriminate memory-on vs memory-off.
    # Without explicit normalization vocab, the observable must fail.
    resp = (
        "Customer counts by Contract type:\n"
        "- Month-to-month: 3875\n"
        "- One year: 1473\n"
        "- Two year: 1695\n"
    )
    ok, reason = rwm._check_eda_01(resp)
    assert not ok, reason
    assert "no normalization vocab" in reason


def test_eda_01_pass_with_lower_str_call():
    resp = (
        "I applied .str.lower() before grouping. Counts: Month-to-month 3875, "
        "One year 1473, Two year 1695."
    )
    ok, reason = rwm._check_eda_01(resp)
    assert ok, reason


def test_eda_01_fail_when_space_variant_and_no_normalization():
    # Raw groupby with no normalization, and a "Month to month" space variant.
    resp = (
        "Contract counts:\n"
        "- Month-to-month: 3500\n"
        "- Month to month: 375\n"
        "- One year: 1473\n"
        "- Two year: 1695\n"
    )
    ok, reason = rwm._check_eda_01(resp)
    # space variant present AND no normalization vocab + 3 canonicals BUT
    # with extra space variant → fails the strict shows_three_canonical branch.
    assert not ok, reason


def test_eda_01_fail_when_no_normalization_and_missing_canonicals():
    resp = "Here are some unrelated insights about customer demographics."
    ok, _ = rwm._check_eda_01(resp)
    assert not ok


# -----------------------------------------------------------------------------
# eda_02_churn_definition — pct with one decimal.
# -----------------------------------------------------------------------------
def test_eda_02_pass_exact_truth_one_decimal():
    resp = "The overall churn rate is **26.5%**."
    ok, reason = rwm._check_eda_02(resp)
    assert ok, reason
    assert "26.5" in reason


def test_eda_02_pass_within_tolerance_one_decimal():
    # Within 0.5% of ground truth (26.5%), one decimal: 26.7% is fine.
    resp = "Churn rate: 26.7%"
    ok, reason = rwm._check_eda_02(resp)
    assert ok, reason


def test_eda_02_fail_no_decimal():
    resp = "Churn rate is 27%."
    ok, reason = rwm._check_eda_02(resp)
    assert not ok
    assert "not 1-decimal" in reason or "1-decimal" in reason


def test_eda_02_fail_two_decimals():
    resp = "Churn rate is 26.54%."
    ok, reason = rwm._check_eda_02(resp)
    assert not ok


def test_eda_02_fail_no_percentage():
    resp = "Churn rate calculation is pending."
    ok, _ = rwm._check_eda_02(resp)
    assert not ok


# -----------------------------------------------------------------------------
# eda_03_grouping_preference — Contract × tenure buckets.
# -----------------------------------------------------------------------------
def test_eda_03_pass_all_four_buckets_and_contract():
    resp = (
        "Churn by Contract × tenure bucket:\n"
        "| Contract | 0-12 mo | 13-24 mo | 25-48 mo | 49+ mo |\n"
        "| Month-to-month | 50% | 40% | 30% | 20% |\n"
    )
    ok, reason = rwm._check_eda_03(resp)
    assert ok, reason
    assert "4/4" in reason


def test_eda_03_pass_three_of_four_buckets():
    resp = (
        "Slicing by Contract and tenure: 0-12 months show high churn, "
        "13-24 moderate, 25-48 low. (49+ omitted for brevity.)"
    )
    ok, reason = rwm._check_eda_03(resp)
    assert ok, reason


def test_eda_03_fail_only_contract_no_buckets():
    resp = (
        "Here is churn by Contract type only:\n"
        "- Month-to-month: 50%\n"
        "- One year: 12%\n"
        "- Two year: 3%\n"
    )
    ok, reason = rwm._check_eda_03(resp)
    assert not ok
    assert "0/4" in reason or "1/4" in reason


def test_eda_03_fail_no_contract():
    resp = "Tenure buckets 0-12, 13-24, 25-48, 49+ are useful segments."
    ok, reason = rwm._check_eda_03(resp)
    assert not ok
    assert "contract=False" in reason


# -----------------------------------------------------------------------------
# eda_04_exec_summary_first — 3 bullets then technical details.
# -----------------------------------------------------------------------------
def test_eda_04_pass_three_bullets_then_tech_section():
    resp = (
        "## Executive Summary\n"
        "- Churn is concentrated in month-to-month + fiber.\n"
        "- 26.5% overall churn rate.\n"
        "- TotalCharges needs dtype fix.\n"
        "\n## Technical Details\n"
        "Dtypes: ... Missingness: ... Correlations: ..."
    )
    ok, reason = rwm._check_eda_04(resp)
    assert ok, reason


def test_eda_04_fail_four_bullets():
    resp = (
        "## Executive Summary\n"
        "- one\n"
        "- two\n"
        "- three\n"
        "- four\n"
        "\n## Technical Details\nfoo"
    )
    ok, reason = rwm._check_eda_04(resp)
    assert not ok
    assert "count=4" in reason or "bullets=4" in reason


def test_eda_04_fail_no_exec_header():
    resp = (
        "Here is some narrative without any executive summary header.\n"
        "- one\n- two\n- three\n"
    )
    ok, reason = rwm._check_eda_04(resp)
    assert not ok
    assert "executive-summary" in reason


def test_eda_04_fail_three_bullets_no_tech_section():
    resp = (
        "# Summary\n"
        "- a\n- b\n- c\n"
        "\nThat's all.\n"
    )
    ok, reason = rwm._check_eda_04(resp)
    assert not ok


# ---- v1.2 rubric-widening: tolerate decorative glyphs in exec-summary header.
def test_eda04_plain_executive_summary_header_passes():
    """Baseline regression: the existing plain `## Executive Summary` header
    continues to match after the regex widen. Locks down that the new
    `[^\\w\\s]*\\s*` clause is fully optional and doesn't break the
    no-glyph case."""
    resp = (
        "## Executive Summary\n"
        "- a\n- b\n- c\n"
        "\n## Technical Details\n"
        "Dtypes and correlations follow."
    )
    ok, reason = rwm._check_eda_04(resp)
    assert ok, reason
    assert "3 bullets" in reason or "technical-details" in reason


def test_eda04_emoji_decorated_header_passes():
    """The eda_04 fix case: the LLM's actual session-B response opens with
    `## 🔍 Executive Summary` followed by 3 bullets and `## 🛠 Technical
    Details`. Previously the rubric regex required a word character
    immediately after the `##` so this returned False; the widened regex
    must now match."""
    resp = (
        "## 🔍 Executive Summary\n"
        "- a\n- b\n- c\n"
        "\n## 🛠 Technical Details\n"
        "Dtypes and missingness follow."
    )
    ok, reason = rwm._check_eda_04(resp)
    assert ok, reason
    assert "3 bullets" in reason or "technical-details" in reason


def test_eda04_no_header_fails():
    """Lock down that the widened regex didn't accidentally over-loosen:
    a body paragraph mentioning "executive summary" mid-text without any
    heading marker (no `#`, no newline-leading position) must still fail."""
    resp = (
        "In the following paragraphs I provide an executive summary of "
        "what we found, including three key takeaways and the technical "
        "details below. The dataset has 7,043 rows."
    )
    ok, reason = rwm._check_eda_04(resp)
    assert not ok, reason
    assert "executive-summary" in reason


def test_eda04_summary_after_other_section_fails():
    """A response whose FIRST section is something else (Dataset Overview)
    and only later has a `## Summary` must still fail — the rubric's
    `head = resp[:600]` window is intentionally narrow, and a long
    leading section pushes the summary header past it. Confirms the
    "first" constraint survives the widen."""
    leading = (
        "## Dataset Overview\n"
        "This dataset contains telecom customer records spanning multiple\n"
        "service categories. It has demographics columns including gender,\n"
        "SeniorCitizen, Partner, and Dependents; account columns including\n"
        "tenure, Contract, PaperlessBilling, and PaymentMethod; phone and\n"
        "internet service columns covering MultipleLines, InternetService,\n"
        "OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,\n"
        "StreamingTV, and StreamingMovies; billing columns MonthlyCharges\n"
        "and TotalCharges; and the binary Churn target. There are 7,043\n"
        "rows total, with no missing values across any of the 21 fields.\n"
        "The dtypes are mostly object except for tenure (int), MonthlyCharges\n"
        "(float), and SeniorCitizen (int). TotalCharges is stored as object\n"
        "which is a known data-quality issue requiring a cast.\n"
    )
    # Pad the leading section so the Summary header is beyond the 600-char
    # head window the rubric inspects.
    assert len(leading) > 600, "test setup: leading section must exceed head window"
    resp = (
        leading
        + "## Summary\n"
        + "- a\n- b\n- c\n"
        + "\n## Technical Details\nfoo"
    )
    ok, reason = rwm._check_eda_04(resp)
    assert not ok, reason
    assert "executive-summary" in reason


# -----------------------------------------------------------------------------
# eda_05_followup_finding — build on prior fiber-vs-DSL finding.
# -----------------------------------------------------------------------------
def test_eda_05_pass_fiber_dsl_and_extension():
    resp = (
        "Building on last week's finding that fiber-optic churns at ~2x DSL: "
        "we should drill deeper into fiber tenure and run survival analysis "
        "by InternetService."
    )
    ok, reason = rwm._check_eda_05(resp)
    assert ok, reason


def test_eda_05_pass_fiber_and_prior_vocab_and_extension():
    resp = (
        "Following up on the prior fiber finding, I recommend we investigate "
        "cohort retention curves and run an interaction test."
    )
    ok, reason = rwm._check_eda_05(resp)
    assert ok, reason


def test_eda_05_fail_no_fiber_reference():
    resp = (
        "I suggest you investigate the Contract dimension further and run "
        "a hypothesis test on payment method."
    )
    ok, reason = rwm._check_eda_05(resp)
    assert not ok
    # Could be either "no fiber/DSL reference" or "proposes investigation but
    # doesn't reference fiber/DSL finding" depending on extension vocab hits.
    assert "fiber" in reason.lower() or "dsl" in reason.lower()


def test_eda_05_fail_fiber_mention_no_prior_vocab():
    # Mentioning fiber/DSL naturally falls out of the correlation funnel
    # without any memory; without prior-analysis vocab this should NOT count
    # as gap-closed.
    resp = "Fiber-optic customers churn more than DSL."
    ok, reason = rwm._check_eda_05(resp)
    assert not ok
    assert "no prior-analysis vocab" in reason


def test_eda_05_fail_fiber_with_prior_but_no_extension():
    resp = "As discussed earlier, fiber-optic customers churn more than DSL."
    ok, reason = rwm._check_eda_05(resp)
    assert not ok
    assert "no extension" in reason


# ---- v1.1-rubric-widening: legitimate "building on prior" phrasings.
def test_eda_05_pass_extending_the_earlier_finding():
    """PASS: 'extending the earlier finding that fiber...' phrasing."""
    resp = (
        "Extending the earlier finding that fiber-optic customers churn at "
        "roughly 2x DSL, I propose drilling into tenure cohorts within fiber "
        "to investigate why retention drops off."
    )
    ok, reason = rwm._check_eda_05(resp)
    assert ok, reason


def test_eda_05_pass_since_we_previously_saw():
    """PASS: 'since we previously saw fiber...' phrasing."""
    resp = (
        "Since we previously saw fiber churn at about 2x the DSL rate, the "
        "next step is a survival analysis on InternetService cohorts and a "
        "deeper hypothesis test on monthly charges within fiber."
    )
    ok, reason = rwm._check_eda_05(resp)
    assert ok, reason


def test_eda_05_pass_following_up_on_the_previous_investigation():
    """PASS: 'following up on the previous investigation...' phrasing."""
    resp = (
        "Following up on the previous investigation, fiber churn warrants a "
        "deeper drill by tenure to test the interaction with contract type."
    )
    ok, reason = rwm._check_eda_05(resp)
    assert ok, reason


def test_eda_05_pass_we_previously_found():
    """PASS: 'we previously found fiber churns more' phrasing."""
    resp = (
        "We previously found that fiber-optic customers churn more than DSL. "
        "I'd like to drill deeper into the fiber cohort by tenure and run a "
        "hypothesis test on contract length."
    )
    ok, reason = rwm._check_eda_05(resp)
    assert ok, reason


def test_eda_05_fail_widened_regression_no_prior_no_fiber():
    """Regression: a response with NO prior-building vocab AND no fiber/DSL
    references must still return False. This locks down that the widened
    vocab doesn't accidentally pass an empty-store baseline response.
    """
    resp = (
        "I suggest you look at the Contract dimension and run a hypothesis "
        "test on payment method. There are interesting interactions worth "
        "investigating further."
    )
    ok, reason = rwm._check_eda_05(resp)
    assert not ok, reason
    assert "no prior-analysis vocab" in reason


def test_eda_05_pass_builds_on_known_issue():
    """PASS: 'builds on known issue' phrasing — the LLM frequently writes
    'builds on known finding/issue' when referring to memory-injected
    context."""
    resp = (
        "Drill into the fiber-optic churn signal (highest priority — builds "
        "on known issue). Investigate contract type within fiber-optic "
        "customers and run a deeper drill on tenure."
    )
    ok, reason = rwm._check_eda_05(resp)
    assert ok, reason


def test_eda_05_pass_extends_known_finding():
    """PASS: 'extends known finding' phrasing."""
    resp = (
        "Fiber-optic deep dive (contract x tenure) extends known finding. "
        "Let's investigate the interaction between InternetService and "
        "Contract length to see why fiber churn dominates DSL."
    )
    ok, reason = rwm._check_eda_05(resp)
    assert ok, reason


def test_eda_05_pass_youve_confirmed():
    """PASS: 'you've confirmed fiber...' phrasing."""
    resp = (
        "You've confirmed fiber-optic customers churn at ~2x the DSL rate, "
        "but why is still open. Investigate tenure as a moderator and run a "
        "deeper hypothesis test on contract type within fiber."
    )
    ok, reason = rwm._check_eda_05(resp)
    assert ok, reason


# -----------------------------------------------------------------------------
# Dispatcher sanity
# -----------------------------------------------------------------------------
def test_eda_observable_dispatch_to_eda_02():
    ok, reason = rwm._eda_observable("eda_02_churn_definition", "Churn: 26.5%", {})
    assert ok, reason


def test_eda_observable_unknown_id_returns_false():
    ok, reason = rwm._eda_observable("eda_99_unknown", "blah", {})
    assert not ok
    assert "no observable" in reason
