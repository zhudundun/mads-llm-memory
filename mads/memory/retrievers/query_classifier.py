"""Cheap deterministic classifier for whether a query is a data-analysis question.

Used by :class:`mads.memory.retrievers.HybridRetriever` to skip memory retrieval
on clearly non-analytical queries (poems, jokes, off-topic chat) where injected
memories would only pollute the prompt. The gate lives at the hybrid layer so
all four sub-retrievers (Preference, Metric, Dataset, Episode) are skipped
uniformly; individual sub-retrievers do not call this helper.

Precedence (important):

1. Lowercase + normalize whitespace.
2. If any pattern in :data:`CREATIVE_PATTERNS` is a substring of the query →
   return ``False`` immediately. The denylist runs FIRST so a query like
   ``"write me a poem about churn"`` correctly returns ``False`` even though
   it contains an analytical token (``"churn"``).
3. Else if any pattern in :data:`ANALYTICAL_PATTERNS` is a substring → return
   ``True``.
4. Default → ``True`` (bias toward NOT breaking the data flow on ambiguous
   inputs). Empty string is treated as ambiguous and therefore ``True``.

Design bias: the gate's purpose is to catch obvious non-analytical queries,
not gatekeep every borderline case. False negatives (missing an off-topic
query) are an acceptable tradeoff to avoid false positives (incorrectly
suppressing memory for a real data question).

No LLM call. No external state. Pure stdlib substring scan.
"""

from __future__ import annotations


# Substrings that, when found anywhere in the lowercased query, mark it as a
# clearly NON-analytical request. The list is intentionally narrow and skewed
# toward creative-writing / generic-chat openers — patterns we have observed
# (or anticipate) producing memory-pollution failures.
CREATIVE_PATTERNS: frozenset[str] = frozenset(
    {
        # Creative-writing forms.
        "poem",
        "story",
        "joke",
        "haiku",
        "limerick",
        "rap",
        "song",
        "lyric",
        "rhyme",
        "fiction",
        "novel",
        "essay",
        "riddle",
        "fable",
        "ballad",
        # Common creative-writing imperatives.
        "write me a",
        "tell me a",
        "compose a",
        "compose me a",
        "make me a",
        "creative writing",
        # Generic chat / meta-questions that aren't data questions.
        "what's your name",
        "what is your name",
        "who are you",
        "hello",
        "hi there",
        "good morning",
        "good evening",
    }
)


# Substrings that, when found anywhere in the lowercased query, mark it as a
# data-analysis question. List is permissive on purpose — overlap with words
# that appear in user preferences / metric names (e.g. "churn", "revenue") is
# fine because the gate biases toward the data flow.
ANALYTICAL_PATTERNS: frozenset[str] = frozenset(
    {
        # Analytical verbs.
        "show",
        "compute",
        "calculate",
        "analyze",
        "analyse",
        "analysis",
        "plot",
        "chart",
        "graph",
        "visualize",
        "visualise",
        "describe",
        "summarize",
        "summarise",
        "summary",
        "explain",
        "list",
        "count",
        "find",
        "filter",
        "group",
        "groupby",
        "rank",
        "sort",
        "aggregate",
        "select",
        "join",
        "merge",
        "compare",
        "report",
        # Statistical / quantitative nouns.
        "distribution",
        "histogram",
        "mean",
        "median",
        "mode",
        "variance",
        "std",
        "stddev",
        "sum",
        "average",
        "rate",
        "ratio",
        "percentage",
        "percent",
        "correlation",
        "regression",
        "trend",
        "outlier",
        "quantile",
        "percentile",
        # Data-domain nouns. Overlap with user-preference / metric vocabulary
        # is intentional — the gate is permissive.
        "data",
        "dataset",
        "table",
        "column",
        "row",
        "record",
        "metric",
        "kpi",
        "feature",
        "target",
        "label",
        "churn",
        "revenue",
        "conversion",
        "retention",
        "customer",
        "user",
        "segment",
        "cohort",
        # Loading / EDA vocabulary.
        "load",
        "read",
        "csv",
        "parquet",
        "json",
        "dataframe",
        " df ",
        "eda",
        "preview",
        "head",
        "tail",
        "info",
        "missing",
        "null",
        "nan",
        "dtype",
        "schema",
    }
)


def is_data_question(query: str) -> bool:
    """Return ``True`` if ``query`` looks like a data-analysis question.

    Precedence (see module docstring for rationale):

    1. ``CREATIVE_PATTERNS`` substring → ``False`` (denylist wins).
    2. ``ANALYTICAL_PATTERNS`` substring → ``True``.
    3. Default → ``True``.

    Empty / whitespace-only ``query`` returns ``True`` (consistent with the
    default — biased toward the data flow).
    """
    if not query or not query.strip():
        return True

    # Normalize: lowercase + collapse whitespace. We pad with spaces on both
    # sides so single-token patterns like ``" df "`` can match at word
    # boundaries without a regex.
    q = " " + " ".join(query.lower().split()) + " "

    # Step 1: explicit creative — short-circuit False. Denylist wins over
    # analytical tokens so "write me a poem about churn" returns False.
    for pat in CREATIVE_PATTERNS:
        if pat in q:
            return False

    # Step 2: any analytical token → True.
    for pat in ANALYTICAL_PATTERNS:
        if pat in q:
            return True

    # Step 3: default — bias toward the data flow.
    return True


__all__ = ["is_data_question", "CREATIVE_PATTERNS", "ANALYTICAL_PATTERNS"]
