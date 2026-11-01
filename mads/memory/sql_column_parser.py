"""Best-effort column-identifier extraction from a SQL fragment.

Used as a fallback in :func:`MetricDefinitionStore.upsert` when the caller
omits ``required_columns`` but does provide a ``sql_definition``. The parser
is deliberately simplistic: it tokenizes on word boundaries, rejects a
hardcoded set of SQL keywords / aggregate function names / boolean literals,
strips out string literals before tokenizing, and rejects pure-numeric
tokens. The output is deduplicated and preserves order of first occurrence.

False-positive caveat: the parser cannot distinguish a column identifier
from a table identifier (e.g. ``"SELECT COUNT(*) FROM churn_data WHERE
Churn = 'Yes'"`` yields ``["churn_data", "Churn"]``). The downstream
applicability check is ``required ⊆ available_columns``, so an extra
"column" that doesn't exist in the bound df marks the metric INAPPLICABLE
— this is the conservative direction (false-negative for applicability,
not a false-positive). LLM callers can always pass ``required_columns``
explicitly to override the parser.
"""

from __future__ import annotations

import re

# SQL keywords / operators / function names we always strip. Matched
# case-insensitively. Picked to cover the typical metric-definition surface
# (filter predicates, simple aggregates, GROUP/ORDER clauses) without
# pulling in a full SQL grammar.
_KEYWORDS: frozenset[str] = frozenset(
    {
        "select",
        "from",
        "where",
        "and",
        "or",
        "not",
        "is",
        "null",
        "count",
        "sum",
        "avg",
        "min",
        "max",
        "group",
        "by",
        "order",
        "asc",
        "desc",
        "as",
        "in",
        "between",
        "like",
        "join",
        "on",
        "left",
        "right",
        "inner",
        "outer",
        "distinct",
        "having",
        "limit",
        "case",
        "when",
        "then",
        "else",
        "end",
        "true",
        "false",
    }
)

# Strip single-quoted string literals first so the tokens inside them don't
# leak into the identifier set. Multi-line, non-greedy.
_STRING_LITERAL_RE = re.compile(r"'[^']*'")

# Strip SQL line comments (``-- ...`` to end-of-line) and block comments
# (``/* ... */``). Order: block first, then line, then literal strip.
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"--[^\n]*")

# Identifier tokenizer: word-boundary-delimited alphanumeric runs (plus
# underscore). Excludes anything that starts with a digit because pure
# numbers like ``12`` should never count as a column.
_TOKEN_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


def extract_columns_from_sql(sql: str) -> list[str]:
    """Return plausible column identifiers from a SQL fragment.

    Best-effort:

    * Strips block + line comments, then string literals.
    * Tokenizes on word boundaries (identifier-shaped runs only).
    * Rejects SQL keywords (case-insensitive), aggregate function names,
      and boolean literals.
    * Rejects pure-numeric tokens (handled implicitly by the tokenizer's
      identifier-shaped pattern, but documented here for callers).
    * Deduplicates while preserving order of first occurrence.

    Examples:

    >>> extract_columns_from_sql("tenure > 12 AND MonthlyCharges > 50")
    ['tenure', 'MonthlyCharges']
    >>> extract_columns_from_sql("SELECT COUNT(*) FROM churn_data WHERE Churn = 'Yes'")
    ['churn_data', 'Churn']
    >>> extract_columns_from_sql("tenure>12 and PaymentMethod='Credit card'")
    ['tenure', 'PaymentMethod']
    >>> extract_columns_from_sql("")
    []
    >>> extract_columns_from_sql("   ")
    []
    """
    if not isinstance(sql, str) or not sql.strip():
        return []

    cleaned = _BLOCK_COMMENT_RE.sub(" ", sql)
    cleaned = _LINE_COMMENT_RE.sub(" ", cleaned)
    cleaned = _STRING_LITERAL_RE.sub(" ", cleaned)

    seen: set[str] = set()
    out: list[str] = []
    for tok in _TOKEN_RE.findall(cleaned):
        # Case-insensitive keyword reject.
        if tok.lower() in _KEYWORDS:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


__all__ = ["extract_columns_from_sql"]
