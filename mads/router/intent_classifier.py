"""Heuristic request router.

Classifies a question into one of ``loader``, ``eda``, or ``viz``. The router
is pure: no I/O, no LLM call, deterministic. Viz patterns are checked BEFORE
loader patterns because phrases like "plot the columns of x.csv" mention
"columns" (a loader signal) but the intent is clearly viz.

See V1_DESIGN_BRIEF §5 for rationale.
"""

from __future__ import annotations

import re
from typing import Literal

AgentName = Literal["loader", "eda", "viz"]


_LOADER_PATTERNS = [
    r"\bload\b",
    r"\bread\s+csv\b",
    r"\bimport\b.*\bdata\b",
    r"\bingest\b",
    r"\bpreview\b",
    r"\bfirst\s+\d+\s+rows?\b",
    r"\bdtype\b",
    r"\bcolumn\s+list\b",
    r"\blist\s+(?:the\s+)?columns\b",
]

_VIZ_PATTERNS = [
    r"\bplot\b",
    r"\bchart\b",
    r"\bvisuali[sz]e\b",
    r"\bscatter\b",
    r"\bhistogram\b",
    r"\bhist\b",
    r"\bbar\s+chart\b",
    r"\bbox\s*plot\b",
    r"\bheatmap\b",
    r"\bdraw\b",
    r"\bgraph\b",
]


def route(question: str) -> AgentName:
    """Classify ``question`` to one of loader/eda/viz. Heuristic only.

    EDA is the residual category, so any question that does not match a viz
    or loader pattern routes there.
    """
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    q = question.lower()
    for p in _VIZ_PATTERNS:
        if re.search(p, q):
            return "viz"
    for p in _LOADER_PATTERNS:
        if re.search(p, q):
            return "loader"
    return "eda"


__all__ = ["route", "AgentName"]
