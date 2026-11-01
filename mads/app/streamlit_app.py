"""streamlit_app — v1.1 architectural scaffold.

Design intent: a Streamlit-based web UI for ``mads`` that lets users browse
their memory categories, inspect dataset/column/metric history, review
candidate extractions before persisting, and run ``mads ask`` interactively
with a live transcript view. Deferred because (1) the ADS "Pipeline Studio"
upstream doesn't exist yet, so there is no canonical visual idiom to follow,
and (2) the CLI is the v1 surface and we want one well-tested entry point
before adding a second.

v1.1 trigger: when the CLI demonstrably constrains real users — e.g. the
eval shows users need to see candidate-extraction diffs to confirm them,
which is painful in a terminal.

Status: STUB — interface only, no behavior. See docs/architecture.md.
"""
from __future__ import annotations


def main() -> None:
    """Launch the Streamlit app.

    Intended invocation (v1.1): ``streamlit run mads/app/streamlit_app.py``.
    Until then, calling this function raises :class:`NotImplementedError`.
    """
    raise NotImplementedError(
        "streamlit_app is a v1.1 scaffold; not yet implemented."
    )


__all__ = ["main"]
