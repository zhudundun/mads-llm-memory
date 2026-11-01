"""Backward-compat-flavored re-export: ``mads.wrappers.wrapper``.

The canonical wrapper implementation remains at :mod:`mads.agents.wrapper`
because the existing test suite monkeypatches
``mads.agents.wrapper.make_loader`` / ``make_eda`` / ``make_viz`` and the
``wrap_*`` functions look those names up via their defining module's
globals. Physically relocating the file would break that monkeypatch path.

This module re-exports the public API under the proposed v1.1 layout
(``mads.wrappers.wrapper``) so new code can import from the conceptually
correct location:

    from mads.wrappers.wrapper import wrap_loader, wrap_eda, wrap_viz

When the wrapper grows additional surface (e.g. supervisor-aware wrappers
for LangGraph), the canonical location will migrate here and
``mads.agents.wrapper`` will become the shim.
"""

from mads.agents.wrapper import (  # noqa: F401
    WrappedEDA,
    WrappedLoader,
    WrappedViz,
    _classify_tool_calls,
    _DEFAULT_NUDGE,
    _extract_name,
    _MEMORY_ALL_NAMES,
    _MEMORY_READ_NAMES,
    _MEMORY_WRITE_NAMES,
    _normalize,
    wrap_eda,
    wrap_loader,
    wrap_viz,
)

__all__ = [
    "WrappedLoader",
    "WrappedEDA",
    "WrappedViz",
    "wrap_loader",
    "wrap_eda",
    "wrap_viz",
]
