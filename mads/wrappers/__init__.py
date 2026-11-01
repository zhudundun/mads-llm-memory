"""mads.wrappers: memory-aware wrappers over ADS agent handles.

Public surface: ``wrap_loader``, ``wrap_eda``, ``wrap_viz`` (and the
``Wrapped*`` classes they return). The single ``wrapper.py`` module holds
all three to keep them in lockstep — they share the nudge text and the
tool-call classification logic.
"""

from mads.wrappers.memory_aware import MemoryAwareEDA
from mads.wrappers.wrapper import (
    WrappedEDA,
    WrappedLoader,
    WrappedViz,
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
    "MemoryAwareEDA",
]
