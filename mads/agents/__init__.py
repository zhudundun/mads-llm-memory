"""Memory-aware wrappers around the ADS handles from :mod:`mads.adapters.ads`."""

from mads.agents.types import AgentResult, TurnArtifacts
from mads.agents.wrapper import (
    WrappedEDA,
    WrappedLoader,
    WrappedViz,
    wrap_eda,
    wrap_loader,
    wrap_viz,
)

__all__ = [
    "AgentResult",
    "TurnArtifacts",
    "WrappedEDA",
    "WrappedLoader",
    "WrappedViz",
    "wrap_eda",
    "wrap_loader",
    "wrap_viz",
]
