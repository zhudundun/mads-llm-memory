"""mads.dataset: dataset-side utilities (fingerprinting, profiling, schema summary).

Ships :func:`compute_fingerprint` (alias for :func:`semantic_fingerprint`),
:func:`exact_fingerprint`, :func:`semantic_fingerprint`, plus the
:class:`DatasetContext` profiler. The ``schema_summary`` module is a v1.1
scaffold.
"""

from mads.dataset.fingerprint import (
    compute_fingerprint,
    exact_fingerprint,
    semantic_fingerprint,
)
from mads.dataset.profiler import DatasetContext, profile

__all__ = [
    "compute_fingerprint",
    "exact_fingerprint",
    "semantic_fingerprint",
    "DatasetContext",
    "profile",
]
