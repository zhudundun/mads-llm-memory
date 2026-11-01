"""mads.memory.retrievers: specialized read-side retrievers (v1.1+).

Each retriever wraps one (or more) store ``find``/``list``/``lookup``
method and produces a uniform :class:`RetrievedMemory` row with an
explicit score and reason. The :class:`HybridRetriever` fans out to all
four and merges the results so the caller has a single ranked view
across categories.

This subpackage is standalone-API-only — it is not wired into the
LangChain tool factories or the CLI. Callers construct retrievers
directly with their stores and call ``retrieve(...)``.
"""

from mads.memory.retrievers.dataset_retriever import DatasetRetriever
from mads.memory.retrievers.episode_retriever import EpisodeRetriever
from mads.memory.retrievers.hybrid_retriever import HybridRetriever
from mads.memory.retrievers.metric_retriever import MetricRetriever
from mads.memory.retrievers.preference_retriever import PreferenceRetriever
from mads.memory.retrievers.types import Category, Retriever, RetrievedMemory

__all__ = [
    "RetrievedMemory",
    "Category",
    "Retriever",
    "MetricRetriever",
    "DatasetRetriever",
    "EpisodeRetriever",
    "PreferenceRetriever",
    "HybridRetriever",
]
