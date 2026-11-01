"""mads.memory: long-term-memory categories for the mads agent."""

from mads.memory.user_preference import PreferenceStore
from mads.memory.preference_tools import make_preference_tools
from mads.memory.dataset_schema import DatasetSchemaStore, compute_fingerprint
from mads.memory.dataset_tools import make_dataset_tools
from mads.memory.metric_definition import (
    MetricDefinition,
    MetricDefinitionStore,
)
from mads.memory.metric_tools import MetricToolFactory, make_metric_factory
from mads.memory.data_quality import (
    DataQualityIssue,
    DataQualityStore,
)
from mads.memory.data_quality_tools import (
    DataQualityToolFactory,
    make_data_quality_factory,
)
from mads.memory.embedding import (
    Embedder,
    HashEmbedder,
    SentenceTransformerEmbedder,
)
from mads.memory.analysis_episode import (
    AnalysisEpisode,
    AnalysisEpisodeStore,
)
from mads.memory.analysis_episode_tools import (
    AnalysisEpisodeToolFactory,
    make_analysis_episode_factory,
)

def __getattr__(name):
    # Lazy re-export of MemoryManager to avoid circular import (manager imports
    # from mads.memory). Per V1_DESIGN_BRIEF §1.
    if name == "MemoryManager":
        from mads.manager import MemoryManager
        return MemoryManager
    raise AttributeError(f"module 'mads.memory' has no attribute {name!r}")


__all__ = [
    "MemoryManager",
    "PreferenceStore",
    "make_preference_tools",
    "DatasetSchemaStore",
    "make_dataset_tools",
    "compute_fingerprint",
    "MetricDefinition",
    "MetricDefinitionStore",
    "MetricToolFactory",
    "make_metric_factory",
    "DataQualityIssue",
    "DataQualityStore",
    "DataQualityToolFactory",
    "make_data_quality_factory",
    "Embedder",
    "HashEmbedder",
    "SentenceTransformerEmbedder",
    "AnalysisEpisode",
    "AnalysisEpisodeStore",
    "AnalysisEpisodeToolFactory",
    "make_analysis_episode_factory",
]
