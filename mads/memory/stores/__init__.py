"""mads.memory.stores: store-layer scaffolding.

Today, each memory category ships its own concrete store class in its
category file. This subpackage re-exports those five concrete stores for
discoverability and adds three v1.1 scaffolds for shared base classes
(SQLite, sqlite-vec, artifact pointers).
"""

from mads.memory.analysis_episode import AnalysisEpisodeStore
from mads.memory.data_quality import DataQualityStore
from mads.memory.dataset_schema import DatasetSchemaStore
from mads.memory.metric_definition import MetricDefinitionStore
from mads.memory.user_preference import PreferenceStore

__all__ = [
    "AnalysisEpisodeStore",
    "DataQualityStore",
    "DatasetSchemaStore",
    "MetricDefinitionStore",
    "PreferenceStore",
]
