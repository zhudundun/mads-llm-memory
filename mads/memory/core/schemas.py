"""schemas — v1.1 architectural scaffold.

Design intent: centralize the pydantic models that currently live inline in
each category file (``MetricDefinition`` in ``metric_definition.py``,
``AnalysisEpisode`` in ``analysis_episode.py``, ``DataQualityIssue`` in
``data_quality.py``, the implicit dataset/column shapes in
``dataset_schema.py``, the preference dict in ``user_preference.py``). Pulling
them into one module makes cross-category referencing (e.g. a recipe that
binds to a specific ``dataset_fingerprint`` plus a specific
``metric_definition.id``) typed end-to-end, and gives the validator layer a
single import surface.

v1.1 trigger: when adding category 6 (workflow_recipe) or category 9
(experiment_record), both of which carry typed references into cats 2 and 3.
Doing it before then would be premature centralization.

Status: STUB — interface only, no behavior. See docs/architecture.md.
"""
from __future__ import annotations


class CoreSchemas:
    """Marker class for the centralized schemas namespace."""

    def __init__(self) -> None:
        raise NotImplementedError(
            "schemas is a v1.1 scaffold; not yet implemented."
        )


__all__ = ["CoreSchemas"]
