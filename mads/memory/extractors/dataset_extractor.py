"""dataset_extractor — v1.1 architectural scaffold.

Design intent: mines dataset- and column-level facts from loader and EDA
transcripts. Looks for the LLM saying things like "the primary key is
``customer_id``", "``signup_date`` is in UTC", "this dataset is at the
session grain" and proposes ``dataset_memory`` / ``column_memory`` updates.
Pairs naturally with :mod:`mads.dataset.profiler` (profiler produces
structural facts; this extractor mines semantic ones).

v1.1 trigger: after observed gap-closure rate <30% in the with-memory eval
specifically for dataset/column facts (which the LLM has shown a tendency to
state inline in answers rather than recording via ``register_dataset`` /
``register_column``).

Status: STUB — interface only, no behavior. See docs/architecture.md.
"""
from __future__ import annotations

from typing import Any


class DatasetExtractor:
    """Mines candidate dataset/column memory updates from agent transcripts."""

    def extract(self, transcript: str, *, fingerprint: str) -> list[dict[str, Any]]:
        """Return candidate dataset_memory / column_memory payloads.

        Inputs: the agent transcript and the fingerprint of the df the agent
        is operating on. Output: candidate payloads tagged by which table
        they belong to (``dataset`` vs ``column``).
        """
        raise NotImplementedError(
            "dataset_extractor is a v1.1 scaffold; not yet implemented."
        )


__all__ = ["DatasetExtractor"]
