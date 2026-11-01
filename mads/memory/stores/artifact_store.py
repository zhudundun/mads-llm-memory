"""artifact_store — v1.1 architectural scaffold.

Design intent: URI-addressable non-tabular artifact pointers. Memory rows must
not inline dataframes or chart bytes (CLAUDE.md "DataFrames travel by
reference, not value"). The artifact store mints a stable URI for each
external artifact (a parquet file produced by the loader, a PNG produced by
the viz agent, a pickled feature pipeline) and records the
``(uri, content_hash, created_at, namespace)`` tuple so memory rows can
reference them by URI alone.

v1.1 trigger: when categories 6 (recipe) and 9 (experiment_record) land —
both will produce real artifacts (recipe code, model files, charts) that
need addressable storage.

Status: STUB — interface only, no behavior. See docs/architecture.md.
"""
from __future__ import annotations

from typing import Any


class ArtifactStore:
    """URI-addressable store for non-tabular artifacts (charts, parquet, models)."""

    def put(self, payload: bytes, *, kind: str, namespace: tuple[str, str]) -> str:
        """Persist ``payload`` and return a stable URI.

        Inputs: the raw bytes, a kind tag (``"png"`` / ``"parquet"`` /
        ``"pickle"``), and the tenancy namespace. Output: a URI string
        of the form ``mads-artifact://<namespace>/<content_hash>.<kind>``.
        """
        raise NotImplementedError(
            "artifact_store is a v1.1 scaffold; not yet implemented."
        )

    def get(self, uri: str) -> bytes:
        """Return the bytes previously written under ``uri``."""
        raise NotImplementedError(
            "artifact_store is a v1.1 scaffold; not yet implemented."
        )


__all__ = ["ArtifactStore"]
