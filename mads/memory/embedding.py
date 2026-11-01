"""Embedder protocol + two implementations.

This module is intentionally generic. It is introduced by the
``analysis_episode`` memory category but lives at the top level of
``mads.memory`` because future categories (e.g. ``workflow_recipe``,
``analysis_conclusion``-style search elsewhere) may reuse it.

Two concrete embedders are provided:

* :class:`SentenceTransformerEmbedder` — production-grade embedder backed by
  ``all-MiniLM-L6-v2`` (384-dim, runs on CPU, ~80 MB model download). The
  model is loaded lazily on first call so that simply importing the class
  doesn't trigger the download.

* :class:`HashEmbedder` — deterministic, non-semantic embedder used by the
  unit test suite. Same input → identical vector; "revenue drop" and
  "revenue rose" share the "revenue" token bucket, so they end up closer
  to each other than to "churn model". L2-normalized so cosine distance is
  well-defined.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Minimal interface every embedder must satisfy."""

    @property
    def dim(self) -> int: ...

    def embed(self, text: str) -> list[float]: ...


# --------------------------------------------------------------------- prod


class SentenceTransformerEmbedder:
    """Production embedder using ``all-MiniLM-L6-v2`` (384-dim, CPU).

    The model is loaded lazily on the first call to :meth:`embed`. The first
    instantiation in a fresh environment will trigger an ~80 MB HuggingFace
    download; subsequent runs reuse the cache under
    ``~/.cache/huggingface/hub``.
    """

    DIM = 384

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None  # loaded lazily

    @property
    def dim(self) -> int:
        return self.DIM

    def _ensure_loaded(self) -> None:
        if self._model is None:
            # Local import — avoid paying the import cost (torch, transformers)
            # for callers that only need the protocol or the HashEmbedder.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)

    def embed(self, text: str) -> list[float]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        self._ensure_loaded()
        # ``normalize_embeddings=True`` keeps vectors unit-norm so distance
        # in the vec0 table (default L2) is consistent with cosine.
        vec = self._model.encode(  # type: ignore[union-attr]
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        out = [float(x) for x in vec.tolist()]
        if len(out) != self.DIM:
            raise RuntimeError(
                f"SentenceTransformerEmbedder produced {len(out)}-dim "
                f"vector, expected {self.DIM}"
            )
        return out


# --------------------------------------------------------------------- test


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


class HashEmbedder:
    """Deterministic, non-semantic embedder for tests.

    Recipe: lowercase, tokenize on word-characters, hash each token via
    blake2b to a stable 64-bit int, mod ``dim`` for bucket, increment that
    bucket. After tokenization, L2-normalize so :py:func:`embed` outputs a
    unit-norm vector.

    Properties (used by the test suite):

    * Same input → identical output (no random state).
    * Different inputs → different vectors (with overwhelming probability
      given the 384-bucket space).
    * Strings sharing tokens land in overlapping buckets, so cosine distance
      between "revenue drop" and "revenue rose" is smaller than between
      either of them and "churn model".
    """

    def __init__(self, dim: int = 384) -> None:
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("dim must be a positive int")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        buckets = [0.0] * self._dim
        for tok in _TOKEN_RE.findall(text.lower()):
            h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h, "big") % self._dim
            buckets[idx] += 1.0
        norm = math.sqrt(sum(b * b for b in buckets))
        if norm == 0.0:
            # Empty / punctuation-only string -> fall back to a fixed
            # sentinel vector. Returning all-zeros would make cosine
            # distance undefined; instead, push everything into bucket 0.
            buckets[0] = 1.0
            return buckets
        return [b / norm for b in buckets]


__all__ = ["Embedder", "SentenceTransformerEmbedder", "HashEmbedder"]
