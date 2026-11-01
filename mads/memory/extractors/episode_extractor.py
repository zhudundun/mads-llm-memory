"""episode_extractor — v1.1 architectural scaffold.

Design intent: mines ``analysis_episode`` candidates from agent transcripts
when the LLM doesn't call ``record_analysis_episode`` explicitly. Episodes
are the highest-value memory category (they're embedded and drive
find_similar) but also the easiest for the LLM to forget — every turn
*should* produce an episode if a real analytical conclusion was reached. The
extractor scans the agent's final answer for analytical conclusions
("revenue grew 12% Q-over-Q", "churn is driven by tenure < 90d") and
proposes an episode payload.

v1.1 trigger: after observed gap-closure rate <30% in the with-memory eval
for episodes specifically — the brief calls out that catching these is the
single highest-impact lever for downstream recall quality.

Status: STUB — interface only, no behavior. See docs/architecture.md.
"""
from __future__ import annotations

from typing import Any


class EpisodeExtractor:
    """Mines candidate analysis_episode rows from agent transcripts."""

    def extract(self, transcript: str, *, agent_name: str) -> list[dict[str, Any]]:
        """Return candidate analysis_episode payloads.

        Inputs: the agent transcript and the agent name (loader/eda/viz) so
        the extractor can apply agent-specific heuristics. Output: zero or
        more candidate payloads. Episodes are append-only so the extractor's
        false-positive cost is bounded — duplicates simply add to the
        embedding index — but the implementation should still de-dupe
        against the most recent same-day episode for the same fingerprint.
        """
        raise NotImplementedError(
            "episode_extractor is a v1.1 scaffold; not yet implemented."
        )


__all__ = ["EpisodeExtractor"]
