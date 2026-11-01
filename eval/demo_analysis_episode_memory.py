"""End-to-end CLI demo for the ``analysis_episode`` memory module.

Exercises every tool produced by :func:`make_analysis_episode_factory`
against an in-memory SQLite store, using the REAL
:class:`SentenceTransformerEmbedder` (``all-MiniLM-L6-v2``). Walks the
three user-provided examples (revenue drop, churn model, retention
analysis), demonstrates substring vs. semantic search, re-embedding on
user-feedback update, and project isolation.

Run from the repo root::

    cd /Users/wang/LLM_memory && \\
        .venv/bin/python eval/demo_analysis_episode_memory.py

The first run downloads the ``all-MiniLM-L6-v2`` model (~80 MB) from
HuggingFace. Subsequent runs reuse the cache under
``~/.cache/huggingface/hub``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path("/Users/wang/LLM_memory")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mads.memory.analysis_episode import AnalysisEpisodeStore  # noqa: E402
from mads.memory.analysis_episode_tools import (  # noqa: E402
    make_analysis_episode_factory,
)
from mads.memory.embedding import SentenceTransformerEmbedder  # noqa: E402


_UUID_RE = re.compile(r"[0-9a-f]{32}")


def header(n: int, title: str) -> None:
    print()
    line = f">>> Step {n}: {title}"
    print(line)
    print("-" * len(line))


def _extract_id(record_output: str) -> str:
    m = _UUID_RE.search(record_output)
    if m is None:
        raise RuntimeError(
            f"could not extract episode id from tool output: {record_output!r}"
        )
    return m.group(0)


def main() -> int:
    print("loading sentence-transformers model (first run may take 5-15s)...")
    embedder = SentenceTransformerEmbedder()
    store = AnalysisEpisodeStore(db_path=":memory:", embedder=embedder)

    # --------------------------------------------------------------------- 1
    header(1, "Set up factory for alice on project=growth")
    factory = make_analysis_episode_factory(store, "alice", "growth")
    by_name = {t.name: t for t in factory.tools()}
    print(
        "Tool names ("
        + str(len(by_name))
        + "): "
        + ", ".join(sorted(by_name.keys()))
    )
    record = by_name["record_analysis_episode"]
    get_ep = by_name["get_analysis_episode"]
    list_recent = by_name["list_recent_analyses"]
    find_sim = by_name["find_similar_analyses"]
    search = by_name["search_analyses"]
    update_fb = by_name["update_user_feedback"]

    # --------------------------------------------------------------------- 2
    header(2, "Record episode 1: revenue drop on 2026-05-15 (root-cause)")
    out1 = record.invoke(
        {
            "question": "Why did revenue drop on 2026-05-15?",
            "conclusion": (
                "Revenue drop was mainly caused by paid-search traffic "
                "mix shift."
            ),
            "analysis_type": "root_cause",
            "datasets_used": "payments, traffic_attribution",
            "metrics_used": "net_revenue, traffic_share",
        }
    )
    print(out1)
    revenue_id = _extract_id(out1)

    # --------------------------------------------------------------------- 3
    header(3, "Record episode 2: churn model improvement")
    out2 = record.invoke(
        {
            "question": "How can we improve the churn model?",
            "conclusion": (
                "Churn model improved after adding tenure and "
                "last_7d_sessions features."
            ),
            "analysis_type": "model_improvement",
            "metrics_used": "churn_rate, auc",
        }
    )
    print(out2)
    churn_id = _extract_id(out2)

    # --------------------------------------------------------------------- 4
    header(4, "Record episode 3: retention methodology")
    out3 = record.invoke(
        {
            "question": "What did the previous retention analysis look like?",
            "conclusion": (
                "Retention analysis used signup-week cohorts and excluded "
                "test users via is_test_user=false."
            ),
            "analysis_type": "retention_methodology",
            "methods_used": "cohort_analysis",
            "assumptions": "test users excluded; signup-week granularity",
        }
    )
    print(out3)
    retention_id = _extract_id(out3)

    # --------------------------------------------------------------------- 5
    header(5, "list_recent_analyses(limit=5) — newest first, all 3")
    print(list_recent.invoke({"limit": 5}))

    # --------------------------------------------------------------------- 6
    header(6, "get_analysis_episode(churn_id) — full detail")
    print(get_ep.invoke({"episode_id": churn_id}))

    # --------------------------------------------------------------------- 7
    header(7, "search_analyses('retention') — substring hits episode 3")
    print(search.invoke({"query": "retention"}))

    # --------------------------------------------------------------------- 8
    header(
        8,
        "find_similar_analyses('Did the revenue drop come from a specific "
        "traffic source?', k=2)",
    )
    print(
        find_sim.invoke(
            {
                "query": (
                    "Did the revenue drop come from a specific "
                    "traffic source?"
                ),
                "k": 2,
            }
        )
    )

    # --------------------------------------------------------------------- 9
    header(9, "update_user_feedback on episode 1 (re-embeds the row)")
    print(
        update_fb.invoke(
            {
                "episode_id": revenue_id,
                "feedback": (
                    "Follow-up: paid search recovered the next week; this "
                    "was a one-off mix-shift, not a structural change."
                ),
            }
        )
    )

    # --------------------------------------------------------------------- 10
    header(10, "get_analysis_episode(revenue_id) — verify feedback present")
    print(get_ep.invoke({"episode_id": revenue_id}))

    # --------------------------------------------------------------------- 11
    header(
        11,
        "find_similar_analyses('paid search recovery', k=2) — episode 1 "
        "now ranks at/near top",
    )
    print(find_sim.invoke({"query": "paid search recovery", "k": 2}))

    # --------------------------------------------------------------------- 12
    header(
        12,
        "Switch project_id=marketing -> empty; growth still has 3",
    )
    other_factory = make_analysis_episode_factory(store, "alice", "marketing")
    other_tools = {t.name: t for t in other_factory.tools()}
    print("marketing project ->")
    print(other_tools["list_recent_analyses"].invoke({"limit": 5}))
    print()
    print("growth project still ->")
    print(list_recent.invoke({"limit": 5}))

    # --------------------------------------------------------------------- 13
    print()
    print("=== DEMO COMPLETE ===")
    print(
        "Recorded ids: revenue=%s churn=%s retention=%s"
        % (revenue_id, churn_id, retention_id)
    )
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
