"""``mads`` command-line entry point (stdlib argparse).

See V1_DESIGN_BRIEF §7. Two top-level subcommands:

* ``mads ask "<question>"``  — route → wrap → invoke → compose → stdout.
* ``mads memory ...``        — browse / inspect the memory stores.

Exit codes::

    0  success
    1  caught domain error (no metric found, invalid args from tool layer)
    2  argparse/usage error (missing --user, unknown subcommand) — argparse default
    3  unexpected exception (uncaught) — print traceback to stderr

Defaults are resolved from environment variables ``MADS_USER``,
``MADS_PROJECT``, ``MADS_DB_PATH`` and from the user's shell login (``$USER``).
The LLM provider key (``ANTHROPIC_API_KEY``) is only required for ``mads ask``;
the ``mads memory`` subcommands do not call an LLM.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

from mads.manager import MemoryManager


# -----------------------------------------------------------------------------
# Help epilogs (kept short; the CLI docs live in the brief and CLAUDE.md)
# -----------------------------------------------------------------------------
_EPILOG_ASK = (
    "Environment:\n"
    "  MADS_USER       default --user\n"
    "  MADS_PROJECT    default --project (else 'default')\n"
    "  MADS_DB_PATH    default --db (else ./.mads/memory.sqlite)\n"
    "  ANTHROPIC_API_KEY  required for `mads ask` (loads the Sonnet 4.6 model)\n"
)


# -----------------------------------------------------------------------------
# Resolver helpers
# -----------------------------------------------------------------------------
def _resolve_user(arg: Optional[str]) -> Optional[str]:
    """``--user`` arg > ``MADS_USER`` env > ``$USER``. Returns None if all unset."""
    if arg:
        return arg
    return os.environ.get("MADS_USER") or os.environ.get("USER") or None


def _resolve_project(arg: Optional[str]) -> str:
    if arg:
        return arg
    return os.environ.get("MADS_PROJECT") or "default"


def _resolve_db(arg: Optional[str]) -> Optional[str]:
    """Explicit ``--db`` wins over ``$MADS_DB_PATH``. Returns None to defer
    to ``MemoryManager``'s own default (``./.mads/memory.sqlite``)."""
    if arg:
        return arg
    return os.environ.get("MADS_DB_PATH") or None


def _require_user(user: Optional[str]) -> str:
    if not user:
        print(
            "error: --user is required (or set $MADS_USER or $USER).",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return user


# -----------------------------------------------------------------------------
# Manager construction helper
# -----------------------------------------------------------------------------
def _build_manager(
    *, user_id: str, project_id: str, db_path: Optional[str]
) -> MemoryManager:
    return MemoryManager(
        user_id=user_id,
        project_id=project_id,
        db_path=db_path,
    )


# -----------------------------------------------------------------------------
# `mads ask`
# -----------------------------------------------------------------------------
def cmd_ask(args: argparse.Namespace) -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "error: ANTHROPIC_API_KEY is not set. `mads ask` requires Anthropic "
            "credentials. Set it in your shell (e.g. via .env) and retry.",
            file=sys.stderr,
        )
        return 2

    user = _require_user(_resolve_user(args.user))
    project = _resolve_project(args.project)
    db_path = _resolve_db(args.db)

    # Late imports so `mads memory ...` and `mads --help` don't pay for the
    # ADS + langchain-anthropic import cost (and don't crash on missing ADS).
    from mads.router import route
    from mads.composer import compose
    from mads.agents.wrapper import wrap_eda, wrap_loader, wrap_viz

    agent_name = route(args.question)
    memory_aware = bool(getattr(args, "memory_aware", False))

    manager = _build_manager(user_id=user, project_id=project, db_path=db_path)
    try:
        if agent_name == "loader":
            # Loader has no df yet; --data (if any) is its responsibility.
            # MemoryAwareEDA only covers the EDA path; loader still uses the
            # reactive wrapper.
            wrapped = wrap_loader(manager)
            turn = wrapped.invoke(args.question)
        else:
            if not args.data:
                print(
                    f"error: --data PATH is required for agent '{agent_name}' "
                    "(loader is the only agent that loads its own data in v1).",
                    file=sys.stderr,
                )
                return 2
            import pandas as pd

            df = pd.read_csv(args.data)
            dataset_name = Path(args.data).stem
            table_name = dataset_name
            if agent_name == "eda" and memory_aware:
                from mads.wrappers.memory_aware import MemoryAwareEDA

                wrapped = MemoryAwareEDA(manager)
                turn = wrapped.run(
                    args.question,
                    df,
                    user_id=user,
                    project_id=project,
                    dataset_name=dataset_name,
                    table_name=table_name,
                )
            elif agent_name == "eda":
                wrapped = wrap_eda(
                    manager,
                    df,
                    dataset_name=dataset_name,
                    table_name=table_name,
                )
                turn = wrapped.invoke(args.question)
            else:  # viz — --memory-aware is silently ignored for viz in this build
                wrapped = wrap_viz(
                    manager,
                    df,
                    dataset_name=dataset_name,
                    table_name=table_name,
                )
                turn = wrapped.invoke(args.question)
        rendered = compose(turn)
        sys.stdout.write(rendered)
        if not rendered.endswith("\n"):
            sys.stdout.write("\n")
        return 0
    finally:
        manager.close()


# -----------------------------------------------------------------------------
# `mads memory` helpers
# -----------------------------------------------------------------------------
def _find_tool(tools: list, name: str):
    for t in tools:
        if getattr(t, "name", None) == name:
            return t
    raise KeyError(name)


def _emit(text: Any) -> None:
    s = text if isinstance(text, str) else str(text)
    sys.stdout.write(s)
    if not s.endswith("\n"):
        sys.stdout.write("\n")


# ---- memory list -------------------------------------------------------------
def cmd_memory_list_metric(args: argparse.Namespace) -> int:
    user = _require_user(_resolve_user(args.user))
    project = _resolve_project(args.project)
    db_path = _resolve_db(args.db)
    manager = _build_manager(user_id=user, project_id=project, db_path=db_path)
    try:
        tool = _find_tool(manager.metric.tools(), "list_metric_definitions")
        _emit(tool.invoke({}))
        return 0
    finally:
        manager.close()


def cmd_memory_list_dataset(args: argparse.Namespace) -> int:
    user = _require_user(_resolve_user(args.user))
    project = _resolve_project(args.project)
    db_path = _resolve_db(args.db)
    manager = _build_manager(user_id=user, project_id=project, db_path=db_path)
    try:
        # list_datasets is a dataset tool; we need an arbitrary 1-row df to
        # build the tool list, but list_datasets only consults the store. We
        # bind a sentinel empty frame so dataset_tools() succeeds.
        import pandas as pd

        manager.bind_dataframe(
            pd.DataFrame({"_sentinel": [0]}),
            dataset_name="_browse",
            table_name="_browse",
        )
        tool = _find_tool(manager.dataset_tools(), "list_datasets")
        _emit(tool.invoke({}))
        return 0
    finally:
        manager.close()


def cmd_memory_list_episode(args: argparse.Namespace) -> int:
    user = _require_user(_resolve_user(args.user))
    project = _resolve_project(args.project)
    db_path = _resolve_db(args.db)
    manager = _build_manager(user_id=user, project_id=project, db_path=db_path)
    try:
        tool = _find_tool(manager.episode.tools(), "list_recent_analyses")
        _emit(tool.invoke({"limit": args.limit}))
        return 0
    finally:
        manager.close()


# ---- memory show -------------------------------------------------------------
def cmd_memory_show_metric(args: argparse.Namespace) -> int:
    user = _require_user(_resolve_user(args.user))
    project = _resolve_project(args.project)
    db_path = _resolve_db(args.db)
    manager = _build_manager(user_id=user, project_id=project, db_path=db_path)
    try:
        tool = _find_tool(manager.metric.tools(), "get_metric_definition")
        _emit(tool.invoke({"metric_name": args.name}))
        return 0
    finally:
        manager.close()


def cmd_memory_show_episode(args: argparse.Namespace) -> int:
    user = _require_user(_resolve_user(args.user))
    project = _resolve_project(args.project)
    db_path = _resolve_db(args.db)
    manager = _build_manager(user_id=user, project_id=project, db_path=db_path)
    try:
        tool = _find_tool(manager.episode.tools(), "get_analysis_episode")
        _emit(tool.invoke({"episode_id": args.id}))
        return 0
    finally:
        manager.close()


def cmd_memory_show_dataset(args: argparse.Namespace) -> int:
    user = _require_user(_resolve_user(args.user))
    project = _resolve_project(args.project)
    db_path = _resolve_db(args.db)
    manager = _build_manager(user_id=user, project_id=project, db_path=db_path)
    try:
        # We need a df to construct the dataset tools; recall_dataset / list_columns
        # are bound to (fingerprint, table_name). Sentinel df is fine because
        # recall_dataset reads by table_name from the store, but list_columns
        # filters by fingerprint — so this is a best-effort browse on a fresh
        # sentinel and won't return rows. Document and accept.
        import pandas as pd

        manager.bind_dataframe(
            pd.DataFrame({"_sentinel": [0]}),
            dataset_name=args.table,
            table_name=args.table,
        )
        tool = _find_tool(manager.dataset_tools(), "recall_dataset")
        _emit(tool.invoke({}))
        return 0
    finally:
        manager.close()


# ---- memory history ----------------------------------------------------------
def cmd_memory_history_metric(args: argparse.Namespace) -> int:
    user = _require_user(_resolve_user(args.user))
    project = _resolve_project(args.project)
    db_path = _resolve_db(args.db)
    manager = _build_manager(user_id=user, project_id=project, db_path=db_path)
    try:
        tool = _find_tool(manager.metric.tools(), "get_metric_history")
        _emit(tool.invoke({"metric_name": args.name}))
        return 0
    finally:
        manager.close()


# ---- memory similar ----------------------------------------------------------
def cmd_memory_similar_episode(args: argparse.Namespace) -> int:
    user = _require_user(_resolve_user(args.user))
    project = _resolve_project(args.project)
    db_path = _resolve_db(args.db)
    manager = _build_manager(user_id=user, project_id=project, db_path=db_path)
    try:
        tool = _find_tool(manager.episode.tools(), "find_similar_analyses")
        _emit(tool.invoke({"query": args.query, "k": args.k}))
        return 0
    finally:
        manager.close()


# -----------------------------------------------------------------------------
# Parser construction
# -----------------------------------------------------------------------------
def _add_common(p: argparse.ArgumentParser, *, with_user: bool = True) -> None:
    if with_user:
        p.add_argument("--user", default=None)
    p.add_argument("--project", default=None)
    p.add_argument("--db", default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mads",
        description="Memory-augmented data scientist CLI.",
        epilog=_EPILOG_ASK,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd")

    # -- ask ---------------------------------------------------------------
    p_ask = sub.add_parser("ask", help="Ask a question routed to one of loader/eda/viz.")
    p_ask.add_argument("question", help="Natural-language question for the agent.")
    p_ask.add_argument(
        "--data", default=None, help="CSV path passed to eda/viz (ignored by loader)."
    )
    p_ask.add_argument(
        "--memory-aware",
        action="store_true",
        dest="memory_aware",
        help=(
            "Route EDA through the proactive MemoryAwareEDA wrapper (memory "
            "is retrieved + injected into the prompt before the LLM runs). "
            "No effect on loader or viz in this build."
        ),
    )
    _add_common(p_ask)
    p_ask.set_defaults(func=cmd_ask)

    # -- memory ------------------------------------------------------------
    p_mem = sub.add_parser("memory", help="Browse the memory stores.")
    mem_sub = p_mem.add_subparsers(dest="mem_cmd")

    # memory list
    p_list = mem_sub.add_parser("list", help="List memories of a given category.")
    list_sub = p_list.add_subparsers(dest="list_cat")

    pl_m = list_sub.add_parser("metric", help="List active metric definitions.")
    _add_common(pl_m)
    pl_m.set_defaults(func=cmd_memory_list_metric)

    pl_d = list_sub.add_parser("dataset", help="List registered datasets.")
    _add_common(pl_d)
    pl_d.set_defaults(func=cmd_memory_list_dataset)

    pl_e = list_sub.add_parser("episode", help="List recent analysis episodes.")
    _add_common(pl_e)
    pl_e.add_argument("--limit", type=int, default=10)
    pl_e.set_defaults(func=cmd_memory_list_episode)

    # memory show
    p_show = mem_sub.add_parser("show", help="Show a single memory by key.")
    show_sub = p_show.add_subparsers(dest="show_cat")

    ps_m = show_sub.add_parser("metric", help="Show one metric by name.")
    ps_m.add_argument("name")
    _add_common(ps_m)
    ps_m.set_defaults(func=cmd_memory_show_metric)

    ps_e = show_sub.add_parser("episode", help="Show one episode by id.")
    ps_e.add_argument("id")
    _add_common(ps_e)
    ps_e.set_defaults(func=cmd_memory_show_episode)

    ps_d = show_sub.add_parser("dataset", help="Show one dataset by table name.")
    ps_d.add_argument("table")
    _add_common(ps_d)
    ps_d.set_defaults(func=cmd_memory_show_dataset)

    # memory history
    p_hist = mem_sub.add_parser("history", help="Show full version history.")
    hist_sub = p_hist.add_subparsers(dest="hist_cat")
    ph_m = hist_sub.add_parser("metric", help="Show full history of a metric.")
    ph_m.add_argument("name")
    _add_common(ph_m)
    ph_m.set_defaults(func=cmd_memory_history_metric)

    # memory similar
    p_sim = mem_sub.add_parser("similar", help="Semantic similarity search.")
    sim_sub = p_sim.add_subparsers(dest="sim_cat")
    psm_e = sim_sub.add_parser("episode", help="Find similar analysis episodes.")
    psm_e.add_argument("query")
    _add_common(psm_e)
    psm_e.add_argument("--k", type=int, default=5)
    psm_e.set_defaults(func=cmd_memory_similar_episode)

    return parser


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    func = getattr(args, "func", None)
    if func is None:
        parser.print_help(sys.stderr)
        return 2

    try:
        rc = func(args)
        return int(rc) if rc is not None else 0
    except SystemExit as e:
        # Honor cmd-internal SystemExit (e.g. _require_user). argparse's own
        # parse_args already raised SystemExit before we got here; this is for
        # downstream domain-level exits.
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        return code
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception:
        traceback.print_exc()
        return 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
