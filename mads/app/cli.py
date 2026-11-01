"""Re-export shim: ``mads.app.cli`` re-exports the CLI entry point.

The canonical CLI lives at :mod:`mads.cli` (kept at the top level so
``python -m mads`` and the installed ``mads`` console-script don't have to
change). This shim exposes :func:`main` under ``mads.app.cli`` to match the
proposed v1.1 layout, where all user-facing entry points sit under
``mads.app``.
"""

from mads.cli import main

__all__ = ["main"]
