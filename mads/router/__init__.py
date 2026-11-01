"""mads.router: request routing layer.

Public surface: ``route(question) -> AgentName``. The current implementation is
a pure-regex heuristic in :mod:`mads.router.intent_classifier`; the package
shape leaves room for a v1.1 read-side ``context_builder`` that assembles
prompts from memory recalls (see :mod:`mads.router.context_builder`).
"""

from mads.router.intent_classifier import AgentName, route

__all__ = ["route", "AgentName"]
