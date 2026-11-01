"""namespace — v1.1 architectural scaffold.

Design intent: formalize the ``(user_id, project_id)`` tenancy model that is
today expressed implicitly as a pair of keyword arguments threaded through
every store and tool factory. A first-class ``Namespace`` value lets us
(1) write tenancy invariants once and assert them at every store boundary,
(2) carry namespace into derived artifacts (URIs, log lines) without losing
the project context, and (3) extend cleanly to org/team scopes if multi-tenant
deployments need them.

v1.1 trigger: when the first cross-project read pattern appears (e.g. a user
asking "show me metrics across all my projects"). Until then, the two-tuple
of strings is fine.

Status: STUB — interface only, no behavior. See docs/architecture.md.
"""
from __future__ import annotations


class Namespace:
    """Formalized ``(user_id, project_id)`` tenancy handle."""

    def __init__(self, user_id: str, project_id: str) -> None:
        raise NotImplementedError(
            "namespace is a v1.1 scaffold; not yet implemented."
        )


__all__ = ["Namespace"]
