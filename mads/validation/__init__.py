"""mads.validation: post-agent validation scaffolding.

Today the wrapper returns an agent's :class:`TurnArtifacts` without inspecting
the answer for correctness. This subpackage scaffolds three v1.1 validators
that will run after every turn (result sanity, metric-usage correctness,
cross-category consistency).
"""
