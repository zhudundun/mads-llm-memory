"""Adapters wrap upstream third-party libraries behind stable surfaces.

The :mod:`mads.adapters.ads` adapter is the ONLY module outside ``eval/`` that
may ``import ai_data_science_team``. Everything else in ``mads`` consumes
:class:`mads.adapters.ads.AgentResult` from it.
"""
