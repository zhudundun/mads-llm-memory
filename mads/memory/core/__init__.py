"""mads.memory.core: cross-category memory primitives.

Currently re-exports :class:`MemoryManager` from :mod:`mads.manager` so it can
be reached at ``mads.memory.core.manager.MemoryManager``. Everything else in
this subpackage is a v1.1 scaffold (centralized schemas, namespace formalism,
policies, scoring).
"""

from mads.memory.core.manager import MemoryManager

__all__ = ["MemoryManager"]
