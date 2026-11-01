"""Re-export shim for :class:`MemoryManager`.

The manager lives at the top of the ``mads`` package because it is a *facade*
over the memory categories (it builds tool lists, binds dataframes, owns
tenancy), not a member of the memory subpackage. This shim exposes it under
``mads.memory.core.manager`` so the layout matches the proposed v1.1 shape
where the entire memory layer can be reasoned about as a subtree.
"""

from mads.manager import MemoryManager

__all__ = ["MemoryManager"]
