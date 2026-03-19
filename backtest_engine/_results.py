"""TradeResults — numpy structured array with quality metadata."""

from __future__ import annotations

import numpy as np


class TradeResults(np.ndarray):
    """numpy structured array subclass that carries a ``.quality`` attribute.

    All standard numpy operations work unchanged::

        results["pnl_r"]         # field access
        results[0]               # indexing
        len(results)             # length
        results[results["pnl_r"] > 0]  # boolean slicing

    The extra attribute provides pre-flight quality info::

        results.quality.grade    # "A", "B", or "C"
    """

    def __new__(cls, input_array, quality=None):
        obj = np.asarray(input_array).view(cls)
        obj.quality = quality
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.quality = getattr(obj, "quality", None)

    def __getitem__(self, key):
        result = super().__getitem__(key)
        # Field access (e.g. results["pnl_r"]) should return a plain ndarray,
        # not TradeResults.  Otherwise np.mean(results["pnl_r"]) returns a
        # TradeResults scalar that breaks JSON serialization and isinstance checks.
        if isinstance(key, str):
            return np.asarray(result)
        return result

    def __reduce__(self):
        # pickle support
        pickled_state = super().__reduce__()
        new_state = pickled_state[2] + (self.quality,)
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        self.quality = state[-1]
        super().__setstate__(state[:-1])
