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

    # ── Cost awareness ────────────────────────────────────────────────

    @property
    def is_gross(self) -> bool:
        """True if no execution costs were applied (all cost_r == 0)."""
        if len(self) == 0:
            return True
        return float(np.sum(np.abs(np.asarray(self["cost_r"])))) == 0.0

    @property
    def cost_label(self) -> str:
        """'GROSS (no execution costs)' or 'NET (costs applied)'."""
        return "GROSS (no execution costs)" if self.is_gross else "NET (costs applied)"

    # ── Convenience metrics ─────────────────────────────────────────────

    @property
    def profit_factor(self) -> float:
        """Gross profit / gross loss.  Returns inf if no losses, 0.0 if empty."""
        if len(self) == 0:
            return 0.0
        pnl = np.asarray(self["pnl_r"])
        gross_profit = float(np.sum(pnl[pnl > 0]))
        gross_loss = float(abs(np.sum(pnl[pnl < 0])))
        if gross_loss == 0.0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @property
    def win_rate(self) -> float:
        """Fraction of trades with pnl_r > 0.  Returns 0.0 if empty."""
        if len(self) == 0:
            return 0.0
        return float(np.mean(np.asarray(self["pnl_r"]) > 0))

    @property
    def expectancy_r(self) -> float:
        """Mean pnl_r (expected R per trade).  Returns 0.0 if empty."""
        if len(self) == 0:
            return 0.0
        return float(np.mean(np.asarray(self["pnl_r"])))

    @property
    def geometric_mean_r(self) -> float:
        """Geometric mean of (1 + risk_frac * pnl_r) - 1, using risk_frac=0.01.

        Approximates the per-trade compound growth rate.  Returns 0.0 if
        empty or if any trade causes total ruin (equity <= 0).
        """
        if len(self) == 0:
            return 0.0
        risk_frac = 0.01
        pnl = np.asarray(self["pnl_r"])
        factors = 1.0 + risk_frac * pnl
        if np.any(factors <= 0):
            return 0.0
        log_mean = float(np.mean(np.log(factors)))
        return float(np.exp(log_mean) - 1.0)

    @property
    def sharpe_r(self) -> float:
        """Sharpe-like ratio: mean(pnl_r) / std(pnl_r).  Returns 0.0 if < 2 trades."""
        if len(self) < 2:
            return 0.0
        pnl = np.asarray(self["pnl_r"])
        std = float(np.std(pnl, ddof=1))
        if std == 0.0:
            return 0.0
        return float(np.mean(pnl)) / std

    @property
    def sortino_r(self) -> float:
        """Sortino-like ratio: mean(pnl_r) / downside_std(pnl_r).  Returns 0.0 if < 2 trades."""
        if len(self) < 2:
            return 0.0
        pnl = np.asarray(self["pnl_r"])
        downside = pnl[pnl < 0]
        if len(downside) < 2:
            return float("inf") if float(np.mean(pnl)) > 0 else 0.0
        downside_std = float(np.std(downside, ddof=1))
        if downside_std == 0.0:
            return 0.0
        return float(np.mean(pnl)) / downside_std

    @property
    def max_drawdown_r(self) -> float:
        """Maximum drawdown of cumulative R curve.  Returns 0.0 if empty."""
        if len(self) == 0:
            return 0.0
        cum = np.concatenate(([0.0], np.cumsum(np.asarray(self["pnl_r"]))))
        peak = np.maximum.accumulate(cum)
        dd = peak - cum
        return float(np.max(dd))

    @property
    def recovery_factor(self) -> float:
        """total_r / max_drawdown_r.  Returns 0.0 if no drawdown."""
        if len(self) == 0:
            return 0.0
        total = float(np.sum(np.asarray(self["pnl_r"])))
        mdd = self.max_drawdown_r
        if mdd == 0.0:
            return float("inf") if total > 0 else 0.0
        return total / mdd
