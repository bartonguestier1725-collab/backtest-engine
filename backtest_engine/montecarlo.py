"""Monte Carlo drawdown analysis, Kelly criterion, and risk sizing."""

from __future__ import annotations

import numpy as np
import numba


@numba.njit(cache=True)
def _mc_shuffle_compound(
    pnl_r: np.ndarray,
    n_sims: int,
    risk_pct: float,
    seed: int,
) -> np.ndarray:
    """Run Monte Carlo simulations with Fisher-Yates shuffle + compound equity.

    Returns max drawdown (as positive fraction) for each simulation.
    """
    n_trades = len(pnl_r)
    max_dds = np.empty(n_sims, dtype=np.float64)

    # Simple LCG RNG (numba doesn't support np.random with seed per call easily)
    rng_state = np.uint64(seed)

    for sim in range(n_sims):
        # Fisher-Yates shuffle (in-place on a copy)
        shuffled = pnl_r.copy()
        for i in range(n_trades - 1, 0, -1):
            # LCG: state = (a * state + c) mod m
            rng_state = np.uint64(np.uint64(6364136223846793005) * rng_state + np.uint64(1442695040888963407))
            j = int(rng_state >> np.uint64(33)) % (i + 1)
            shuffled[i], shuffled[j] = shuffled[j], shuffled[i]

        # Compound equity curve
        equity = 1.0
        peak = 1.0
        max_dd = 0.0

        for i in range(n_trades):
            equity *= (1.0 + risk_pct * shuffled[i])
            if equity <= 0.0:
                max_dd = 1.0
                break
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd

        max_dds[sim] = max_dd

    return max_dds


class MonteCarloDD:
    """Monte Carlo drawdown analysis.

    Parameters
    ----------
    pnl_r : 1-D array of trade results in R-multiples.
    n_sims : Number of simulations (default 10_000).
    risk_pct : Risk per trade as decimal (default 0.01 = 1%).
    seed : Random seed for reproducibility.
    """

    def __init__(
        self,
        pnl_r: np.ndarray,
        n_sims: int = 10_000,
        risk_pct: float = 0.01,
        seed: int = 42,
    ):
        self.pnl_r = np.ascontiguousarray(pnl_r, dtype=np.float64)
        self.n_sims = n_sims
        self.risk_pct = risk_pct
        self.seed = seed
        self._max_dds: np.ndarray | None = None

    def run(self) -> np.ndarray:
        """Run Monte Carlo simulations. Returns array of max drawdowns."""
        self._max_dds = _mc_shuffle_compound(
            self.pnl_r, self.n_sims, self.risk_pct, self.seed,
        )
        return self._max_dds

    @property
    def max_dds(self) -> np.ndarray:
        if self._max_dds is None:
            self.run()
        return self._max_dds

    def dd_percentile(self, pct: float) -> float:
        """Drawdown at given percentile (e.g. 95 → 95th percentile DD)."""
        return float(np.percentile(self.max_dds, pct))

    def ruin_probability(self, threshold: float) -> float:
        """Probability of max DD exceeding threshold (e.g. 0.10 = 10% DD)."""
        return float(np.mean(self.max_dds > threshold))

    def kelly_fraction(self) -> float:
        """Kelly criterion: f* = (p*b - q) / b

        where p = win rate, b = avg win / avg loss, q = 1 - p.
        """
        wins = self.pnl_r[self.pnl_r > 0]
        losses = self.pnl_r[self.pnl_r < 0]

        if len(wins) == 0 or len(losses) == 0:
            return 0.0

        p = len(wins) / len(self.pnl_r)
        q = 1.0 - p
        b = np.mean(wins) / abs(np.mean(losses))

        kelly = (p * b - q) / b
        return max(0.0, kelly)

    def optimal_risk_pct(
        self,
        max_dd: float = 0.20,
        target_pct: float = 95.0,
        lo: float = 0.001,
        hi: float = 0.10,
        tol: float = 0.0005,
    ) -> float:
        """Find optimal risk% via binary search so that DD at target_pct ≤ max_dd.

        Parameters
        ----------
        max_dd : Maximum acceptable DD (e.g. 0.20 = 20%).
        target_pct : Percentile for DD constraint (e.g. 95).
        lo, hi : Search bounds for risk%.
        tol : Convergence tolerance.
        """
        best = lo
        for _ in range(50):  # max iterations
            mid = (lo + hi) / 2.0
            mc = MonteCarloDD(self.pnl_r, self.n_sims, mid, self.seed)
            mc.run()
            dd_at_pct = mc.dd_percentile(target_pct)

            if dd_at_pct <= max_dd:
                best = mid
                lo = mid + tol
            else:
                hi = mid - tol

            if hi - lo < tol:
                break

        return best

    def prop_firm_check(
        self,
        max_dd_limit: float = 0.04,
        total_dd_limit: float = 0.08,
        confidence: float = 95.0,
        # Backward compat
        daily_dd_limit: float | None = None,
    ) -> dict:
        """Check if strategy meets prop firm DD constraints.

        Parameters
        ----------
        max_dd_limit : Max drawdown at confidence percentile (e.g. 0.04 = 4%).
                       This checks the Monte Carlo max DD distribution, NOT daily DD.
        total_dd_limit : Max drawdown at 99th percentile (e.g. 0.08 = 8%).
        confidence : Percentile for max_dd_limit check (e.g. 95.0).
        daily_dd_limit : Deprecated alias for max_dd_limit.

        Returns dict with pass/fail and DD statistics.
        """
        # Backward compat: daily_dd_limit → max_dd_limit
        if daily_dd_limit is not None:
            import warnings
            warnings.warn(
                "daily_dd_limit is renamed to max_dd_limit. "
                "The check uses Monte Carlo max DD, not daily DD.",
                DeprecationWarning,
                stacklevel=2,
            )
            max_dd_limit = daily_dd_limit

        dd_conf = self.dd_percentile(confidence)
        dd99 = self.dd_percentile(99.0)

        return {
            "risk_pct": self.risk_pct,
            "dd_confidence": dd_conf,
            "dd_99": dd99,
            "max_dd_ok": dd_conf < max_dd_limit,
            "total_dd_ok": dd99 < total_dd_limit,
            "pass": dd_conf < max_dd_limit and dd99 < total_dd_limit,
            # Backward compat alias
            "daily_dd_ok": dd_conf < max_dd_limit,
        }

    def fundora_check(
        self,
        daily_dd_limit: float = 0.04,
        total_dd_limit: float = 0.08,
        confidence: float = 95.0,
    ) -> dict:
        """Deprecated: use prop_firm_check() instead."""
        import warnings
        warnings.warn(
            "fundora_check() is deprecated, use prop_firm_check() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        result = self.prop_firm_check(max_dd_limit=daily_dd_limit, total_dd_limit=total_dd_limit, confidence=confidence)
        result["dd_95"] = result["dd_confidence"]
        return result
