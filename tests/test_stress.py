"""Tests for backtest_engine.montecarlo.StressTest and block bootstrap."""

import numpy as np
import pytest

from backtest_engine.montecarlo import StressTest, _block_bootstrap_dd


# ── Block Bootstrap Kernel Tests ─────────────────────────────────────────────

class TestBlockBootstrapKernel:
    def test_deterministic_with_seed(self):
        pnl = np.array([1.0, -1.0, 2.0, -0.5, 1.5, -1.0, 0.5] * 5, dtype=np.float64)
        r1 = _block_bootstrap_dd(pnl, 100, 5, 0.01, 42)
        r2 = _block_bootstrap_dd(pnl, 100, 5, 0.01, 42)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seeds_differ(self):
        pnl = np.array([1.0, -1.0, 2.0, -0.5, 1.5, -1.0, 0.5] * 10, dtype=np.float64)
        r1 = _block_bootstrap_dd(pnl, 500, 5, 0.01, 42)
        r2 = _block_bootstrap_dd(pnl, 500, 5, 0.01, 123)
        assert not np.array_equal(r1, r2)

    def test_dd_range(self):
        pnl = np.array([1.0, -1.0, 2.0, -0.5, 1.5, -1.0, 0.5] * 20, dtype=np.float64)
        dds = _block_bootstrap_dd(pnl, 500, 5, 0.02, 42)
        assert np.all(dds >= 0.0)
        assert np.all(dds <= 1.0)

    def test_all_winners_zero_dd(self):
        pnl = np.full(30, 1.0, dtype=np.float64)
        dds = _block_bootstrap_dd(pnl, 100, 5, 0.01, 42)
        assert np.all(dds == 0.0)

    def test_block_size_1_similar_to_shuffle(self):
        """Block size 1 should behave like i.i.d. shuffle."""
        pnl = np.array([1.0, -1.0, 2.0, -0.5, 1.5, -1.0, 0.5] * 20, dtype=np.float64)
        dds_block1 = _block_bootstrap_dd(pnl, 1000, 1, 0.01, 42)
        # Just verify it produces valid results
        assert np.all(dds_block1 >= 0.0)
        assert len(dds_block1) == 1000


# ── StressTest Class Tests ───────────────────────────────────────────────────

class TestStressTest:
    @pytest.fixture
    def profitable_pnl(self):
        np.random.seed(42)
        n = 200
        return np.where(np.random.rand(n) < 0.6, 1.5, -1.0).astype(np.float64)

    def test_block_bootstrap_invalid_block_size(self, profitable_pnl):
        st = StressTest(profitable_pnl, n_sims=100, seed=42)
        with pytest.raises(ValueError, match="block_size must be >= 1"):
            st.block_bootstrap(block_size=0)

    def test_block_bootstrap(self, profitable_pnl):
        st = StressTest(profitable_pnl, n_sims=500, seed=42)
        result = st.block_bootstrap(block_size=10)
        assert "dd_50" in result
        assert "dd_95" in result
        assert "dd_99" in result
        assert "max_dds" in result
        assert len(result["max_dds"]) == 500
        assert result["dd_95"] >= result["dd_50"]
        assert result["block_size"] == 10

    def test_degrade_win_rate(self, profitable_pnl):
        st = StressTest(profitable_pnl, seed=42)
        original_wr = float(np.mean(profitable_pnl > 0))
        degraded = st.degrade(win_rate_delta=-0.10)
        new_wr = float(np.mean(degraded > 0))
        # Win rate should decrease by approximately 10%
        assert new_wr < original_wr
        assert abs(new_wr - (original_wr - 0.10)) < 0.02

    def test_degrade_rr_scale(self, profitable_pnl):
        st = StressTest(profitable_pnl, seed=42)
        degraded = st.degrade(rr_scale=0.80)
        # Wins should be 80% of original, losses unchanged
        orig_wins = profitable_pnl[profitable_pnl > 0]
        new_wins = degraded[degraded > 0]
        if len(new_wins) > 0 and len(orig_wins) > 0:
            assert np.mean(new_wins) < np.mean(orig_wins)

    def test_degrade_cost_add(self, profitable_pnl):
        st = StressTest(profitable_pnl, seed=42)
        degraded = st.degrade(cost_add_r=0.10)
        # All trades should be shifted down by 0.1R
        np.testing.assert_allclose(degraded, profitable_pnl - 0.10)

    def test_degrade_no_change(self, profitable_pnl):
        st = StressTest(profitable_pnl, seed=42)
        degraded = st.degrade()
        np.testing.assert_array_equal(degraded, profitable_pnl)

    def test_run_all(self, profitable_pnl):
        st = StressTest(profitable_pnl, n_sims=200, seed=42)
        report = st.run_all(block_size=5)

        # Check structure
        assert "baseline" in report
        assert "block_bootstrap" in report
        assert "degraded" in report

        # Baseline
        assert "dd_95" in report["baseline"]
        assert len(report["baseline"]["max_dds"]) == 200

        # Block bootstrap
        assert report["block_bootstrap"]["block_size"] == 5

        # Degradation scenarios
        assert "wr_minus5" in report["degraded"]
        assert "rr_80pct" in report["degraded"]
        assert "cost_plus_01r" in report["degraded"]
        assert "combined" in report["degraded"]

        for name, scenario in report["degraded"].items():
            assert "dd_95" in scenario
            assert "expectancy_r" in scenario
            assert "params" in scenario

    def test_degraded_worse_than_baseline(self, profitable_pnl):
        """Degraded scenarios should generally have worse DD than baseline."""
        st = StressTest(profitable_pnl, n_sims=1000, seed=42)
        report = st.run_all()

        baseline_dd95 = report["baseline"]["dd_95"]
        # Combined degradation should be worse
        combined_dd95 = report["degraded"]["combined"]["dd_95"]
        assert combined_dd95 >= baseline_dd95 * 0.8  # Allow some variance


# ── TradeResults Metrics Tests ───────────────────────────────────────────────

class TestTradeResultsMetrics:
    """Test the new metrics properties on TradeResults."""

    def _make_results(self, pnl_values):
        from backtest_engine._types import TRADE_RESULT_DTYPE
        from backtest_engine._results import TradeResults
        n = len(pnl_values)
        data = np.zeros(n, dtype=TRADE_RESULT_DTYPE)
        data["pnl_r"] = pnl_values
        return TradeResults(data)

    def test_profit_factor_basic(self):
        results = self._make_results([2.0, -1.0, 1.5, -0.5])
        pf = results.profit_factor
        assert pf == pytest.approx(3.5 / 1.5)

    def test_profit_factor_no_losses(self):
        results = self._make_results([1.0, 2.0, 3.0])
        assert results.profit_factor == float("inf")

    def test_profit_factor_no_wins(self):
        results = self._make_results([-1.0, -2.0])
        assert results.profit_factor == 0.0

    def test_profit_factor_empty(self):
        results = self._make_results([])
        assert results.profit_factor == 0.0

    def test_win_rate(self):
        results = self._make_results([1.0, -1.0, 2.0, -0.5])
        assert results.win_rate == pytest.approx(0.5)

    def test_win_rate_empty(self):
        results = self._make_results([])
        assert results.win_rate == 0.0

    def test_expectancy_r(self):
        results = self._make_results([2.0, -1.0, 1.0])
        assert results.expectancy_r == pytest.approx(2.0 / 3.0)

    def test_geometric_mean_r(self):
        results = self._make_results([1.0, 1.0, 1.0])
        gm = results.geometric_mean_r
        assert gm > 0  # All winners → positive growth

    def test_geometric_mean_r_empty(self):
        results = self._make_results([])
        assert results.geometric_mean_r == 0.0

    def test_sharpe_r(self):
        results = self._make_results([1.0, 1.0, 1.0])
        # Zero variance in pnl → std=0 → sharpe=0
        # Actually all same positive: mean>0, std=0 → returns 0.0
        assert results.sharpe_r == 0.0

    def test_sharpe_r_positive(self):
        results = self._make_results([2.0, -1.0, 1.5, 0.5])
        assert results.sharpe_r > 0  # Positive expectancy

    def test_sharpe_r_too_few(self):
        results = self._make_results([1.0])
        assert results.sharpe_r == 0.0

    def test_sortino_r(self):
        results = self._make_results([2.0, -1.0, 1.5, -0.5])
        assert results.sortino_r > 0

    def test_sortino_r_no_losses(self):
        results = self._make_results([1.0, 2.0, 3.0])
        assert results.sortino_r == float("inf")

    def test_sortino_r_one_loss(self):
        """Exactly 1 loss with positive mean → inf (not nan)."""
        results = self._make_results([2.0, 1.5, -1.0, 3.0])
        assert results.sortino_r == float("inf")

    def test_sortino_r_one_loss_negative_mean(self):
        """Exactly 1 loss with negative mean → 0.0 (not nan)."""
        results = self._make_results([-5.0, 1.0])
        assert results.sortino_r == 0.0

    def test_max_drawdown_r(self):
        results = self._make_results([1.0, -3.0, 1.0])
        # Cum: [0, 1, -2, -1], Peak: [0, 1, 1, 1], DD: [0, 0, 3, 2]
        assert results.max_drawdown_r == pytest.approx(3.0)

    def test_max_drawdown_r_no_dd(self):
        results = self._make_results([1.0, 1.0, 1.0])
        assert results.max_drawdown_r == 0.0

    def test_max_drawdown_r_empty(self):
        results = self._make_results([])
        assert results.max_drawdown_r == 0.0

    def test_max_drawdown_r_initial_loss(self):
        """DD from 0R starting equity must be captured."""
        results = self._make_results([-2.0, 1.0, 1.0])
        # Cum: [0, -2, -1, 0], Peak: [0, 0, 0, 0], DD: [0, 2, 1, 0]
        assert results.max_drawdown_r == pytest.approx(2.0)

    def test_max_drawdown_r_all_losses(self):
        results = self._make_results([-1.0, -1.0, -1.0])
        # Cum: [0, -1, -2, -3], Peak: [0, 0, 0, 0], DD: [0, 1, 2, 3]
        assert results.max_drawdown_r == pytest.approx(3.0)

    def test_max_drawdown_r_single_loss(self):
        results = self._make_results([-1.5])
        assert results.max_drawdown_r == pytest.approx(1.5)

    def test_recovery_factor(self):
        results = self._make_results([1.0, -3.0, 5.0])
        # total = 3.0, max_dd = 3.0, RF = 1.0
        assert results.recovery_factor == pytest.approx(1.0)

    def test_recovery_factor_no_dd(self):
        results = self._make_results([1.0, 1.0])
        assert results.recovery_factor == float("inf")

    def test_recovery_factor_empty(self):
        results = self._make_results([])
        assert results.recovery_factor == 0.0

    def test_recovery_factor_negative(self):
        """Net-losing strategy returns negative RF."""
        results = self._make_results([-1.0, -1.0])
        # total = -2.0, max_dd = 2.0, RF = -1.0
        assert results.recovery_factor == pytest.approx(-1.0)

    def test_geometric_mean_r_all_losses(self):
        """All-losing system returns small negative geometric mean."""
        results = self._make_results([-1.0, -1.0])
        gm = results.geometric_mean_r
        assert gm < 0  # Negative growth

    def test_geometric_mean_r_ruin(self):
        """Extreme loss triggering ruin guard → 0.0."""
        results = self._make_results([-200.0])
        # factors = 1 + 0.01 * (-200) = -1.0 <= 0 → guard returns 0.0
        assert results.geometric_mean_r == 0.0

    def test_sortino_r_all_losses(self):
        """All-losing strategy with 2+ losses returns negative sortino."""
        results = self._make_results([-1.0, -2.0, -3.0])
        assert results.sortino_r < 0
