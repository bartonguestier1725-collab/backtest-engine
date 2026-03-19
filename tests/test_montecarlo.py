"""Tests for backtest_engine.montecarlo."""

import warnings

import numpy as np
import pytest

from backtest_engine.montecarlo import MonteCarloDD, _mc_shuffle_compound


class TestMCShuffleCompound:
    def test_deterministic_with_seed(self):
        """Same seed should give same results."""
        pnl = np.array([1.0, -1.0, 2.0, -0.5, 1.5, -1.0, 0.5], dtype=np.float64)
        r1 = _mc_shuffle_compound(pnl, 100, 0.01, 42)
        r2 = _mc_shuffle_compound(pnl, 100, 0.01, 42)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seeds_differ(self):
        pnl = np.array([1.0, -1.0, 2.0, -0.5, 1.5, -1.0, 0.5], dtype=np.float64)
        r1 = _mc_shuffle_compound(pnl, 1000, 0.01, 42)
        r2 = _mc_shuffle_compound(pnl, 1000, 0.01, 123)
        # Very unlikely to be exactly equal
        assert not np.array_equal(r1, r2)

    def test_all_winners_low_dd(self):
        """All winning trades should have very low drawdown."""
        pnl = np.full(50, 1.0, dtype=np.float64)
        dds = _mc_shuffle_compound(pnl, 100, 0.01, 42)
        assert np.all(dds == 0.0)  # No drawdown with all winners

    def test_ruin_scenario(self):
        """Very large losses at high risk should cause ruin."""
        pnl = np.full(20, -2.0, dtype=np.float64)
        dds = _mc_shuffle_compound(pnl, 100, 0.5, 42)
        assert np.all(dds == 1.0)  # Total ruin

    def test_dd_range(self):
        """Drawdowns should be in [0, 1]."""
        pnl = np.array([1.0, -1.0, 2.0, -0.5, 1.5, -1.0, 0.5] * 20, dtype=np.float64)
        dds = _mc_shuffle_compound(pnl, 1000, 0.02, 42)
        assert np.all(dds >= 0.0)
        assert np.all(dds <= 1.0)


class TestMonteCarloDD:
    @pytest.fixture
    def profitable_system(self):
        """A profitable system: 60% WR, 1.5 avg win, -1.0 avg loss."""
        np.random.seed(42)
        n = 200
        pnl = np.where(
            np.random.rand(n) < 0.6,
            1.5,
            -1.0,
        ).astype(np.float64)
        return pnl

    def test_run_returns_array(self, profitable_system):
        mc = MonteCarloDD(profitable_system, n_sims=100, seed=42)
        dds = mc.run()
        assert len(dds) == 100
        assert dds.dtype == np.float64

    def test_dd_percentile(self, profitable_system):
        mc = MonteCarloDD(profitable_system, n_sims=1000, seed=42)
        mc.run()
        dd50 = mc.dd_percentile(50.0)
        dd95 = mc.dd_percentile(95.0)
        assert dd95 >= dd50  # Higher percentile = worse DD

    def test_ruin_probability(self, profitable_system):
        mc = MonteCarloDD(profitable_system, n_sims=1000, risk_pct=0.01, seed=42)
        mc.run()
        # Low risk + profitable system → low ruin probability at 50% DD
        ruin = mc.ruin_probability(0.50)
        assert ruin < 0.1

    def test_kelly_fraction(self, profitable_system):
        mc = MonteCarloDD(profitable_system, seed=42)
        kelly = mc.kelly_fraction()
        assert kelly > 0.0  # Profitable system should have positive Kelly
        assert kelly < 1.0  # Should be reasonable

    def test_kelly_all_losses(self):
        pnl = np.full(50, -1.0, dtype=np.float64)
        mc = MonteCarloDD(pnl)
        assert mc.kelly_fraction() == 0.0

    def test_optimal_risk_pct(self, profitable_system):
        mc = MonteCarloDD(profitable_system, n_sims=500, seed=42)
        mc.run()
        opt = mc.optimal_risk_pct(max_dd=0.20, target_pct=95.0)
        assert 0.001 <= opt <= 0.10

    def test_prop_firm_check(self, profitable_system):
        mc = MonteCarloDD(profitable_system, n_sims=500, risk_pct=0.005, seed=42)
        mc.run()
        result = mc.prop_firm_check()
        assert "pass" in result
        assert "dd_confidence" in result
        assert "dd_99" in result
        assert isinstance(result["pass"], bool)
        assert "max_dd_ok" in result
        assert "daily_dd_ok" in result  # backward compat
        assert result["max_dd_ok"] == result["daily_dd_ok"]

    def test_prop_firm_check_custom_limits(self, profitable_system):
        mc = MonteCarloDD(profitable_system, n_sims=500, risk_pct=0.005, seed=42)
        mc.run()
        result = mc.prop_firm_check(max_dd_limit=0.05, total_dd_limit=0.10, confidence=90.0)
        assert "dd_confidence" in result
        assert isinstance(result["max_dd_ok"], bool)
        assert isinstance(result["total_dd_ok"], bool)
        # backward compat: daily_dd_ok should also exist
        assert isinstance(result["daily_dd_ok"], bool)

    def test_fundora_check_deprecated(self, profitable_system):
        mc = MonteCarloDD(profitable_system, n_sims=500, risk_pct=0.005, seed=42)
        mc.run()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = mc.fundora_check()
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 1
        assert "pass" in result
        assert "dd_95" in result  # backward compat key
        assert "dd_confidence" in result  # new key also present
        assert isinstance(result["pass"], bool)

    def test_lazy_run(self, profitable_system):
        """Accessing max_dds without explicit run() should auto-run."""
        mc = MonteCarloDD(profitable_system, n_sims=100, seed=42)
        dds = mc.max_dds  # Should trigger auto-run
        assert len(dds) == 100
