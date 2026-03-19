"""Tests for backtest_engine.validation — WalkForward and CSCV."""

import numpy as np
import pytest

from backtest_engine.validation import WalkForward, CSCV


# ── Helpers ──────────────────────────────────────────────────────────────────

def _dummy_evaluate(params, start, end):
    """Dummy evaluate function: returns param['alpha'] * number_of_bars."""
    return params["alpha"] * (end - start)


def _noisy_evaluate(params, start, end):
    """Evaluation with noise based on position — simulates overfitting."""
    np.random.seed(start + int(params["alpha"] * 1000))
    signal = params["alpha"] * (end - start)
    noise = np.random.randn() * 10
    return signal + noise


# ── WalkForward Tests ────────────────────────────────────────────────────────

class TestWalkForward:
    def test_rolling_splits(self):
        wf = WalkForward(n_bars=1000, is_ratio=0.7, n_splits=5, anchored=False)
        assert len(wf.splits) == 5
        for (is_s, is_e), (oos_s, oos_e) in wf.splits:
            assert is_s < is_e
            assert oos_s < oos_e
            assert is_e == oos_s  # IS ends where OOS starts

    def test_anchored_splits(self):
        wf = WalkForward(n_bars=1000, is_ratio=0.7, n_splits=5, anchored=True)
        assert len(wf.splits) > 0
        for (is_s, is_e), (oos_s, oos_e) in wf.splits:
            assert is_s == 0  # Anchored: IS always starts at 0
            assert is_e == oos_s

    def test_run_basic(self):
        param_grid = [
            {"alpha": 0.1},
            {"alpha": 0.5},
            {"alpha": 1.0},
        ]
        wf = WalkForward(n_bars=500, is_ratio=0.7, n_splits=3)
        result = wf.run(param_grid, _dummy_evaluate)

        assert "splits" in result
        assert "oos_metrics" in result
        assert "oos_mean" in result
        assert len(result["splits"]) == 3
        assert len(result["oos_metrics"]) == 3

    def test_best_param_selected(self):
        """The best IS param should be used for OOS evaluation."""
        param_grid = [
            {"alpha": 0.1},
            {"alpha": 1.0},  # This should always be best (linear eval)
        ]
        wf = WalkForward(n_bars=500, is_ratio=0.7, n_splits=3)
        result = wf.run(param_grid, _dummy_evaluate)

        for split in result["splits"]:
            assert split["best_params"]["alpha"] == 1.0

    def test_oos_positive_frac(self):
        param_grid = [{"alpha": 1.0}]
        wf = WalkForward(n_bars=500, is_ratio=0.7, n_splits=3)
        result = wf.run(param_grid, _dummy_evaluate)
        assert result["oos_positive_frac"] == 1.0  # All positive with alpha=1.0

    def test_is_metrics_returned(self):
        """run() should return IS metrics alongside OOS metrics."""
        param_grid = [{"alpha": 0.5}, {"alpha": 1.0}]
        wf = WalkForward(n_bars=500, is_ratio=0.7, n_splits=3)
        result = wf.run(param_grid, _dummy_evaluate)
        assert "is_metrics" in result
        assert "is_mean" in result
        assert "is_std" in result
        assert len(result["is_metrics"]) == 3
        assert result["is_mean"] > 0

    def test_is_oos_ratio(self):
        """Deterministic strategy should have is_oos_ratio close to 1."""
        param_grid = [{"alpha": 1.0}]
        wf = WalkForward(n_bars=500, is_ratio=0.7, n_splits=3)
        result = wf.run(param_grid, _dummy_evaluate)
        assert "is_oos_ratio" in result
        # For a deterministic linear function, OOS should be proportional to IS
        # Tighter bounds: ratio should be close to 1.0 for a stable strategy
        assert 0.1 < result["is_oos_ratio"] < 3.0


# ── CSCV Tests ───────────────────────────────────────────────────────────────

class TestCSCV:
    def test_even_splits_required(self):
        with pytest.raises(ValueError, match="even"):
            CSCV(n_splits=7)

    def test_basic_run(self):
        param_grid = [
            {"alpha": 0.1},
            {"alpha": 0.5},
            {"alpha": 1.0},
        ]
        cscv = CSCV(n_splits=10)
        result = cscv.run(param_grid, _dummy_evaluate, n_bars=1000)

        assert "pbo" in result
        assert "logit_distribution" in result
        assert "n_combinations" in result
        # C(10,5) = 252
        assert result["n_combinations"] == 252
        assert 0.0 <= result["pbo"] <= 1.0

    def test_no_overfitting_with_stable_strategy(self):
        """A deterministic strategy should not show overfitting."""
        param_grid = [
            {"alpha": 0.1},
            {"alpha": 0.5},
            {"alpha": 1.0},  # Always best
        ]
        cscv = CSCV(n_splits=10)
        result = cscv.run(param_grid, _dummy_evaluate, n_bars=1000)
        # With deterministic evaluation, best IS param should also be best OOS
        assert result["pbo"] == 0.0

    def test_single_param(self):
        """Single param should have PBO = 0 (no overfitting possible)."""
        param_grid = [{"alpha": 1.0}]
        cscv = CSCV(n_splits=10)
        result = cscv.run(param_grid, _dummy_evaluate, n_bars=1000)
        assert result["pbo"] == 0.0

    def test_logit_distribution_shape(self):
        param_grid = [{"alpha": float(i)} for i in range(5)]
        cscv = CSCV(n_splits=6)
        result = cscv.run(param_grid, _noisy_evaluate, n_bars=600)
        # C(6,3) = 20
        assert result["n_combinations"] == 20
        assert len(result["logit_distribution"]) == 20

    def test_small_splits(self):
        """Minimum case: 2 splits."""
        param_grid = [{"alpha": 0.5}, {"alpha": 1.0}]
        cscv = CSCV(n_splits=2)
        result = cscv.run(param_grid, _dummy_evaluate, n_bars=100)
        # C(2,1) = 2
        assert result["n_combinations"] == 2
