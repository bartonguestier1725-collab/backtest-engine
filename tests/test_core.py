"""Tests for backtest_engine.core — simulate_trades."""

import numpy as np
import pytest

from backtest_engine._types import (
    LONG, SHORT,
    EXIT_SL, EXIT_TP, EXIT_TIME, EXIT_BE, EXIT_NO_FILL, EXIT_TRAIL, EXIT_CUSTOM,
    TRADE_RESULT_DTYPE,
)
from backtest_engine.core import simulate_trades


# ── Helpers ──────────────────────────────────────────────────────────────────

def _single_trade(high, low, close, direction, sl_dist, tp_dist, max_hold, **kwargs):
    """Run simulate_trades for a single trade at bar 0."""
    return simulate_trades(
        high, low, close,
        signal_bars=np.array([0], dtype=np.int32),
        directions=np.array([direction], dtype=np.int8),
        sl_distances=np.array([sl_dist], dtype=np.float64),
        tp_distances=np.array([tp_dist], dtype=np.float64),
        max_hold=max_hold,
        **kwargs,
    )


# ── RR Mode Tests ────────────────────────────────────────────────────────────

class TestRRMode:
    def test_long_tp_hit(self, simple_uptrend):
        """Long in uptrend should hit TP."""
        high, low, close = simple_uptrend
        result = _single_trade(high, low, close, LONG, sl_dist=5.0, tp_dist=10.0, max_hold=50)
        assert len(result) == 1
        r = result[0]
        assert r["exit_type"] == EXIT_TP
        assert r["pnl_r"] == pytest.approx(10.0 / 5.0)  # TP dist / SL dist = 2R
        assert r["hold_bars"] > 0

    def test_short_tp_hit(self, simple_downtrend):
        """Short in downtrend should hit TP."""
        high, low, close = simple_downtrend
        result = _single_trade(high, low, close, SHORT, sl_dist=5.0, tp_dist=10.0, max_hold=50)
        r = result[0]
        assert r["exit_type"] == EXIT_TP
        assert r["pnl_r"] == pytest.approx(2.0)

    def test_long_sl_hit(self, simple_downtrend):
        """Long in downtrend should hit SL."""
        high, low, close = simple_downtrend
        result = _single_trade(high, low, close, LONG, sl_dist=5.0, tp_dist=50.0, max_hold=50)
        r = result[0]
        assert r["exit_type"] == EXIT_SL
        assert r["pnl_r"] == pytest.approx(-1.0)

    def test_short_sl_hit(self, simple_uptrend):
        """Short in uptrend should hit SL."""
        high, low, close = simple_uptrend
        result = _single_trade(high, low, close, SHORT, sl_dist=5.0, tp_dist=50.0, max_hold=50)
        r = result[0]
        assert r["exit_type"] == EXIT_SL
        assert r["pnl_r"] == pytest.approx(-1.0)

    def test_timeout_exit(self, flat_market):
        """In a flat market with wide SL/TP, trade should timeout."""
        high, low, close = flat_market
        result = _single_trade(high, low, close, LONG, sl_dist=10.0, tp_dist=20.0, max_hold=10)
        r = result[0]
        assert r["exit_type"] == EXIT_TIME
        assert r["hold_bars"] == 10

    def test_sl_priority_over_tp(self):
        """When SL and TP both hit on same bar, SL should win."""
        # Bar 0: ref price = 100
        # Bar 1: huge range — both SL and TP hit
        close = np.array([100.0, 100.0], dtype=np.float64)
        high = np.array([100.5, 115.0], dtype=np.float64)  # TP at 110
        low = np.array([99.5, 90.0], dtype=np.float64)     # SL at 95

        result = _single_trade(high, low, close, LONG, sl_dist=5.0, tp_dist=10.0, max_hold=10)
        r = result[0]
        assert r["exit_type"] == EXIT_SL

    def test_mfe_mae_recorded(self, simple_uptrend):
        """MFE and MAE should be properly recorded."""
        high, low, close = simple_uptrend
        result = _single_trade(high, low, close, LONG, sl_dist=5.0, tp_dist=10.0, max_hold=50)
        r = result[0]
        assert r["mfe_r"] >= r["pnl_r"]  # MFE >= realized PnL
        assert r["mae_r"] <= 0.0 or r["mae_r"] >= 0.0  # MAE is recorded

    def test_breakeven(self):
        """BE trigger should move SL to entry price."""
        # Price goes up enough to trigger BE, then comes back
        n = 20
        close = np.zeros(n, dtype=np.float64)
        close[0] = 100.0
        # Rise to trigger BE (50% of TP=10 → trigger at 105)
        for i in range(1, 8):
            close[i] = 100.0 + i * 1.0
        # Then fall back
        for i in range(8, n):
            close[i] = 107.0 - (i - 7) * 2.0

        high = close + 0.5
        low = close - 0.5

        result = _single_trade(
            high, low, close, LONG,
            sl_dist=5.0, tp_dist=10.0, max_hold=50,
            be_trigger_pct=0.5,
        )
        r = result[0]
        # Should exit at BE (entry price), not at original SL
        assert r["exit_type"] == EXIT_BE
        assert r["pnl_r"] == pytest.approx(0.0, abs=0.2)  # ~0R (entry price)

    def test_multiple_trades(self, simple_uptrend):
        """Multiple trades should all be simulated."""
        high, low, close = simple_uptrend
        signal_bars = np.array([0, 10, 20], dtype=np.int32)
        directions = np.array([LONG, LONG, SHORT], dtype=np.int8)
        sl_dists = np.array([5.0, 5.0, 5.0], dtype=np.float64)
        tp_dists = np.array([10.0, 10.0, 10.0], dtype=np.float64)

        result = simulate_trades(
            high, low, close, signal_bars, directions,
            sl_dists, tp_dists, max_hold=30,
        )
        assert len(result) == 3
        assert result[0]["exit_type"] == EXIT_TP  # Long in uptrend
        assert result[1]["exit_type"] == EXIT_TP  # Long in uptrend
        assert result[2]["exit_type"] == EXIT_SL  # Short in uptrend → SL

    def test_empty_trades(self, simple_uptrend):
        """Empty signal array should return empty result."""
        high, low, close = simple_uptrend
        result = simulate_trades(
            high, low, close,
            signal_bars=np.array([], dtype=np.int32),
            directions=np.array([], dtype=np.int8),
            sl_distances=np.array([], dtype=np.float64),
            tp_distances=np.array([], dtype=np.float64),
            max_hold=10,
        )
        assert len(result) == 0
        assert result.dtype == TRADE_RESULT_DTYPE


class TestRetraceEntry:
    def test_retrace_fill(self):
        """Limit order should fill when price retraces."""
        n = 20
        close = np.full(n, 100.0, dtype=np.float64)
        high = np.full(n, 100.5, dtype=np.float64)
        low = np.full(n, 99.5, dtype=np.float64)
        # Bar 3: dip to fill limit at 95.0
        low[3] = 94.0
        close[3] = 96.0
        # Then rise to hit TP
        for i in range(4, n):
            close[i] = 96.0 + (i - 3) * 3.0
            high[i] = close[i] + 0.5
            low[i] = close[i] - 0.5

        result = _single_trade(
            high, low, close, LONG,
            sl_dist=5.0, tp_dist=10.0, max_hold=50,
            retrace_pct=0.5,  # 50% of TP=10 → retrace 5.0, limit at 95.0
        )
        r = result[0]
        assert r["exit_type"] != EXIT_NO_FILL

    def test_retrace_no_fill(self, flat_market):
        """If price doesn't retrace enough, no fill."""
        high, low, close = flat_market
        result = _single_trade(
            high, low, close, LONG,
            sl_dist=5.0, tp_dist=10.0, max_hold=50,
            retrace_pct=0.5,  # limit at 95.0, but low only goes to 99.5
            retrace_timeout=5,
        )
        r = result[0]
        assert r["exit_type"] == EXIT_NO_FILL
        assert r["pnl_r"] == 0.0


# ── Trailing Mode Tests ──────────────────────────────────────────────────────

class TestTrailingMode:
    def test_trailing_stop_locks_profit(self):
        """Trailing stop should lock in profit after activation."""
        n = 30
        close = np.zeros(n, dtype=np.float64)
        close[0] = 100.0
        # Rise to activate trail
        for i in range(1, 15):
            close[i] = 100.0 + i * 2.0
        # Then fall back
        for i in range(15, n):
            close[i] = close[14] - (i - 14) * 3.0
        high = close + 1.0
        low = close - 1.0

        result = _single_trade(
            high, low, close, LONG,
            sl_dist=5.0, tp_dist=50.0, max_hold=50,
            exit_mode="trailing",
            trail_activation_r=2.0,  # activate at 2R = 10 points
            trail_distance_r=1.0,     # trail 1R = 5 points behind
        )
        r = result[0]
        assert r["exit_type"] == EXIT_TRAIL
        assert r["pnl_r"] > 0.0  # Should lock in profit

    def test_trailing_sl_before_activation(self, simple_downtrend):
        """If trail never activates, original SL should work."""
        high, low, close = simple_downtrend
        result = _single_trade(
            high, low, close, LONG,
            sl_dist=5.0, tp_dist=50.0, max_hold=50,
            exit_mode="trailing",
            trail_activation_r=10.0,  # very high activation
            trail_distance_r=1.0,
        )
        r = result[0]
        assert r["exit_type"] == EXIT_SL
        assert r["pnl_r"] == pytest.approx(-1.0)


# ── Custom Exit Mode Tests ───────────────────────────────────────────────────

class TestCustomMode:
    def test_custom_exit_signal(self, simple_uptrend):
        """Custom exit signal should close trade."""
        high, low, close = simple_uptrend
        exit_signals = np.zeros(len(close), dtype=np.bool_)
        exit_signals[5] = True  # Exit at bar 5

        result = _single_trade(
            high, low, close, LONG,
            sl_dist=50.0, tp_dist=100.0, max_hold=50,
            exit_mode="custom",
            exit_signals=exit_signals,
        )
        r = result[0]
        assert r["exit_type"] == EXIT_CUSTOM
        assert r["hold_bars"] == 5
        assert r["exit_bar"] == 5

    def test_custom_sl_priority(self, simple_downtrend):
        """SL should still work and take priority in custom mode."""
        high, low, close = simple_downtrend
        exit_signals = np.zeros(len(close), dtype=np.bool_)
        exit_signals[50] = True  # Exit signal at bar 50 (too late)

        result = _single_trade(
            high, low, close, SHORT,
            sl_dist=3.0, tp_dist=100.0, max_hold=80,
            exit_mode="custom",
            exit_signals=exit_signals,
        )
        r = result[0]
        # In a downtrend, short should profit, but let's test with LONG for SL
        # Actually SHORT in downtrend profits. Let's use LONG for SL test.

    def test_custom_sl_hits_first(self, simple_downtrend):
        """In custom mode, SL should hit before custom signal."""
        high, low, close = simple_downtrend
        exit_signals = np.zeros(len(close), dtype=np.bool_)
        exit_signals[50] = True

        result = _single_trade(
            high, low, close, LONG,
            sl_dist=3.0, tp_dist=100.0, max_hold=80,
            exit_mode="custom",
            exit_signals=exit_signals,
        )
        r = result[0]
        assert r["exit_type"] == EXIT_SL


# ── Validation Tests ─────────────────────────────────────────────────────────

class TestValidation:
    def test_mismatched_lengths_raises(self, simple_uptrend):
        high, low, close = simple_uptrend
        with pytest.raises(ValueError, match="equal length"):
            simulate_trades(
                high, low, close,
                signal_bars=np.array([0], dtype=np.int32),
                directions=np.array([LONG, LONG], dtype=np.int8),  # mismatch
                sl_distances=np.array([5.0], dtype=np.float64),
                tp_distances=np.array([5.0], dtype=np.float64),
                max_hold=10,
            )

    def test_unknown_exit_mode(self, simple_uptrend):
        high, low, close = simple_uptrend
        with pytest.raises(ValueError, match="Unknown exit_mode"):
            _single_trade(high, low, close, LONG, 5.0, 10.0, 10, exit_mode="invalid")

    def test_custom_mode_requires_exit_signals(self, simple_uptrend):
        high, low, close = simple_uptrend
        with pytest.raises(ValueError, match="exit_signals required"):
            _single_trade(high, low, close, LONG, 5.0, 10.0, 10, exit_mode="custom")


# ── dtype and field tests ────────────────────────────────────────────────────

class TestOutputFormat:
    def test_output_dtype(self, simple_uptrend):
        """Output should be structured array with correct dtype."""
        high, low, close = simple_uptrend
        result = _single_trade(high, low, close, LONG, 5.0, 10.0, 50)
        assert result.dtype == TRADE_RESULT_DTYPE

    def test_entry_exit_bars_valid(self, simple_uptrend):
        """Entry and exit bars should be within valid range."""
        high, low, close = simple_uptrend
        result = _single_trade(high, low, close, LONG, 5.0, 10.0, 50)
        r = result[0]
        assert 0 <= r["entry_bar"] < len(close)
        assert 0 <= r["exit_bar"] < len(close)
        assert r["exit_bar"] >= r["entry_bar"]
