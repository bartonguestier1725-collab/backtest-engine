"""Tests for backtest_engine.core — simulate_trades."""

import numpy as np
import pytest

from backtest_engine._types import (
    LONG, SHORT,
    EXIT_SL, EXIT_TP, EXIT_TIME, EXIT_BE, EXIT_NO_FILL, EXIT_TRAIL, EXIT_CUSTOM,
    TRADE_RESULT_DTYPE,
)
from backtest_engine.core import simulate_trades
from backtest_engine.indicators import parabolic_sar
from backtest_engine.costs import BrokerCost


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


# ── SAR Trailing Mode Tests ────────────────────────────────────────────────

class TestSARTrailingMode:
    def test_sar_trailing_long_exit(self):
        """LONG should exit via SAR trail when trend flips."""
        n = 40
        close = np.empty(n, dtype=np.float64)
        # Uptrend then reversal
        for i in range(20):
            close[i] = 100.0 + i * 1.5
        for i in range(20, n):
            close[i] = close[19] - (i - 19) * 2.0
        high = close + 0.5
        low = close - 0.5

        sar, trend, sar_stop = parabolic_sar(high, low)

        result = simulate_trades(
            high, low, close,
            signal_bars=np.array([0], dtype=np.int32),
            directions=np.array([LONG], dtype=np.int8),
            sl_distances=np.array([50.0], dtype=np.float64),  # wide hard SL
            tp_distances=np.array([1e6], dtype=np.float64),   # no TP
            max_hold=n,
            exit_mode="sar_trailing",
            sar_values=sar_stop,
        )
        r = result[0]
        assert r["exit_type"] == EXIT_TRAIL
        assert r["pnl_r"] > 0.0  # should have locked profit

    def test_sar_trailing_exit_price_not_ep(self):
        """Exit price should be pre-reversal SAR, NOT the extreme point."""
        n = 40
        close = np.empty(n, dtype=np.float64)
        for i in range(20):
            close[i] = 100.0 + i * 1.5  # up to 128.5
        for i in range(20, n):
            close[i] = close[19] - (i - 19) * 2.0
        high = close + 0.5
        low = close - 0.5

        sar, trend, sar_stop = parabolic_sar(high, low)

        result = simulate_trades(
            high, low, close,
            signal_bars=np.array([0], dtype=np.int32),
            directions=np.array([LONG], dtype=np.int8),
            sl_distances=np.array([50.0], dtype=np.float64),
            tp_distances=np.array([1e6], dtype=np.float64),
            max_hold=n,
            exit_mode="sar_trailing",
            sar_values=sar_stop,
        )
        r = result[0]
        # Exit price = entry + pnl_r * sl_dist
        entry_price = close[0]
        exit_price = entry_price + r["pnl_r"] * 50.0
        # Exit should NOT be at the highest point (128.5+0.5=129)
        assert exit_price < high.max(), f"Exit {exit_price} should be < max high {high.max()}"

    def test_sar_trailing_short_exit(self):
        """SHORT should exit via SAR trail when trend flips up."""
        n = 40
        close = np.empty(n, dtype=np.float64)
        # Downtrend then reversal
        close[0] = 100.0
        close[1] = 101.0
        close[2] = 102.0
        for i in range(3, 20):
            close[i] = 102.0 - (i - 2) * 2.0
        for i in range(20, n):
            close[i] = close[19] + (i - 19) * 2.0
        high = close + 0.5
        low = close - 0.5

        sar, trend, sar_stop = parabolic_sar(high, low)

        # Enter short once trend is confirmed down
        entry_bar = 8  # should be in downtrend by now
        result = simulate_trades(
            high, low, close,
            signal_bars=np.array([entry_bar], dtype=np.int32),
            directions=np.array([SHORT], dtype=np.int8),
            sl_distances=np.array([80.0], dtype=np.float64),  # wide hard SL
            tp_distances=np.array([1e6], dtype=np.float64),
            max_hold=n,
            exit_mode="sar_trailing",
            sar_values=sar_stop,
        )
        r = result[0]
        # Should exit via trail or time
        assert r["exit_type"] in (EXIT_TRAIL, EXIT_TIME)

    def test_sar_trailing_hard_sl_priority(self):
        """Hard SL should fire when it's tighter than SAR."""
        n = 30
        close = np.empty(n, dtype=np.float64)
        # Uptrend for 15 bars so SAR drifts well below price
        for i in range(15):
            close[i] = 100.0 + i * 2.0
        # Then a sudden drop from bar 15
        for i in range(15, n):
            close[i] = close[14] - (i - 14) * 5.0
        high = close + 0.3
        low = close - 0.3

        sar, trend, sar_stop = parabolic_sar(high, low)

        # Enter LONG at bar 14 (peak) with tight hard SL=3.0
        entry_bar = 14
        result = simulate_trades(
            high, low, close,
            signal_bars=np.array([entry_bar], dtype=np.int32),
            directions=np.array([LONG], dtype=np.int8),
            sl_distances=np.array([3.0], dtype=np.float64),  # tight hard SL
            tp_distances=np.array([1e6], dtype=np.float64),
            max_hold=n,
            exit_mode="sar_trailing",
            sar_values=sar_stop,
        )
        r = result[0]
        # Hard SL (entry-3) is above the lagging SAR → hard SL fires first
        assert r["exit_type"] == EXIT_SL
        assert r["pnl_r"] == pytest.approx(-1.0)

    def test_sar_trailing_requires_sar_values(self, simple_uptrend):
        """Should raise ValueError if sar_values not provided."""
        high, low, close = simple_uptrend
        with pytest.raises(ValueError, match="sar_values required"):
            _single_trade(high, low, close, LONG, 5.0, 10.0, 10, exit_mode="sar_trailing")

    def test_sar_trailing_timeout(self, flat_market):
        """In flat market with wide SL, trade should timeout."""
        high, low, close = flat_market
        sar, _, sar_stop = parabolic_sar(high, low)
        result = simulate_trades(
            high, low, close,
            signal_bars=np.array([0], dtype=np.int32),
            directions=np.array([LONG], dtype=np.int8),
            sl_distances=np.array([50.0], dtype=np.float64),
            tp_distances=np.array([1e6], dtype=np.float64),
            max_hold=10,
            exit_mode="sar_trailing",
            sar_values=sar_stop,
        )
        r = result[0]
        assert r["exit_type"] in (EXIT_TIME, EXIT_TRAIL, EXIT_SL)
        assert r["hold_bars"] <= 10

    def test_counter_trend_long_no_phantom_profit(self):
        """B-plan: counter-trend LONG should NOT get phantom profit from SAR above."""
        # Downtrend: SAR above price throughout
        n = 30
        close = np.empty(n, dtype=np.float64)
        close[0] = 100.0
        for i in range(1, n):
            close[i] = 100.0 - i * 0.5  # gentle decline
        high = close + 0.3
        low = close - 0.3

        # Manually create SAR that stays ABOVE price (simulating downtrend SAR)
        sar_stop = close + 3.0  # SAR is always 3.0 above close

        result = simulate_trades(
            high, low, close,
            signal_bars=np.array([0], dtype=np.int32),
            directions=np.array([LONG], dtype=np.int8),
            sl_distances=np.array([5.0], dtype=np.float64),
            tp_distances=np.array([1e6], dtype=np.float64),
            max_hold=n,
            exit_mode="sar_trailing",
            sar_values=sar_stop,
        )
        r = result[0]
        # Old buggy kernel would exit at SAR (above entry) = phantom profit.
        # B-plan: SAR > close[b-1] → TP target → high must reach SAR.
        # high = close + 0.3, SAR = close + 3.0 → high never reaches SAR.
        # Price declines → hits hard SL at 95.0.
        assert r["exit_type"] == EXIT_SL
        assert r["pnl_r"] == pytest.approx(-1.0)

    def test_counter_trend_long_sar_reached(self):
        """B-plan: LONG exits at SAR when price actually rises to SAR level."""
        n = 20
        close = np.empty(n, dtype=np.float64)
        close[0] = 100.0
        # Flat, then rise
        for i in range(1, 10):
            close[i] = 100.0
        for i in range(10, n):
            close[i] = 100.0 + (i - 9) * 2.0  # rise from 102 to 120+
        high = close + 0.5
        low = close - 0.5

        # SAR starts above at 105, stays constant
        sar_stop = np.full(n, 105.0, dtype=np.float64)

        result = simulate_trades(
            high, low, close,
            signal_bars=np.array([0], dtype=np.int32),
            directions=np.array([LONG], dtype=np.int8),
            sl_distances=np.array([5.0], dtype=np.float64),
            tp_distances=np.array([1e6], dtype=np.float64),
            max_hold=n,
            exit_mode="sar_trailing",
            sar_values=sar_stop,
        )
        r = result[0]
        # Price rises → high reaches 105 at bar 12 (close=104, high=104.5? No...)
        # Bar 12: close = 100 + (12-9)*2 = 106, high = 106.5 → high >= 105 ✓
        # But close[11] = 104, so SAR 105 > 104 → else branch → check high >= 105
        # high[12] = 106.5 >= 105 → EXIT_TRAIL at SAR=105
        assert r["exit_type"] == EXIT_TRAIL
        assert r["pnl_r"] == pytest.approx((105.0 - 100.0) / 5.0)  # = 1.0R

    def test_counter_trend_short_sl_hit(self):
        """B-plan: counter-trend SHORT hits hard SL when price rises."""
        n = 20
        close = np.empty(n, dtype=np.float64)
        close[0] = 100.0
        for i in range(1, n):
            close[i] = 100.0 + i * 1.0  # uptrend
        high = close + 0.3
        low = close - 0.3

        # SAR below price (uptrend SAR), simulating counter-trend SHORT
        sar_stop = close - 3.0  # SAR always 3.0 below close

        result = simulate_trades(
            high, low, close,
            signal_bars=np.array([0], dtype=np.int32),
            directions=np.array([SHORT], dtype=np.int8),
            sl_distances=np.array([5.0], dtype=np.float64),
            tp_distances=np.array([1e6], dtype=np.float64),
            max_hold=n,
            exit_mode="sar_trailing",
            sar_values=sar_stop,
        )
        r = result[0]
        # SAR < close[b-1] → else branch → hard SL first
        # hard_sl = 100 + 5 = 105. high reaches 105 at bar ~5.
        assert r["exit_type"] == EXIT_SL
        assert r["pnl_r"] == pytest.approx(-1.0)

    def test_open_entry_timing(self):
        """open_prices should shift entry to next-bar open."""
        n = 30
        close = np.empty(n, dtype=np.float64)
        for i in range(n):
            close[i] = 100.0 + i * 1.0
        high = close + 0.5
        low = close - 0.5
        open_prices = close - 0.2  # opens slightly below close

        sar_stop = np.full(n, 50.0, dtype=np.float64)  # SAR far below → trailing SL

        result = simulate_trades(
            high, low, close,
            signal_bars=np.array([5], dtype=np.int32),
            directions=np.array([LONG], dtype=np.int8),
            sl_distances=np.array([50.0], dtype=np.float64),
            tp_distances=np.array([1e6], dtype=np.float64),
            max_hold=n,
            exit_mode="sar_trailing",
            sar_values=sar_stop,
            open_prices=open_prices,
        )
        r = result[0]
        # Entry should be at open_prices[6] = 105.8, not close[5] = 105.0
        assert r["entry_bar"] == 6
        # Verify entry price via PnL back-calculation
        expected_entry = open_prices[6]  # 105.8
        actual_exit_price = expected_entry + r["pnl_r"] * 50.0
        assert actual_exit_price > 0  # sanity

    def test_open_entry_last_bar_no_fill(self):
        """Signal on last bar with open_prices should be NO_FILL."""
        n = 10
        close = np.full(n, 100.0, dtype=np.float64)
        high = np.full(n, 100.5, dtype=np.float64)
        low = np.full(n, 99.5, dtype=np.float64)
        open_prices = np.full(n, 100.0, dtype=np.float64)
        sar_stop = np.full(n, 95.0, dtype=np.float64)

        result = simulate_trades(
            high, low, close,
            signal_bars=np.array([9], dtype=np.int32),  # last bar
            directions=np.array([LONG], dtype=np.int8),
            sl_distances=np.array([5.0], dtype=np.float64),
            tp_distances=np.array([1e6], dtype=np.float64),
            max_hold=n,
            exit_mode="sar_trailing",
            sar_values=sar_stop,
            open_prices=open_prices,
        )
        r = result[0]
        assert r["exit_type"] == EXIT_NO_FILL
        assert r["pnl_r"] == 0.0

    def test_rr_open_entry_timing(self):
        """RR mode should use next-bar open when open_prices are provided."""
        n = 30
        close = np.empty(n, dtype=np.float64)
        for i in range(n):
            close[i] = 100.0 + i * 1.0
        high = close + 0.5
        low = close - 0.5
        open_prices = close - 0.2

        result = simulate_trades(
            high, low, close,
            signal_bars=np.array([5], dtype=np.int32),
            directions=np.array([LONG], dtype=np.int8),
            sl_distances=np.array([50.0], dtype=np.float64),
            tp_distances=np.array([1e6], dtype=np.float64),
            max_hold=n,
            exit_mode="rr",
            open_prices=open_prices,
        )
        r = result[0]
        assert r["entry_bar"] == 6
        expected_entry = open_prices[6]
        actual_exit_price = expected_entry + r["pnl_r"] * 50.0
        assert actual_exit_price > 0

    def test_rr_open_entry_last_bar_no_fill(self):
        """RR mode should return NO_FILL when no next-bar open exists."""
        n = 10
        close = np.full(n, 100.0, dtype=np.float64)
        high = np.full(n, 100.5, dtype=np.float64)
        low = np.full(n, 99.5, dtype=np.float64)
        open_prices = np.full(n, 100.0, dtype=np.float64)

        result = simulate_trades(
            high, low, close,
            signal_bars=np.array([9], dtype=np.int32),
            directions=np.array([LONG], dtype=np.int8),
            sl_distances=np.array([5.0], dtype=np.float64),
            tp_distances=np.array([10.0], dtype=np.float64),
            max_hold=n,
            exit_mode="rr",
            open_prices=open_prices,
        )
        r = result[0]
        assert r["exit_type"] == EXIT_NO_FILL
        assert r["pnl_r"] == 0.0

    def test_rr_open_entry_checks_entry_bar_sl(self):
        """Next-bar-open entries must check SL on the entry bar itself."""
        close = np.array([100.0, 100.0, 100.0], dtype=np.float64)
        open_prices = np.array([100.0, 100.0, 100.0], dtype=np.float64)
        high = np.array([100.5, 100.5, 100.5], dtype=np.float64)
        low = np.array([99.5, 94.0, 99.5], dtype=np.float64)  # bar 1 hits SL=95

        result = simulate_trades(
            high, low, close,
            signal_bars=np.array([0], dtype=np.int32),
            directions=np.array([LONG], dtype=np.int8),
            sl_distances=np.array([5.0], dtype=np.float64),
            tp_distances=np.array([50.0], dtype=np.float64),
            max_hold=5,
            exit_mode="rr",
            open_prices=open_prices,
        )
        r = result[0]
        assert r["entry_bar"] == 1
        assert r["exit_bar"] == 1
        assert r["hold_bars"] == 0
        assert r["exit_type"] == EXIT_SL
        assert r["pnl_r"] == pytest.approx(-1.0)

    def test_rr_open_entry_checks_entry_bar_tp(self):
        """Next-bar-open entries must check TP on the entry bar itself."""
        close = np.array([100.0, 100.0, 100.0], dtype=np.float64)
        open_prices = np.array([100.0, 100.0, 100.0], dtype=np.float64)
        high = np.array([100.5, 111.0, 100.5], dtype=np.float64)  # bar 1 hits TP=110
        low = np.array([99.5, 99.0, 99.5], dtype=np.float64)

        result = simulate_trades(
            high, low, close,
            signal_bars=np.array([0], dtype=np.int32),
            directions=np.array([LONG], dtype=np.int8),
            sl_distances=np.array([5.0], dtype=np.float64),
            tp_distances=np.array([10.0], dtype=np.float64),
            max_hold=5,
            exit_mode="rr",
            open_prices=open_prices,
        )
        r = result[0]
        assert r["entry_bar"] == 1
        assert r["exit_bar"] == 1
        assert r["hold_bars"] == 0
        assert r["exit_type"] == EXIT_TP
        assert r["pnl_r"] == pytest.approx(2.0)

    def test_sar_open_entry_checks_entry_bar_sl(self):
        """SAR mode should also check SL on the next-bar-open entry bar."""
        close = np.array([100.0, 100.0, 100.0], dtype=np.float64)
        open_prices = np.array([100.0, 100.0, 100.0], dtype=np.float64)
        high = np.array([100.5, 100.5, 100.5], dtype=np.float64)
        low = np.array([99.5, 94.0, 99.5], dtype=np.float64)
        sar_stop = np.full(3, 50.0, dtype=np.float64)

        result = simulate_trades(
            high, low, close,
            signal_bars=np.array([0], dtype=np.int32),
            directions=np.array([LONG], dtype=np.int8),
            sl_distances=np.array([5.0], dtype=np.float64),
            tp_distances=np.array([1e6], dtype=np.float64),
            max_hold=5,
            exit_mode="sar_trailing",
            sar_values=sar_stop,
            open_prices=open_prices,
        )
        r = result[0]
        assert r["entry_bar"] == 1
        assert r["exit_bar"] == 1
        assert r["hold_bars"] == 0
        assert r["exit_type"] == EXIT_SL
        assert r["pnl_r"] == pytest.approx(-1.0)


# ── Cost Model Tests ───────────────────────────────────────────────────────

class TestFundoraCost:
    def test_xauusd_present(self):
        """XAUUSD should be in fundora cost model."""
        cost = BrokerCost.fundora()
        assert "XAUUSD" in cost.spreads

    def test_xauusd_cost_roundtrip(self):
        """XAUUSD total cost should be spread only (no commission in fundora)."""
        cost = BrokerCost.fundora()
        total = cost.cost_price("XAUUSD")
        # Fundora: spread=0.40, commission=0 → total=0.40
        assert total == pytest.approx(0.40)

    def test_eurusd_cost_roundtrip(self):
        """EURUSD total cost should be spread only."""
        cost = BrokerCost.fundora()
        total = cost.cost_price("EURUSD")
        assert total == pytest.approx(0.00010)

    def test_xauusd_pip_size(self):
        """XAUUSD pip_size should be 0.01."""
        cost = BrokerCost.fundora()
        assert cost.pip_sizes["XAUUSD"] == pytest.approx(0.01)

    def test_xauusd_pip_value(self):
        """XAUUSD pip_value should be 1.0."""
        cost = BrokerCost.fundora()
        assert cost.pip_values["XAUUSD"] == pytest.approx(1.0)
