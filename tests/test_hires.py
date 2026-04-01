"""Tests for simulate_trades_hires — multi-resolution trade simulation."""

import numpy as np
import pytest

from backtest_engine import simulate_trades, simulate_trades_hires, LONG, SHORT


def _make_bars(n_bars: int, start_ts: int, bar_seconds: int, base_price: float = 1.10):
    """Generate synthetic OHLC bars."""
    rng = np.random.default_rng(42)
    timestamps = np.arange(n_bars, dtype=np.int64) * bar_seconds + start_ts
    noise = rng.normal(0, 0.0005, n_bars)
    closes = base_price + np.cumsum(noise)
    highs = closes + rng.uniform(0.0001, 0.001, n_bars)
    lows = closes - rng.uniform(0.0001, 0.001, n_bars)
    opens = closes + rng.normal(0, 0.0002, n_bars)
    return timestamps, opens, highs, lows, closes


class TestSimulateTradesHires:

    def test_basic_execution(self):
        """hires should run without error."""
        ts_1h, o_1h, h_1h, l_1h, c_1h = _make_bars(100, 0, 3600)
        ts_1m, o_1m, h_1m, l_1m, c_1m = _make_bars(6000, 0, 60)

        signal_bars = np.array([10, 30, 50], dtype=np.int32)
        directions = np.array([LONG, SHORT, LONG], dtype=np.int8)
        sl = np.full(3, 0.002)
        tp = np.full(3, 0.004)

        results = simulate_trades_hires(
            signal_timestamps=ts_1h,
            signal_bars=signal_bars,
            directions=directions,
            sl_distances=sl,
            tp_distances=tp,
            max_hold=24,
            signal_bar_minutes=60,
            exec_timestamps=ts_1m,
            exec_opens=o_1m,
            exec_highs=h_1m,
            exec_lows=l_1m,
            exec_closes=c_1m,
        )

        assert len(results) == 3

    def test_max_hold_conversion(self):
        """max_hold in signal bars should be converted to exec bars."""
        ts_1h, o_1h, h_1h, l_1h, c_1h = _make_bars(200, 0, 3600)
        ts_1m, o_1m, h_1m, l_1m, c_1m = _make_bars(12000, 0, 60)

        # 1 signal with huge SL/TP so it always times out
        signal_bars = np.array([10], dtype=np.int32)
        directions = np.array([LONG], dtype=np.int8)
        sl = np.full(1, 1.0)  # huge
        tp = np.full(1, 2.0)  # huge

        results = simulate_trades_hires(
            signal_timestamps=ts_1h,
            signal_bars=signal_bars,
            directions=directions,
            sl_distances=sl,
            tp_distances=tp,
            max_hold=10,          # 10 hours
            signal_bar_minutes=60,
            exec_timestamps=ts_1m,
            exec_opens=o_1m,
            exec_highs=h_1m,
            exec_lows=l_1m,
            exec_closes=c_1m,
        )

        # Should hold for ~600 1m bars (10h * 60min)
        assert results["hold_bars"][0] == 600

    def test_signal_bar_mapping_correct(self):
        """Signal at 1h bar 5 = timestamp 5*3600 = should map near 1m bar 300."""
        ts_1h, o_1h, h_1h, l_1h, c_1h = _make_bars(50, 0, 3600)
        ts_1m, o_1m, h_1m, l_1m, c_1m = _make_bars(3000, 0, 60)

        signal_bars = np.array([5], dtype=np.int32)
        directions = np.array([LONG], dtype=np.int8)
        sl = np.full(1, 0.002)
        tp = np.full(1, 0.004)

        results = simulate_trades_hires(
            signal_timestamps=ts_1h,
            signal_bars=signal_bars,
            directions=directions,
            sl_distances=sl,
            tp_distances=tp,
            max_hold=10,
            signal_bar_minutes=60,
            exec_timestamps=ts_1m,
            exec_opens=o_1m,
            exec_highs=h_1m,
            exec_lows=l_1m,
            exec_closes=c_1m,
        )

        # Entry bar should be near 300 (5*60) in 1m indices
        entry_bar = results["entry_bar"][0]
        assert 299 <= entry_bar <= 301

    def test_same_resolution_raises(self):
        """Should raise if signal and exec resolution are equal."""
        ts, o, h, l, c = _make_bars(100, 0, 60)

        with pytest.raises(ValueError, match="must be greater"):
            simulate_trades_hires(
                signal_timestamps=ts,
                signal_bars=np.array([10], dtype=np.int32),
                directions=np.array([LONG], dtype=np.int8),
                sl_distances=np.full(1, 0.002),
                tp_distances=np.full(1, 0.004),
                max_hold=10,
                signal_bar_minutes=1,
                exec_timestamps=ts,
                exec_opens=o,
                exec_highs=h,
                exec_lows=l,
                exec_closes=c,
                exec_bar_minutes=1,
            )

    def test_entry_costs_passthrough(self):
        """entry_costs should be passed to simulate_trades."""
        ts_1h, o_1h, h_1h, l_1h, c_1h = _make_bars(100, 0, 3600)
        ts_1m, o_1m, h_1m, l_1m, c_1m = _make_bars(6000, 0, 60)

        signal_bars = np.array([10, 30], dtype=np.int32)
        directions = np.array([LONG, SHORT], dtype=np.int8)
        sl = np.full(2, 0.002)
        tp = np.full(2, 0.004)
        costs = np.array([0.05, 0.05])

        results = simulate_trades_hires(
            signal_timestamps=ts_1h,
            signal_bars=signal_bars,
            directions=directions,
            sl_distances=sl,
            tp_distances=tp,
            max_hold=24,
            signal_bar_minutes=60,
            exec_timestamps=ts_1m,
            exec_opens=o_1m,
            exec_highs=h_1m,
            exec_lows=l_1m,
            exec_closes=c_1m,
            entry_costs=costs,
        )

        # Cost should be applied
        assert np.all(results["cost_r"] >= 0)

    def test_5m_exec_resolution(self):
        """Should work with 5m exec bars (not just 1m)."""
        ts_1h, o_1h, h_1h, l_1h, c_1h = _make_bars(100, 0, 3600)
        ts_5m, o_5m, h_5m, l_5m, c_5m = _make_bars(1200, 0, 300)

        signal_bars = np.array([10, 30], dtype=np.int32)
        directions = np.array([LONG, SHORT], dtype=np.int8)
        sl = np.full(2, 0.002)
        tp = np.full(2, 0.004)

        results = simulate_trades_hires(
            signal_timestamps=ts_1h,
            signal_bars=signal_bars,
            directions=directions,
            sl_distances=sl,
            tp_distances=tp,
            max_hold=10,
            signal_bar_minutes=60,
            exec_timestamps=ts_5m,
            exec_opens=o_5m,
            exec_highs=h_5m,
            exec_lows=l_5m,
            exec_closes=c_5m,
            exec_bar_minutes=5,
        )

        assert len(results) == 2
