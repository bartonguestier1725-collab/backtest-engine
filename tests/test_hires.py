"""Tests for simulate_trades_hires — multi-resolution trade simulation."""

import numpy as np
import pytest

from backtest_engine import (
    simulate_trades, simulate_trades_hires, resample_ohlcv, DataQualityWarning,
    LONG, SHORT, EXIT_SL, EXIT_TIME, EXIT_NO_FILL,
)


def _make_bars(n_bars: int, start_ts: int, bar_seconds: int, base_price: float = 1.10):
    """Generate synthetic OHLC bars."""
    rng = np.random.default_rng(42)
    timestamps = np.arange(n_bars, dtype=np.int64) * bar_seconds + start_ts
    noise = rng.normal(0, 0.0005, n_bars)
    closes = base_price + np.cumsum(noise)
    highs = closes + rng.uniform(0.0001, 0.001, n_bars)
    lows = closes - rng.uniform(0.0001, 0.001, n_bars)
    opens = closes + rng.normal(0, 0.0002, n_bars)
    # Keep each bar's open inside its own high/low, as in real data
    highs = np.maximum(highs, opens)
    lows = np.minimum(lows, opens)
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
        """Signal on 1h bar 5 (opens 05:00, closes 06:00) enters at the 06:00 1m bar.

        Before v0.6.0 the entry was the 05:01 bar, 59 minutes before the
        signal bar's close was known (look-ahead).
        """
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

        # 1h bar 5 closes at 6*3600 = 1m bar 360. Enter at its open.
        assert results["entry_bar"][0] == 360
        assert results["signal_bar"][0] == 359
        assert ts_1m[results["entry_bar"][0]] == ts_1h[5] + 3600

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


def _hires(ts_sig, sb, dirs, sl, tp, ts_x, o_x, h_x, l_x, c_x, max_hold=10, **kw):
    return simulate_trades_hires(
        signal_timestamps=ts_sig, signal_bars=np.asarray(sb, dtype=np.int32),
        directions=np.asarray(dirs, dtype=np.int8),
        sl_distances=np.asarray(sl, dtype=np.float64),
        tp_distances=np.asarray(tp, dtype=np.float64),
        max_hold=max_hold, signal_bar_minutes=60,
        exec_timestamps=ts_x, exec_opens=o_x, exec_highs=h_x, exec_lows=l_x,
        exec_closes=c_x, **kw,
    )


class TestHiresNoLookAhead:
    """Regression tests for the v0.5.x look-ahead (entry 1 exec bar after the
    signal bar OPENED instead of after it CLOSED)."""

    def test_entry_time_is_signal_bar_close(self):
        ts_1h, *_ = _make_bars(100, 0, 3600)
        ts_1m, o, h, l, c = _make_bars(6000, 0, 60)
        sb = np.array([3, 10, 41, 77])
        r = _hires(ts_1h, sb, [LONG, SHORT, LONG, SHORT], [0.002] * 4, [0.004] * 4,
                   ts_1m, o, h, l, c)
        entry_ts = ts_1m[r["entry_bar"]]
        np.testing.assert_array_equal(entry_ts, ts_1h[sb] + 3600)
        assert np.all(r["signal_bar"] < r["entry_bar"])
        # The exec signal bar is the last 1m bar inside the signal hour
        np.testing.assert_array_equal(ts_1m[r["signal_bar"]], ts_1h[sb] + 3600 - 60)

    def test_move_inside_signal_bar_cannot_stop_the_trade(self):
        """A crash inside the signal hour happens before the signal is known,
        so it must not stop out the trade."""
        n_1m = 600
        ts_1m = np.arange(n_1m, dtype=np.int64) * 60
        c = np.full(n_1m, 1.1000)
        o, h, l = c.copy(), c + 0.0001, c - 0.0001
        # Signal hour = 1h bar 2 (1m bars 120..179): crash 50 pips at 02:10
        l[130] = 1.0950
        ts_1h = np.arange(10, dtype=np.int64) * 3600
        r = _hires(ts_1h, [2], [LONG], [0.0020], [0.0040], ts_1m, o, h, l, c, max_hold=3)
        assert r["entry_bar"][0] == 180
        assert r["exit_type"][0] == EXIT_TIME  # flat after the signal: no SL

    def test_entry_matches_next_1h_open(self):
        """With 1h bars built from the 1m bars, hires enters at the same price
        as 1h next-bar-open entry."""
        ts_1m, o, h, l, c = _make_bars(6000, 0, 60)
        ts_1h, o_1h, h_1h, l_1h, c_1h, _ = resample_ohlcv(
            ts_1m, o, h, l, c, np.zeros(len(ts_1m)), rule="1h")
        sb = np.array([5, 20, 60])
        r = _hires(ts_1h, sb, [LONG] * 3, [1.0] * 3, [2.0] * 3, ts_1m, o, h, l, c)
        np.testing.assert_allclose(o[r["entry_bar"]], o_1h[sb + 1])


class TestHiresCoverage:
    def test_signal_after_exec_data_end_is_no_fill_with_warning(self):
        ts_1h, *_ = _make_bars(100, 0, 3600)
        ts_1m, o, h, l, c = _make_bars(3000, 0, 60)  # 1m data ends at 50h
        with pytest.warns(DataQualityWarning, match="outside the execution data"):
            r = _hires(ts_1h, [10, 70], [LONG, LONG], [0.002] * 2, [0.004] * 2,
                       ts_1m, o, h, l, c, entry_costs=np.array([0.1, 0.1]))
        assert r["exit_type"][0] != EXIT_NO_FILL
        assert r["exit_type"][1] == EXIT_NO_FILL
        assert r["pnl_r"][1] == 0.0
        assert r["cost_r"][1] == 0.0

    def test_signal_before_exec_data_start_is_no_fill(self):
        ts_1h, *_ = _make_bars(100, 0, 3600)
        ts_1m, o, h, l, c = _make_bars(3000, 20 * 3600, 60)  # 1m data starts at 20h
        with pytest.warns(DataQualityWarning):
            r = _hires(ts_1h, [5, 30], [LONG, LONG], [0.002] * 2, [0.004] * 2,
                       ts_1m, o, h, l, c)
        assert r["exit_type"][0] == EXIT_NO_FILL
        assert r["exit_type"][1] != EXIT_NO_FILL

    def test_gap_in_exec_data_warns(self):
        ts_1h, *_ = _make_bars(100, 0, 3600)
        ts_1m, o, h, l, c = _make_bars(6000, 0, 60)
        keep = (ts_1m < 11 * 3600) | (ts_1m >= 13 * 3600)  # 11:00-13:00 missing
        with pytest.warns(DataQualityWarning, match="gap"):
            r = _hires(ts_1h, [10], [LONG], [0.002], [0.004],
                       ts_1m[keep], o[keep], h[keep], l[keep], c[keep])
        assert ts_1m[keep][r["entry_bar"][0]] == 13 * 3600

    def test_unsorted_exec_timestamps_raise(self):
        ts_1h, *_ = _make_bars(10, 0, 3600)
        ts_1m, o, h, l, c = _make_bars(600, 0, 60)
        ts_1m = ts_1m.copy(); ts_1m[[5, 6]] = ts_1m[[6, 5]]
        with pytest.raises(ValueError, match="strictly increasing"):
            _hires(ts_1h, [2], [LONG], [0.002], [0.004], ts_1m, o, h, l, c)
