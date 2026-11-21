"""Fill-price rules added in v0.6.0: stop gaps and retrace (limit) entries."""

import numpy as np
import pytest

from backtest_engine import (
    simulate_trades, LONG, SHORT,
    EXIT_SL, EXIT_TP, EXIT_BE, EXIT_TRAIL, EXIT_TIME, EXIT_NO_FILL, EXIT_CUSTOM,
)


def _bars(n=10, price=100.0):
    o = np.full(n, price)
    return o, o + 0.5, o - 0.5, o.copy()


def _one(o, h, l, c, direction, sl, tp, use_open=True, max_hold=8, **kw):
    return simulate_trades(
        h, l, c,
        signal_bars=np.array([0], dtype=np.int32),
        directions=np.array([direction], dtype=np.int8),
        sl_distances=np.array([sl]), tp_distances=np.array([tp]),
        max_hold=max_hold, open_prices=o if use_open else None, **kw,
    )[0]


class TestStopGap:
    def test_long_sl_gap_fills_at_open(self):
        o, h, l, c = _bars()
        # entry at open[1] = 100, SL = 95. Bar 3 opens at 90 (gap through SL)
        o[3], h[3], l[3], c[3] = 90.0, 91.0, 89.0, 90.5
        r = _one(o, h, l, c, LONG, 5.0, 20.0)
        assert r["exit_type"] == EXIT_SL
        assert r["exit_bar"] == 3
        assert r["pnl_r"] == pytest.approx((90.0 - 100.0) / 5.0)  # -2R, not -1R

    def test_short_sl_gap_fills_at_open(self):
        o, h, l, c = _bars()
        o[3], h[3], l[3], c[3] = 110.0, 111.0, 109.0, 110.0
        r = _one(o, h, l, c, SHORT, 5.0, 20.0)
        assert r["exit_type"] == EXIT_SL
        assert r["pnl_r"] == pytest.approx(-2.0)

    def test_no_gap_fills_at_stop(self):
        o, h, l, c = _bars()
        l[3] = 94.0  # opens at 100, trades down through 95
        r = _one(o, h, l, c, LONG, 5.0, 20.0)
        assert r["pnl_r"] == pytest.approx(-1.0)

    def test_without_open_prices_gap_is_unknowable(self):
        o, h, l, c = _bars()
        h[3], l[3], c[3] = 91.0, 89.0, 90.5
        r = _one(o, h, l, c, LONG, 5.0, 20.0, use_open=False)
        assert r["exit_type"] == EXIT_SL
        assert r["pnl_r"] == pytest.approx(-1.0)

    def test_tp_gap_still_fills_at_tp(self):
        o, h, l, c = _bars()
        o[3], h[3], l[3], c[3] = 130.0, 131.0, 129.0, 130.0
        r = _one(o, h, l, c, LONG, 5.0, 10.0)
        assert r["exit_type"] == EXIT_TP
        assert r["pnl_r"] == pytest.approx(2.0)

    def test_breakeven_gap(self):
        o, h, l, c = _bars()
        h[2] = 106.0                       # +6 >= 0.5 * TP(10): SL moves to 100
        o[3], h[3], l[3], c[3] = 101.0, 101.5, 100.2, 101.0  # stays above 100
        o[4], h[4], l[4], c[4] = 97.0, 98.0, 96.0, 97.0  # gap below breakeven
        r = _one(o, h, l, c, LONG, 5.0, 10.0, be_trigger_pct=0.5)
        assert r["exit_type"] == EXIT_BE
        assert r["pnl_r"] == pytest.approx(-3.0 / 5.0)

    def test_trailing_gap(self):
        o, h, l, c = _bars()
        h[2] = 110.0                       # best=110, trail 1R=5 -> SL 105
        o[3], h[3], l[3], c[3] = 102.0, 103.0, 101.0, 102.0
        r = _one(o, h, l, c, LONG, 5.0, 100.0, exit_mode="trailing",
                 trail_activation_r=1.0, trail_distance_r=1.0)
        assert r["exit_type"] == EXIT_TRAIL
        assert r["pnl_r"] == pytest.approx(2.0 / 5.0)  # open 102, not stop 105

    def test_custom_mode_gap(self):
        o, h, l, c = _bars()
        o[3], h[3], l[3], c[3] = 90.0, 91.0, 89.0, 90.5
        r = _one(o, h, l, c, LONG, 5.0, 20.0, exit_mode="custom",
                 exit_signals=np.zeros(len(c), dtype=bool))
        assert r["exit_type"] == EXIT_SL
        assert r["pnl_r"] == pytest.approx(-2.0)

    def test_sar_mode_gap(self):
        o, h, l, c = _bars()
        o[3], h[3], l[3], c[3] = 90.0, 91.0, 89.0, 90.5
        sar = np.full(len(c), 50.0)        # far below price: hard SL governs
        r = _one(o, h, l, c, LONG, 5.0, 20.0, exit_mode="sar_trailing", sar_values=sar)
        assert r["exit_type"] == EXIT_SL
        assert r["pnl_r"] == pytest.approx(-2.0)


class TestRetraceEntry:
    """Before v0.6.0 the SL of a retrace (limit) entry was measured from the
    signal-bar close instead of the fill price, and the fill bar was skipped.
    With retrace distance > SL distance a stop-out showed a profit."""

    def _retrace_bars(self):
        o, h, l, c = _bars(12)
        # signal close 100; TP 20, retrace 0.5 -> limit 90. Fill on bar 2.
        h[2], l[2], c[2] = 100.5, 89.5, 91.0
        return o, h, l, c

    def test_stop_out_is_minus_one_r(self):
        o, h, l, c = self._retrace_bars()
        o[3:], h[3:], l[3:], c[3:] = 86.0, 86.5, 84.0, 85.0  # below SL 90-5=85
        r = _one(o, h, l, c, LONG, 5.0, 20.0, retrace_pct=0.5)
        assert r["entry_bar"] == 2
        assert r["exit_type"] == EXIT_SL
        assert r["pnl_r"] == pytest.approx(-1.0)

    def test_tp_measured_from_fill(self):
        o, h, l, c = self._retrace_bars()
        h[4] = 110.5  # TP = 90 + 20 = 110
        r = _one(o, h, l, c, LONG, 5.0, 20.0, retrace_pct=0.5)
        assert r["exit_type"] == EXIT_TP
        assert r["pnl_r"] == pytest.approx(4.0)

    def test_sl_on_fill_bar_is_checked(self):
        o, h, l, c = self._retrace_bars()
        l[2] = 80.0   # passes the limit (90), then the stop (85), in this bar
        r = _one(o, h, l, c, LONG, 5.0, 20.0, retrace_pct=0.5)
        assert r["exit_type"] == EXIT_SL
        assert r["exit_bar"] == 2
        assert r["hold_bars"] == 0
        assert r["pnl_r"] == pytest.approx(-1.0)

    def test_tp_on_fill_bar_is_not_counted(self):
        o, h, l, c = self._retrace_bars()
        h[2] = 120.0  # high may have come before the fill
        r = _one(o, h, l, c, LONG, 5.0, 20.0, retrace_pct=0.5)
        assert r["exit_type"] == EXIT_TIME

    def test_short_retrace(self):
        o, h, l, c = _bars(12)
        h[2], l[2], c[2] = 110.5, 99.5, 109.0  # limit 110
        o[3:], h[3:], l[3:], c[3:] = 114.0, 116.0, 113.5, 115.0  # SL 115
        r = _one(o, h, l, c, SHORT, 5.0, 20.0, retrace_pct=0.5)
        assert r["exit_type"] == EXIT_SL
        assert r["pnl_r"] == pytest.approx(-1.0)

    def test_no_fill(self):
        o, h, l, c = _bars(12)
        r = _one(o, h, l, c, LONG, 5.0, 20.0, retrace_pct=0.5, retrace_timeout=5)
        assert r["exit_type"] == EXIT_NO_FILL


class TestSignalBarField:
    def test_simulate_trades_records_signal_bar(self):
        o, h, l, c = _bars(20)
        r = simulate_trades(h, l, c, np.array([2, 7], dtype=np.int32),
                            np.array([LONG, SHORT], dtype=np.int8),
                            np.array([5.0, 5.0]), np.array([10.0, 10.0]),
                            max_hold=3, open_prices=o)
        np.testing.assert_array_equal(r["signal_bar"], [2, 7])
        np.testing.assert_array_equal(r["entry_bar"], [3, 8])


class TestTpGap:
    def test_open_beyond_tp_is_tp_even_if_sl_touched(self):
        o, h, l, c = _bars()
        # entry 100, SL 95, TP 110. Bar 3 opens at 112 (gap through TP), then
        # trades down to 94: the TP filled at the open, before the SL touch.
        o[3], h[3], l[3], c[3] = 112.0, 112.5, 94.0, 96.0
        r = _one(o, h, l, c, LONG, 5.0, 10.0)
        assert r["exit_type"] == EXIT_TP
        assert r["pnl_r"] == pytest.approx(2.0)

    def test_short_open_beyond_tp(self):
        o, h, l, c = _bars()
        o[3], h[3], l[3], c[3] = 88.0, 106.0, 87.5, 104.0
        r = _one(o, h, l, c, SHORT, 5.0, 10.0)
        assert r["exit_type"] == EXIT_TP

    def test_without_open_prices_sl_keeps_priority(self):
        o, h, l, c = _bars()
        h[3], l[3], c[3] = 112.5, 94.0, 96.0
        r = _one(o, h, l, c, LONG, 5.0, 10.0, use_open=False)
        assert r["exit_type"] == EXIT_SL


class TestHoldWindow:
    """max_hold bars are held in both entry modes (before v0.6.0 an open
    entry held one bar more)."""

    def test_open_entry_holds_max_hold_bars(self):
        o, h, l, c = _bars(20)
        r = _one(o, h, l, c, LONG, 50.0, 50.0, max_hold=3)  # never hits SL/TP
        assert r["exit_type"] == EXIT_TIME
        assert (r["entry_bar"], r["exit_bar"]) == (1, 3)     # bars 1, 2, 3

    def test_close_entry_holds_max_hold_bars(self):
        o, h, l, c = _bars(20)
        r = _one(o, h, l, c, LONG, 50.0, 50.0, use_open=False, max_hold=3)
        assert (r["entry_bar"], r["exit_bar"]) == (0, 3)     # bars 1, 2, 3

    def test_max_hold_zero_raises(self):
        o, h, l, c = _bars()
        with pytest.raises(ValueError, match="max_hold"):
            _one(o, h, l, c, LONG, 5.0, 10.0, max_hold=0)


class TestCustomExitFill:
    def test_exit_at_next_open_with_open_prices(self):
        o, h, l, c = _bars(12)
        sig = np.zeros(12, dtype=bool); sig[3] = True
        o[4] = 103.0; h[4] = 103.5; c[4] = 103.0
        r = _one(o, h, l, c, LONG, 5.0, 20.0, exit_mode="custom", exit_signals=sig)
        assert r["exit_type"] == EXIT_CUSTOM
        assert r["exit_bar"] == 4
        assert r["pnl_r"] == pytest.approx(3.0 / 5.0)

    def test_exit_at_close_without_open_prices(self):
        o, h, l, c = _bars(12)
        sig = np.zeros(12, dtype=bool); sig[3] = True
        c[3] = 102.0; h[3] = 102.5
        r = _one(o, h, l, c, LONG, 5.0, 20.0, use_open=False,
                 exit_mode="custom", exit_signals=sig)
        assert r["exit_bar"] == 3
        assert r["pnl_r"] == pytest.approx(2.0 / 5.0)
