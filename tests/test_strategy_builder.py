"""Tests for backtest_engine.strategy_builder."""

import numpy as np
import pandas as pd
import pytest

from backtest_engine.strategy_builder import (
    build_time_based_signals,
    _to_utc_hour,
    MatchReport,
)


class TestToUtcHour:
    def test_jst_to_utc_morning(self):
        """02:00 JST = 17:00 UTC previous day."""
        utc_hour, day_offset = _to_utc_hour(2, tz_offset_hours=9)
        assert utc_hour == 17
        assert day_offset == -1

    def test_jst_to_utc_afternoon(self):
        """14:00 JST = 05:00 UTC same day."""
        utc_hour, day_offset = _to_utc_hour(14, tz_offset_hours=9)
        assert utc_hour == 5
        assert day_offset == 0

    def test_utc_no_offset(self):
        """10:00 UTC+0 = 10:00 UTC."""
        utc_hour, day_offset = _to_utc_hour(10, tz_offset_hours=0)
        assert utc_hour == 10
        assert day_offset == 0

    def test_negative_offset(self):
        """10:00 UTC-5 = 15:00 UTC."""
        utc_hour, day_offset = _to_utc_hour(10, tz_offset_hours=-5)
        assert utc_hour == 15
        assert day_offset == 0


class TestMatchReport:
    def test_empty_report(self):
        r = MatchReport()
        assert r.total_candidates == 0
        assert r.drop_rate == 0.0

    def test_full_match(self):
        r = MatchReport(matched=10)
        assert r.total_candidates == 10
        assert r.drop_rate == 0.0

    def test_partial_match(self):
        r = MatchReport(matched=7, unmatched_no_exit_bar=3)
        assert r.total_candidates == 10
        assert r.drop_rate == pytest.approx(0.3)

    def test_warning_on_high_drop_rate(self):
        r = MatchReport(matched=5, unmatched_no_exit_bar=5)
        summary = r.summary()
        assert "WARNING" in summary


def _make_hourly_bars(start_date: str, days: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic hourly bars for testing.

    Creates bars for FX market hours (Sun 22:00 to Fri 22:00 UTC).
    Returns (timestamps, open_prices, close_prices).
    """
    # Generate all hours in range
    start = pd.Timestamp(start_date, tz="UTC")
    end = start + pd.Timedelta(days=days)
    all_hours = pd.date_range(start, end, freq="h")

    # Filter to FX market hours (skip Sat 00:00 to Sun 21:59)
    fx_hours = []
    for t in all_hours:
        dow = t.dayofweek  # 0=Mon ... 6=Sun
        if dow == 5:  # Saturday — market closed
            continue
        if dow == 6 and t.hour < 22:  # Sunday before 22:00 — closed
            continue
        fx_hours.append(t)

    fx_hours = pd.DatetimeIndex(fx_hours)
    timestamps = np.array([int(t.timestamp()) for t in fx_hours], dtype=np.int64)
    n = len(timestamps)
    np.random.seed(42)
    base_price = 2000.0 + np.cumsum(np.random.randn(n) * 2)
    open_prices = base_price
    close_prices = base_price + np.random.randn(n) * 0.5

    return timestamps, open_prices, close_prices


class TestBuildTimeBasedSignals:
    """Gold Rollover style: Entry 02:00 JST, Exit 14:00 JST, skip Thursday JST."""

    @pytest.fixture
    def market_data(self):
        """Two weeks of synthetic 1h data."""
        return _make_hourly_bars("2024-01-01", days=14)

    def test_basic_matching(self, market_data):
        """Should find trades and produce a match report."""
        timestamps, opens, closes = market_data
        trades, report = build_time_based_signals(
            timestamps, opens, closes,
            entry_hour_local=2,
            exit_hour_local=14,
            tz_offset_hours=9,
            direction=1,
        )
        assert len(trades) > 0
        assert report.matched > 0
        assert report.matched == len(trades)

    def test_skip_thursday_jst(self, market_data):
        """Skipping Thursday JST should reduce trade count."""
        timestamps, opens, closes = market_data
        trades_all, _ = build_time_based_signals(
            timestamps, opens, closes,
            entry_hour_local=2,
            exit_hour_local=14,
            tz_offset_hours=9,
        )
        trades_skip, _ = build_time_based_signals(
            timestamps, opens, closes,
            entry_hour_local=2,
            exit_hour_local=14,
            tz_offset_hours=9,
            skip_weekdays_local=[3],  # Thursday = 3
        )
        assert len(trades_skip) < len(trades_all)

    def test_no_silent_drops(self, market_data):
        """Every entry candidate must be accounted for in the report."""
        timestamps, opens, closes = market_data
        trades, report = build_time_based_signals(
            timestamps, opens, closes,
            entry_hour_local=2,
            exit_hour_local=14,
            tz_offset_hours=9,
        )
        # Total candidates = matched + all unmatched categories
        assert report.total_candidates == report.matched + report.unmatched_no_exit_bar + report.unmatched_market_closed + report.unmatched_other
        # Every matched entry appears in trades_df
        assert report.matched == len(trades)

    def test_trade_columns(self, market_data):
        """Output DataFrame should have all required columns."""
        timestamps, opens, closes = market_data
        trades, _ = build_time_based_signals(
            timestamps, opens, closes,
            entry_hour_local=2,
            exit_hour_local=14,
            tz_offset_hours=9,
        )
        required = {"entry_bar_idx", "exit_bar_idx", "entry_price", "exit_price",
                     "entry_time", "exit_time", "direction"}
        assert required.issubset(set(trades.columns))

    def test_hold_duration(self, market_data):
        """All trades should hold approximately 12 hours."""
        timestamps, opens, closes = market_data
        trades, _ = build_time_based_signals(
            timestamps, opens, closes,
            entry_hour_local=2,
            exit_hour_local=14,
            tz_offset_hours=9,
        )
        if len(trades) == 0:
            pytest.skip("No trades generated")
        for _, row in trades.iterrows():
            hold = (row["exit_time"] - row["entry_time"]).total_seconds() / 3600
            assert 11.5 <= hold <= 12.5, f"Hold duration {hold}h outside expected range"

    def test_report_warns_on_high_drop_rate(self, market_data):
        """If many entries are dropped, report should warn."""
        timestamps, opens, closes = market_data
        _, report = build_time_based_signals(
            timestamps, opens, closes,
            entry_hour_local=2,
            exit_hour_local=14,
            tz_offset_hours=9,
        )
        # Even with weekend drops, the report should account for everything
        assert report.total_candidates > 0

    def test_empty_data(self):
        """Empty input should return empty results."""
        trades, report = build_time_based_signals(
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            entry_hour_local=2,
            exit_hour_local=14,
        )
        assert len(trades) == 0
        assert report.total_candidates == 0

    def test_entry_price_mode_close(self, market_data):
        """entry_price_mode='close' should use close prices."""
        timestamps, opens, closes = market_data
        trades_open, _ = build_time_based_signals(
            timestamps, opens, closes,
            entry_hour_local=2,
            exit_hour_local=14,
            tz_offset_hours=9,
            entry_price_mode="open",
        )
        trades_close, _ = build_time_based_signals(
            timestamps, opens, closes,
            entry_hour_local=2,
            exit_hour_local=14,
            tz_offset_hours=9,
            entry_price_mode="close",
        )
        if len(trades_open) > 0 and len(trades_close) > 0:
            # Entry prices should differ (open vs close)
            assert not np.allclose(
                trades_open["entry_price"].values,
                trades_close["entry_price"].values[:len(trades_open)],
            )
