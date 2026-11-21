"""Tests for backtest_engine.utils — CSV loading, resampling, bar lookup."""

import numpy as np
import pandas as pd
import pytest

from backtest_engine import load_ohlcv, resample_ohlcv, find_signal_bar

T0 = 1704067200  # 2024-01-01 00:00:00 UTC


def _write_csv(tmp_path, times, name="ohlc.csv", with_volume=True):
    n = len(times)
    df = pd.DataFrame({
        "time": times,
        "open": np.arange(n) + 1.0,
        "high": np.arange(n) + 2.0,
        "low": np.arange(n) + 0.5,
        "close": np.arange(n) + 1.5,
    })
    if with_volume:
        df["volume"] = np.arange(n) * 10.0
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


class TestLoadOhlcv:
    """GitHub issue #1: on pandas >= 3, ``astype(int64) // 10**9`` returned
    1704067 instead of 1704067200 because datetimes are no longer ns."""

    @pytest.mark.parametrize("times", [
        ["2024-01-01 00:00:00", "2024-01-01 01:00:00"],
        ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"],
        ["2024-01-01T09:00:00+09:00", "2024-01-01T10:00:00+09:00"],
        [T0, T0 + 3600],
        [T0 * 1000, (T0 + 3600) * 1000],
    ], ids=["naive", "utc_z", "offset", "epoch_s", "epoch_ms"])
    def test_timestamps_are_epoch_seconds(self, tmp_path, times):
        ts, *_ = load_ohlcv(_write_csv(tmp_path, times))
        assert ts.dtype == np.int64
        np.testing.assert_array_equal(ts, [T0, T0 + 3600])

    def test_prices_and_volume(self, tmp_path):
        ts, o, h, l, c, v = load_ohlcv(_write_csv(tmp_path, [T0, T0 + 60, T0 + 120]))
        np.testing.assert_array_equal(o, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(h, [2.0, 3.0, 4.0])
        np.testing.assert_array_equal(l, [0.5, 1.5, 2.5])
        np.testing.assert_array_equal(c, [1.5, 2.5, 3.5])
        np.testing.assert_array_equal(v, [0.0, 10.0, 20.0])

    def test_missing_volume_is_zeros(self, tmp_path):
        *_, v = load_ohlcv(_write_csv(tmp_path, [T0, T0 + 60], with_volume=False))
        np.testing.assert_array_equal(v, [0.0, 0.0])


class TestResampleOhlcv:
    def test_1m_to_1h(self):
        n = 180
        ts = np.arange(n, dtype=np.int64) * 60 + T0
        o = np.arange(n, dtype=np.float64)
        h, l, c, v = o + 0.5, o - 0.5, o + 0.25, np.ones(n)
        ts_h, o_h, h_h, l_h, c_h, v_h = resample_ohlcv(ts, o, h, l, c, v, rule="1h")
        np.testing.assert_array_equal(ts_h, [T0, T0 + 3600, T0 + 7200])
        np.testing.assert_array_equal(o_h, [0.0, 60.0, 120.0])
        np.testing.assert_array_equal(h_h, [59.5, 119.5, 179.5])
        np.testing.assert_array_equal(l_h, [-0.5, 59.5, 119.5])
        np.testing.assert_array_equal(c_h, [59.25, 119.25, 179.25])
        np.testing.assert_array_equal(v_h, [60.0, 60.0, 60.0])

    def test_empty_bins_are_dropped(self):
        ts = np.array([T0, T0 + 60, T0 + 3 * 3600], dtype=np.int64)
        x = np.ones(3)
        ts_h, *_ = resample_ohlcv(ts, x, x, x, x, x, rule="1h")
        np.testing.assert_array_equal(ts_h, [T0, T0 + 3 * 3600])


class TestFindSignalBar:
    def test_exact_and_between(self):
        ts = np.array([0, 60, 120, 180], dtype=np.int64)
        assert find_signal_bar(ts, 120) == 2
        assert find_signal_bar(ts, 150) == 2
        assert find_signal_bar(ts, 10_000) == 3

    def test_before_first_bar_clamps_to_zero(self):
        ts = np.array([60, 120], dtype=np.int64)
        assert find_signal_bar(ts, 0) == 0
