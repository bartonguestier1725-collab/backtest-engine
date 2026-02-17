"""Tests for backtest_engine.indicators — numba-accelerated technical indicators."""

import numpy as np
import pandas as pd
import pytest

from backtest_engine.indicators import (
    sma, true_range, atr, bollinger_bands, rci, expanding_quantile, map_higher_tf,
)


class TestSMA:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        result = sma(data, 3)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == pytest.approx(2.0)
        assert result[3] == pytest.approx(3.0)
        assert result[4] == pytest.approx(4.0)

    def test_matches_pandas(self):
        np.random.seed(42)
        data = np.random.randn(200).cumsum() + 100
        period = 20
        result = sma(data, period)
        expected = pd.Series(data).rolling(period).mean().values
        # Compare non-NaN values
        mask = ~np.isnan(expected)
        np.testing.assert_allclose(result[mask], expected[mask], atol=1e-10)

    def test_period_1(self):
        data = np.array([3.0, 7.0, 2.0], dtype=np.float64)
        result = sma(data, 1)
        np.testing.assert_allclose(result, data)


class TestTrueRange:
    def test_basic(self):
        high = np.array([10.0, 12.0, 11.0], dtype=np.float64)
        low = np.array([8.0, 9.0, 8.5], dtype=np.float64)
        close = np.array([9.0, 11.0, 9.0], dtype=np.float64)
        result = true_range(high, low, close)
        assert result[0] == pytest.approx(2.0)  # H-L
        assert result[1] == pytest.approx(3.0)  # max(3, 3, 0) = 3
        assert result[2] == pytest.approx(2.5)  # max(2.5, 0, 2.5) = 2.5


class TestATR:
    def test_matches_pandas(self):
        np.random.seed(42)
        n = 200
        close = np.random.randn(n).cumsum() + 100
        high = close + np.abs(np.random.randn(n)) * 0.5
        low = close - np.abs(np.random.randn(n)) * 0.5
        period = 14

        result = atr(high, low, close, period)

        # Compute pandas reference
        tr_series = pd.DataFrame({"h": high, "l": low, "c": close})
        tr_ref = np.maximum(
            tr_series["h"] - tr_series["l"],
            np.maximum(
                abs(tr_series["h"] - tr_series["c"].shift(1)),
                abs(tr_series["l"] - tr_series["c"].shift(1)),
            ),
        )
        # First TR value: H-L (no previous close)
        tr_ref.iloc[0] = high[0] - low[0]
        expected = tr_ref.rolling(period).mean().values

        mask = ~np.isnan(expected)
        np.testing.assert_allclose(result[mask], expected[mask], atol=1e-10)


class TestBollingerBands:
    def test_matches_pandas(self):
        np.random.seed(42)
        data = np.random.randn(200).cumsum() + 100
        period = 20
        num_std = 2.0

        upper, mid, lower = bollinger_bands(data, period, num_std)

        # Pandas reference (ddof=0 for population std)
        s = pd.Series(data)
        pd_mid = s.rolling(period).mean()
        pd_std = s.rolling(period).std(ddof=0)
        pd_upper = pd_mid + num_std * pd_std
        pd_lower = pd_mid - num_std * pd_std

        mask = ~np.isnan(pd_mid.values)
        np.testing.assert_allclose(mid[mask], pd_mid.values[mask], atol=1e-10)
        np.testing.assert_allclose(upper[mask], pd_upper.values[mask], atol=1e-10)
        np.testing.assert_allclose(lower[mask], pd_lower.values[mask], atol=1e-10)

    def test_symmetry(self):
        data = np.array([100.0] * 20, dtype=np.float64)
        upper, mid, lower = bollinger_bands(data, 5, 2.0)
        # In flat market, bands should be at mid
        assert upper[19] == pytest.approx(100.0)
        assert lower[19] == pytest.approx(100.0)


class TestRCI:
    def test_perfect_uptrend(self):
        """Monotonically increasing prices → RCI = +100."""
        data = np.arange(1.0, 21.0, dtype=np.float64)
        result = rci(data, 10)
        assert result[9] == pytest.approx(100.0)
        assert result[19] == pytest.approx(100.0)

    def test_perfect_downtrend(self):
        """Monotonically decreasing prices → RCI = -100."""
        data = np.arange(20.0, 0.0, -1.0, dtype=np.float64)
        result = rci(data, 10)
        assert result[9] == pytest.approx(-100.0)

    def test_range(self):
        """RCI should always be in [-100, +100]."""
        np.random.seed(42)
        data = np.random.randn(100).cumsum() + 100
        result = rci(data, 9)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= -100.0)
        assert np.all(valid <= 100.0)

    def test_nan_prefix(self):
        data = np.arange(1.0, 11.0, dtype=np.float64)
        result = rci(data, 5)
        assert np.isnan(result[0])
        assert np.isnan(result[3])
        assert not np.isnan(result[4])


class TestExpandingQuantile:
    def test_monotone_increasing(self):
        """Each new value is the max → quantile = 1.0."""
        data = np.arange(1.0, 6.0, dtype=np.float64)
        result = expanding_quantile(data, data)
        np.testing.assert_allclose(result, [1.0, 1.0, 1.0, 1.0, 1.0])

    def test_monotone_decreasing(self):
        """Each new value is the min → quantile = 1/n."""
        data = np.arange(5.0, 0.0, -1.0, dtype=np.float64)
        result = expanding_quantile(data, data)
        expected = np.array([1/1, 1/2, 1/3, 1/4, 1/5])
        np.testing.assert_allclose(result, expected)

    def test_range(self):
        np.random.seed(42)
        data = np.random.randn(100)
        result = expanding_quantile(data, data)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)


class TestMapHigherTF:
    def test_basic_mapping(self):
        # Lower TF: 1-minute bars (timestamps in seconds)
        lower_ts = np.array([60, 120, 180, 240, 300, 360, 420, 480], dtype=np.int64)
        # Higher TF: 5-minute bars (open timestamps)
        higher_ts = np.array([0, 300, 600], dtype=np.int64)
        higher_vals = np.array([10.0, 20.0, 30.0], dtype=np.float64)

        result = map_higher_tf(lower_ts, higher_ts, higher_vals)

        # Bars 0-3 (60-240): within first higher bar (0-300), completed = none or bar[0]
        # Before higher_ts[1]=300: h_idx=1, completed = 1-2 = -1 → NaN
        assert np.isnan(result[0])  # ts=60, in first bar, no completed
        # Bar 4 (ts=300): h_idx becomes 2, completed = 2-2 = 0 → val[0]=10
        assert result[4] == pytest.approx(10.0)
        # Bars 5-7 (360-480): h_idx=2, completed = 0 → val[0]=10
        assert result[5] == pytest.approx(10.0)

    def test_all_nan_before_first_complete(self):
        lower_ts = np.array([10, 20, 30], dtype=np.int64)
        higher_ts = np.array([0, 100], dtype=np.int64)
        higher_vals = np.array([1.0, 2.0], dtype=np.float64)

        result = map_higher_tf(lower_ts, higher_ts, higher_vals)
        assert np.all(np.isnan(result))
