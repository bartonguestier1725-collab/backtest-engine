"""Tests for backtest_engine.indicators — numba-accelerated technical indicators."""

import numpy as np
import pandas as pd
import pytest

from backtest_engine.indicators import (
    sma, true_range, atr, bollinger_bands, rci, expanding_quantile, map_higher_tf,
    parabolic_sar,
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


class TestParabolicSAR:
    def test_uptrend_sar_below_price(self):
        """In a monotonic uptrend, SAR should stay below price."""
        n = 50
        close = np.arange(100.0, 100.0 + n, dtype=np.float64)
        high = close + 0.5
        low = close - 0.5
        sar, trend, sar_stop = parabolic_sar(high, low)
        # After initial bars, SAR should be below low in uptrend
        for i in range(5, n):
            assert sar[i] < low[i], f"SAR {sar[i]} >= low {low[i]} at bar {i}"
            assert trend[i] == 1, f"trend should be 1 at bar {i}"

    def test_downtrend_sar_above_price(self):
        """In a monotonic downtrend (after initial uptrend start), SAR should flip."""
        n = 50
        # Start up then go down sharply
        close = np.empty(n, dtype=np.float64)
        close[0] = 100.0
        close[1] = 101.0
        close[2] = 102.0
        for i in range(3, n):
            close[i] = 102.0 - (i - 2) * 2.0
        high = close + 0.3
        low = close - 0.3
        sar, trend, sar_stop = parabolic_sar(high, low)
        # Should eventually be downtrend with SAR above price
        assert trend[-1] == -1
        assert sar[-1] > high[-1]

    def test_reversal_flips_trend(self):
        """Trend should flip on reversal and SAR should jump to extreme point."""
        n = 30
        close = np.empty(n, dtype=np.float64)
        # Uptrend then sharp reversal
        for i in range(15):
            close[i] = 100.0 + i * 1.0
        for i in range(15, n):
            close[i] = 114.0 - (i - 14) * 2.0
        high = close + 0.5
        low = close - 0.5
        sar, trend, sar_stop = parabolic_sar(high, low)
        # Find flip point
        flip_found = False
        for i in range(1, n):
            if trend[i - 1] == 1 and trend[i] == -1:
                flip_found = True
                break
        assert flip_found, "Expected a trend flip from up to down"

    def test_sar_stop_pre_reversal(self):
        """sar_stop should contain pre-reversal SAR, not the EP jump."""
        n = 30
        close = np.empty(n, dtype=np.float64)
        for i in range(15):
            close[i] = 100.0 + i * 1.0
        for i in range(15, n):
            close[i] = 114.0 - (i - 14) * 2.0
        high = close + 0.5
        low = close - 0.5
        sar, trend, sar_stop = parabolic_sar(high, low)

        # At reversal bar: sar jumps to EP, sar_stop stays at pre-reversal level
        for i in range(1, n):
            if trend[i - 1] == 1 and trend[i] == -1:
                # Reversal from up to down
                # sar[i] = EP (highest high), sar_stop[i] = computed SAR (pre-reversal)
                assert sar_stop[i] < sar[i], \
                    f"At reversal bar {i}: sar_stop={sar_stop[i]} should be < sar={sar[i]}"
                # sar_stop is the level where the stop triggered (low[i] < sar_stop[i])
                assert low[i] < sar_stop[i], \
                    f"low should be below sar_stop at reversal"
                break

    def test_sar_stop_equals_sar_during_continuation(self):
        """During trend continuation, sar_stop should equal sar."""
        n = 50
        close = np.arange(100.0, 100.0 + n, dtype=np.float64)
        high = close + 0.5
        low = close - 0.5
        sar, trend, sar_stop = parabolic_sar(high, low)
        # In pure uptrend, sar and sar_stop should be identical
        np.testing.assert_allclose(sar, sar_stop)

    def test_matches_pure_python(self):
        """Numba SAR should produce same results as pure Python reference."""
        np.random.seed(42)
        n = 200
        close = np.random.randn(n).cumsum() + 100
        high = close + np.abs(np.random.randn(n)) * 0.5
        low = close - np.abs(np.random.randn(n)) * 0.5

        # Pure Python reference (inline)
        ref_sar = np.empty(n, dtype=np.float64)
        ref_trend = np.empty(n, dtype=np.int8)
        ref_sar[0] = low[0]
        ref_trend[0] = 1
        ep = high[0]
        af = 0.02
        for i in range(1, n):
            prev = ref_sar[i - 1]
            if ref_trend[i - 1] == 1:
                new_sar = prev + af * (ep - prev)
                new_sar = min(new_sar, low[i - 1])
                if i >= 2:
                    new_sar = min(new_sar, low[i - 2])
                if low[i] < new_sar:
                    ref_trend[i] = -1
                    ref_sar[i] = ep
                    ep = low[i]
                    af = 0.02
                else:
                    ref_trend[i] = 1
                    ref_sar[i] = new_sar
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + 0.02, 0.20)
            else:
                new_sar = prev + af * (ep - prev)
                new_sar = max(new_sar, high[i - 1])
                if i >= 2:
                    new_sar = max(new_sar, high[i - 2])
                if high[i] > new_sar:
                    ref_trend[i] = 1
                    ref_sar[i] = ep
                    ep = high[i]
                    af = 0.02
                else:
                    ref_trend[i] = -1
                    ref_sar[i] = new_sar
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + 0.02, 0.20)

        sar, trend, sar_stop = parabolic_sar(high, low)
        np.testing.assert_allclose(sar, ref_sar, atol=1e-10)
        np.testing.assert_array_equal(trend, ref_trend)


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
