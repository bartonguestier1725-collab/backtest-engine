"""Tests for pre-flight quality check and TradeResults."""

import pickle
import warnings

import numpy as np
import pytest

from backtest_engine._types import (
    LONG, SHORT,
    EXIT_SL, EXIT_TP, EXIT_TIME, EXIT_NO_FILL,
    TRADE_RESULT_DTYPE,
)
from backtest_engine.core import simulate_trades
from backtest_engine.preflight import (
    BacktestQualityWarning,
    PreflightReport,
    run_preflight,
)
from backtest_engine._results import TradeResults


# ── PreflightReport unit tests ────────────────────────────────────────────────

class TestRunPreflight:
    def test_grade_a_both_provided(self):
        report = run_preflight(
            open_prices=np.array([1.0]),
            entry_costs=np.array([0.01]),
        )
        assert report.grade == "A"
        assert report.has_entry_costs
        assert report.has_open_prices

    def test_grade_b_costs_only(self):
        report = run_preflight(open_prices=None, entry_costs=np.array([0.01]))
        assert report.grade == "B"
        assert report.has_entry_costs
        assert not report.has_open_prices

    def test_grade_b_open_only(self):
        report = run_preflight(open_prices=np.array([1.0]), entry_costs=None)
        assert report.grade == "B"
        assert not report.has_entry_costs
        assert report.has_open_prices

    def test_grade_c_neither(self):
        report = run_preflight(open_prices=None, entry_costs=None)
        assert report.grade == "C"
        assert not report.has_entry_costs
        assert not report.has_open_prices

    def test_format_message_grade_c(self):
        report = run_preflight(open_prices=None, entry_costs=None)
        msg = report.format_message()
        assert "Quality: C" in msg
        assert "NOT PROVIDED" in msg
        assert "BrokerCost.per_trade_cost" in msg

    def test_format_message_grade_a(self):
        report = run_preflight(
            open_prices=np.array([1.0]),
            entry_costs=np.array([0.01]),
        )
        msg = report.format_message()
        assert "Quality: A" in msg
        assert "NOT PROVIDED" not in msg

    def test_items_count(self):
        report = run_preflight(open_prices=None, entry_costs=None)
        assert len(report.items) == 2


# ── simulate_trades integration ───────────────────────────────────────────────

def _make_data(n=50):
    """Generate minimal OHLC + signals for integration tests."""
    close = np.arange(100.0, 100.0 + n, dtype=np.float64)
    high = close + 0.5
    low = close - 0.5
    open_prices = close - 0.1
    signal_bars = np.array([0, 10], dtype=np.int32)
    directions = np.array([LONG, LONG], dtype=np.int8)
    sl_distances = np.array([5.0, 5.0], dtype=np.float64)
    tp_distances = np.array([10.0, 10.0], dtype=np.float64)
    return high, low, close, open_prices, signal_bars, directions, sl_distances, tp_distances


class TestSimulateTradesIntegration:
    def test_warning_emitted_grade_c(self):
        """Grade C: no costs, no open_prices → warning."""
        high, low, close, _, sig, dirs, sl, tp = _make_data()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = simulate_trades(
                high, low, close, sig, dirs, sl, tp, max_hold=50,
            )
            quality_warnings = [
                x for x in w if issubclass(x.category, BacktestQualityWarning)
            ]
            assert len(quality_warnings) == 1
            assert "Quality: C" in str(quality_warnings[0].message)

    def test_no_warning_grade_a(self):
        """Grade A: both provided → no warning."""
        high, low, close, open_p, sig, dirs, sl, tp = _make_data()
        entry_costs = np.array([0.01, 0.01], dtype=np.float64)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = simulate_trades(
                high, low, close, sig, dirs, sl, tp, max_hold=50,
                open_prices=open_p, entry_costs=entry_costs,
            )
            quality_warnings = [
                x for x in w if issubclass(x.category, BacktestQualityWarning)
            ]
            assert len(quality_warnings) == 0

    def test_warning_suppressed_by_filter(self):
        """filterwarnings("ignore") should suppress BacktestQualityWarning."""
        high, low, close, _, sig, dirs, sl, tp = _make_data()
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("ignore", category=BacktestQualityWarning)
            result = simulate_trades(
                high, low, close, sig, dirs, sl, tp, max_hold=50,
            )
            quality_warnings = [
                x for x in w if issubclass(x.category, BacktestQualityWarning)
            ]
            assert len(quality_warnings) == 0

    def test_preflight_false_no_warning(self):
        """preflight=False should suppress all pre-flight logic."""
        high, low, close, _, sig, dirs, sl, tp = _make_data()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = simulate_trades(
                high, low, close, sig, dirs, sl, tp, max_hold=50,
                preflight=False,
            )
            quality_warnings = [
                x for x in w if issubclass(x.category, BacktestQualityWarning)
            ]
            assert len(quality_warnings) == 0
        # quality should be None when preflight is disabled
        assert result.quality is None

    def test_quality_attribute_grade_c(self):
        """results.quality.grade should be accessible."""
        high, low, close, _, sig, dirs, sl, tp = _make_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BacktestQualityWarning)
            result = simulate_trades(
                high, low, close, sig, dirs, sl, tp, max_hold=50,
            )
        assert result.quality is not None
        assert result.quality.grade == "C"

    def test_quality_attribute_grade_a(self):
        high, low, close, open_p, sig, dirs, sl, tp = _make_data()
        entry_costs = np.array([0.01, 0.01], dtype=np.float64)
        result = simulate_trades(
            high, low, close, sig, dirs, sl, tp, max_hold=50,
            open_prices=open_p, entry_costs=entry_costs,
        )
        assert result.quality.grade == "A"

    def test_empty_trades_returns_trade_results(self):
        """Zero trades should still return TradeResults with quality."""
        high = np.array([100.0], dtype=np.float64)
        low = np.array([99.0], dtype=np.float64)
        close = np.array([99.5], dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BacktestQualityWarning)
            result = simulate_trades(
                high, low, close,
                signal_bars=np.array([], dtype=np.int32),
                directions=np.array([], dtype=np.int8),
                sl_distances=np.array([], dtype=np.float64),
                tp_distances=np.array([], dtype=np.float64),
                max_hold=10,
            )
        assert isinstance(result, TradeResults)
        assert len(result) == 0
        assert result.quality is not None


# ── TradeResults tests ────────────────────────────────────────────────────────

class TestTradeResults:
    def _make_results(self):
        """Create a TradeResults with synthetic data."""
        data = np.zeros(3, dtype=TRADE_RESULT_DTYPE)
        data["pnl_r"] = [1.0, -0.5, 2.0]
        data["exit_type"] = [EXIT_TP, EXIT_SL, EXIT_TIME]
        report = PreflightReport(
            grade="B", has_entry_costs=True, has_open_prices=False, items=[]
        )
        return TradeResults(data, quality=report)

    def test_isinstance_ndarray(self):
        results = self._make_results()
        assert isinstance(results, np.ndarray)
        assert isinstance(results, TradeResults)

    def test_field_access(self):
        results = self._make_results()
        assert results["pnl_r"][0] == pytest.approx(1.0)
        assert results["exit_type"][1] == EXIT_SL

    def test_indexing(self):
        results = self._make_results()
        assert results[0]["pnl_r"] == pytest.approx(1.0)
        assert len(results) == 3

    def test_slicing_preserves_quality(self):
        results = self._make_results()
        sliced = results[1:]
        assert isinstance(sliced, TradeResults)
        assert sliced.quality is not None
        assert sliced.quality.grade == "B"
        assert len(sliced) == 2

    def test_boolean_indexing_preserves_quality(self):
        results = self._make_results()
        winners = results[results["pnl_r"] > 0]
        assert isinstance(winners, TradeResults)
        assert winners.quality.grade == "B"
        assert len(winners) == 2

    def test_pickle_roundtrip(self):
        results = self._make_results()
        data = pickle.dumps(results)
        restored = pickle.loads(data)
        assert isinstance(restored, TradeResults)
        assert restored.quality.grade == "B"
        assert np.array_equal(restored["pnl_r"], results["pnl_r"])

    def test_quality_none_when_no_preflight(self):
        data = np.zeros(1, dtype=TRADE_RESULT_DTYPE)
        results = TradeResults(data, quality=None)
        assert results.quality is None

    def test_field_access_returns_plain_ndarray(self):
        """results['pnl_r'] should return plain ndarray, not TradeResults."""
        results = self._make_results()
        field = results["pnl_r"]
        assert type(field) is np.ndarray  # exact type, not subclass
        assert not isinstance(field, TradeResults)

    def test_np_mean_on_field_returns_scalar(self):
        """np.mean(results['pnl_r']) must be a numpy scalar, not TradeResults."""
        results = self._make_results()
        m = np.mean(results["pnl_r"])
        assert isinstance(m, np.floating)
        assert not isinstance(m, TradeResults)
