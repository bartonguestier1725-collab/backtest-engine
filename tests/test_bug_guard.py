"""Tests for backtest_engine.bug_guard — check_cost_registry focus."""

import warnings

import numpy as np
import pytest

from backtest_engine.bug_guard import (
    check_cost_registry,
    check_open_prices_provided,
    check_fixed_cost_usage,
    check_spread_filter,
    check_effective_no_sl,
    run_all_checks,
    _FUNDORA_COST_PIPS,
    _cost_pips_to_price,
    _fundora_expected_costs,
)


class TestCheckCostRegistry:
    def test_custom_expected_costs_pass(self):
        """Costs within tolerance should pass."""
        expected = {"EURUSD": 0.00010, "GBPUSD": 0.00020}
        used = {"EURUSD": 0.00010, "GBPUSD": 0.00019}
        result = check_cost_registry(used, expected_costs=expected)
        assert result.passed

    def test_custom_expected_costs_fail(self):
        """Costs significantly under expected should fail."""
        expected = {"EURUSD": 0.00010}
        used = {"EURUSD": 0.00005}  # 50% under
        result = check_cost_registry(used, expected_costs=expected)
        assert not result.passed
        assert "COST MISMATCH" in result.message

    def test_none_expected_costs_warns(self):
        """None expected_costs should emit WARN (not silently pass)."""
        used = {"EURUSD": 0.00001}
        result = check_cost_registry(used, expected_costs=None)
        assert not result.passed
        assert result.severity == "WARN"
        assert "No expected costs" in result.message

    def test_zero_cost_used(self):
        """Zero cost in spreads_used should be flagged."""
        expected = {"EURUSD": 0.00010}
        used = {"EURUSD": 0.0}
        result = check_cost_registry(used, expected_costs=expected)
        assert not result.passed
        assert "cost=0" in result.message

    def test_pair_not_in_expected(self):
        """Pairs not in expected_costs should be silently skipped."""
        expected = {"EURUSD": 0.00010}
        used = {"GBPUSD": 0.00020}  # Not in expected
        result = check_cost_registry(used, expected_costs=expected)
        assert result.passed

    def test_tolerance_boundary(self):
        """Cost exactly at tolerance boundary should pass."""
        expected = {"EURUSD": 0.00010}
        # 20% tolerance: 0.00008 is exactly at boundary (ratio=0.80)
        used = {"EURUSD": 0.00008}
        result = check_cost_registry(used, expected_costs=expected, tolerance=0.20)
        assert result.passed

    def test_fundora_expected_costs_helper(self):
        """_fundora_expected_costs() should delegate to BrokerCost.fundora().cost_prices()."""
        from backtest_engine.costs import BrokerCost

        costs = _fundora_expected_costs()
        canonical = BrokerCost.fundora().cost_prices()
        assert costs == canonical
        for pair, price in costs.items():
            assert price > 0


class TestRunAllChecksBackwardCompat:
    def test_deprecated_broker_kwarg(self):
        """Old broker kwarg should work with DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            report = run_all_checks(
                spreads_used={"EURUSD": 0.00010},
                broker="fundora",
                resolution_minutes=1,
                strict=False,
            )
            # Should have a DeprecationWarning
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 1
            assert "broker" in str(dep_warnings[0].message)

    def test_deprecated_broker_non_fundora_warns(self):
        """Non-fundora broker should emit WARN for cost check (no expected_costs)."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            report = run_all_checks(
                spreads_used={"EURUSD": 0.00001},
                broker="other",
                resolution_minutes=1,
                strict=False,
            )
        # Cost check should have WARN severity
        cost_results = [r for r in report.results if r.check_id == "BG-02"]
        assert len(cost_results) == 1
        assert cost_results[0].severity == "WARN"

    def test_new_expected_costs_kwarg(self):
        """New expected_costs kwarg should work without warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            report = run_all_checks(
                spreads_used={"EURUSD": 0.00010},
                expected_costs={"EURUSD": 0.00010},
                resolution_minutes=1,
                strict=False,
            )
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 0
        assert report.passed

    def test_unexpected_kwargs_raises(self):
        """Unknown kwargs should raise TypeError."""
        with pytest.raises(TypeError, match="Unexpected keyword arguments"):
            run_all_checks(
                resolution_minutes=1,
                strict=False,
                bogus_param=True,
            )


class TestCheckOpenPricesProvided:
    def test_open_prices_provided(self):
        result = check_open_prices_provided(True)
        assert result.passed

    def test_open_prices_not_provided(self):
        result = check_open_prices_provided(False)
        assert not result.passed
        assert result.severity == "WARN"
        assert "ENTRY AT CLOSE" in result.message


class TestCheckFixedCostUsage:
    def test_per_trade_cost_passes(self):
        result = check_fixed_cost_usage(
            sl_distances=np.array([1.0, 1.5, 2.0]),
            cost_is_fixed=False,
        )
        assert result.passed

    def test_fixed_cost_with_varying_sl_warns(self):
        result = check_fixed_cost_usage(
            sl_distances=np.array([1.0, 3.0, 5.0]),
            cost_is_fixed=True,
        )
        assert not result.passed
        assert result.severity == "WARN"

    def test_fixed_cost_with_uniform_sl_passes(self):
        result = check_fixed_cost_usage(
            sl_distances=np.array([1.0, 1.0, 1.0]),
            cost_is_fixed=True,
        )
        assert result.passed


class TestCheckSpreadFilter:
    def test_no_constraint(self):
        result = check_spread_filter(signal_spreads=None, max_spread=0.0)
        assert result.passed

    def test_all_within_spread(self):
        spreads = np.array([0.0001, 0.00008, 0.00012])
        result = check_spread_filter(signal_spreads=spreads, max_spread=0.00015)
        assert result.passed

    def test_violations_detected(self):
        spreads = np.array([0.0001, 0.0002, 0.0003])
        result = check_spread_filter(signal_spreads=spreads, max_spread=0.00015)
        assert not result.passed
        assert "SPREAD FILTER LEAK" in result.message

    def test_max_spread_without_signal_spreads_warns(self):
        """max_spread > 0 but signal_spreads=None should WARN (missing data)."""
        result = check_spread_filter(signal_spreads=None, max_spread=0.0002)
        assert not result.passed
        assert result.severity == "WARN"
        assert "not provided" in result.message


class TestRunAllChecksBG12BG13Integration:
    """BG-12 and BG-13 should be called from run_all_checks() when params provided."""

    def test_bg12_called_when_sl_distances_provided(self):
        """BG-12 should appear in report when sl_distances is passed."""
        report = run_all_checks(
            resolution_minutes=1,
            sl_distances=np.array([1.0, 3.0, 5.0]),
            cost_is_fixed=True,
            strict=False,
        )
        bg12 = [r for r in report.results if r.check_id == "BG-12"]
        assert len(bg12) == 1
        assert not bg12[0].passed  # CV > 10% → WARN

    def test_bg12_not_called_without_sl_distances(self):
        """BG-12 should NOT appear when sl_distances is not passed."""
        report = run_all_checks(
            resolution_minutes=1,
            strict=False,
        )
        bg12 = [r for r in report.results if r.check_id == "BG-12"]
        assert len(bg12) == 0

    def test_bg13_called_when_signal_spreads_provided(self):
        """BG-13 should appear in report when signal_spreads is passed."""
        spreads = np.array([0.0001, 0.0002, 0.0003])
        report = run_all_checks(
            resolution_minutes=1,
            signal_spreads=spreads,
            max_spread=0.00015,
            strict=False,
        )
        bg13 = [r for r in report.results if r.check_id == "BG-13"]
        assert len(bg13) == 1
        assert not bg13[0].passed  # 2/3 violations

    def test_bg13_not_called_without_params(self):
        """BG-13 should NOT appear when neither signal_spreads nor max_spread given."""
        report = run_all_checks(
            resolution_minutes=1,
            strict=False,
        )
        bg13 = [r for r in report.results if r.check_id == "BG-13"]
        assert len(bg13) == 0

    def test_bg13_warns_when_max_spread_only(self):
        """max_spread without signal_spreads → BG-13 WARN (missing data)."""
        report = run_all_checks(
            resolution_minutes=1,
            max_spread=0.0002,
            strict=False,
        )
        bg13 = [r for r in report.results if r.check_id == "BG-13"]
        assert len(bg13) == 1
        assert not bg13[0].passed
        assert bg13[0].severity == "WARN"
        assert "not provided" in bg13[0].message


class TestCheckEffectiveNoSl:
    """BG-14: Detect when SL distances are so large they're effectively disabled."""

    def test_normal_sl_passes(self):
        """Normal SL relative to price should pass."""
        sl = np.array([5.0, 6.0, 4.5])  # ~0.25% of price
        result = check_effective_no_sl(sl, avg_price=2000.0)
        assert result.passed

    def test_huge_sl_warns(self):
        """SL > 5% of price should warn."""
        sl = np.array([200.0, 250.0, 180.0])  # ~10% of price
        result = check_effective_no_sl(sl, avg_price=2000.0)
        assert not result.passed
        assert "EFFECTIVELY NO SL" in result.message
        assert result.severity == "WARN"

    def test_huge_sl_coarse_bar_extra_warning(self):
        """SL > 5% on 1h bars should include coarse bar warning."""
        sl = np.array([200.0])
        result = check_effective_no_sl(sl, avg_price=2000.0, resolution_minutes=60)
        assert not result.passed
        assert "1min bars" in result.message

    def test_huge_sl_fine_bar_no_extra(self):
        """SL > 5% on 1m bars should NOT include coarse bar warning."""
        sl = np.array([200.0])
        result = check_effective_no_sl(sl, avg_price=2000.0, resolution_minutes=1)
        assert not result.passed
        assert "1min bars" not in result.message

    def test_empty_trades(self):
        """Empty sl_distances should pass."""
        result = check_effective_no_sl(np.array([]), avg_price=2000.0)
        assert result.passed

    def test_zero_avg_price_skips(self):
        """avg_price=0 should skip gracefully."""
        sl = np.array([200.0])
        result = check_effective_no_sl(sl, avg_price=0.0)
        assert result.passed  # skipped, not failed

    def test_custom_threshold(self):
        """Custom threshold should be respected."""
        sl = np.array([30.0])  # 1.5% of price
        # Default 5% threshold → pass
        result = check_effective_no_sl(sl, avg_price=2000.0, threshold=0.05)
        assert result.passed
        # Stricter 1% threshold → warn
        result = check_effective_no_sl(sl, avg_price=2000.0, threshold=0.01)
        assert not result.passed


class TestRunAllChecksBG14Integration:
    """BG-14 should be called from run_all_checks() when params provided."""

    def test_bg14_called_when_sl_and_avg_price_provided(self):
        """BG-14 should appear when both sl_distances and avg_price are given."""
        report = run_all_checks(
            resolution_minutes=60,
            sl_distances=np.array([200.0, 250.0]),
            avg_price=2000.0,
            strict=False,
            _suppress_print=True,
        )
        bg14 = [r for r in report.results if r.check_id == "BG-14"]
        assert len(bg14) == 1
        assert not bg14[0].passed

    def test_bg14_not_called_without_avg_price(self):
        """BG-14 should NOT appear when avg_price is not given."""
        report = run_all_checks(
            resolution_minutes=1,
            sl_distances=np.array([200.0]),
            strict=False,
            _suppress_print=True,
        )
        bg14 = [r for r in report.results if r.check_id == "BG-14"]
        assert len(bg14) == 0

    def test_bg14_not_called_without_sl_distances(self):
        """BG-14 should NOT appear when sl_distances is not given."""
        report = run_all_checks(
            resolution_minutes=1,
            avg_price=2000.0,
            strict=False,
            _suppress_print=True,
        )
        bg14 = [r for r in report.results if r.check_id == "BG-14"]
        assert len(bg14) == 0


# ── v0.6.0 additions ────────────────────────────────────────────────────────

from backtest_engine.bug_guard import (
    check_incomplete_bars as _cib, check_bfill_in_source, check_entry_price_type,
)


class TestBfillVariants:
    @pytest.mark.parametrize("code", [
        "df = df.backfill()\n",
        "df.fillna(method='backfill')\n",
        "df.interpolate(limit_direction='both')\n",
        "s.interpolate(method='linear', limit_direction=\"backward\")\n",
    ])
    def test_detected(self, tmp_path, code):
        src = tmp_path / "s.py"
        src.write_text(code)
        r = check_bfill_in_source(src)
        assert not r.passed and r.severity == "ERROR"

    def test_forward_interpolate_ok(self, tmp_path):
        src = tmp_path / "s.py"
        src.write_text("df.interpolate(limit_direction='forward')\ndf.ffill()\n")
        assert check_bfill_in_source(src).passed


class TestEntryPriceLoose:
    @pytest.mark.parametrize("code", [
        "entry_price = close[i]\n",
        "entry_price = closes[sig]\n",
        "entry_price = df['close'].iloc[i]\n",
    ])
    def test_lowercase_close_warns(self, tmp_path, code):
        src = tmp_path / "s.py"
        src.write_text(code)
        r = check_entry_price_type(src)
        assert not r.passed and r.severity == "WARN"

    def test_next_bar_not_flagged(self, tmp_path):
        src = tmp_path / "s.py"
        src.write_text("entry_price = close[i + 1]\nentry_price = open_[i+1]\n")
        assert check_entry_price_type(src).passed

    def test_original_pattern_still_error(self, tmp_path):
        src = tmp_path / "s.py"
        src.write_text("entry_price = df.Close.values[i]\n")
        r = check_entry_price_type(src)
        assert not r.passed and r.severity == "ERROR"


class TestIncompleteBars:
    def test_sorted_complete(self):
        ts = np.arange(10, dtype=np.int64) * 3600
        assert _cib(ts, 60, now=10 * 3600).passed

    def test_last_bar_forming_warns(self):
        ts = np.arange(10, dtype=np.int64) * 3600
        r = _cib(ts, 60, now=9 * 3600 + 600)
        assert not r.passed and r.severity == "WARN"

    def test_duplicates_error(self):
        ts = np.array([0, 3600, 3600, 7200], dtype=np.int64)
        r = _cib(ts, 60, now=10**9)
        assert not r.passed and r.severity == "ERROR"

    def test_run_all_checks_wires_timestamps(self):
        ts = np.array([0, 3600, 3600], dtype=np.int64)
        report = run_all_checks(timestamps=ts, strict=False, _suppress_print=True)
        assert any(r.check_id == "BG-10" and not r.passed for r in report.results)


class TestNanQuantile:
    def test_nanquantile_flagged(self, tmp_path):
        from backtest_engine.bug_guard import check_expanding_quantile
        src = tmp_path / "s.py"
        src.write_text("thr = np.nanquantile(a, 0.7)\n")
        assert not check_expanding_quantile(src).passed


class TestDataPeriodCalendar:
    """check_data_period with start_ts/end_ts uses the calendar span."""

    def test_unix_seconds(self):
        from backtest_engine.bug_guard import check_data_period
        # 2021-01-01 .. 2024-01-01 in Unix seconds (the library's time unit)
        r = check_data_period(100, 60, start_ts=np.int64(1609459200), end_ts=np.int64(1704067200))
        assert r.passed and "36.0 months" in r.message

    def test_timestamp_objects(self):
        import pandas as pd
        from backtest_engine.bug_guard import check_data_period
        r = check_data_period(100, 60, start_ts=pd.Timestamp("2021-01-01"),
                              end_ts=pd.Timestamp("2021-07-01"))
        assert not r.passed and r.severity == "ERROR"


class TestCoarseBarsMessage:
    def test_no_made_up_sl(self):
        from backtest_engine.bug_guard import check_resolution
        r = check_resolution(60)
        assert not r.passed and "pip" not in r.message

    def test_quotes_sl_when_given(self):
        from backtest_engine.bug_guard import check_resolution
        r = check_resolution(60, 1.5, 12.0)
        assert "SL=1.5×ATR≈18pip" in r.message


class TestDataPeriodInputs:
    def test_milliseconds_rejected(self):
        from backtest_engine.bug_guard import check_data_period
        with pytest.raises(ValueError, match="not Unix seconds"):
            check_data_period(100, 60, start_ts=1609459200_000, end_ts=1704067200_000)

    def test_nat_rejected(self):
        import pandas as pd
        from backtest_engine.bug_guard import check_data_period
        with pytest.raises(ValueError, match="NaT"):
            check_data_period(100, 60, start_ts=pd.NaT, end_ts=pd.Timestamp("2024-01-01"))

    def test_seconds_and_aware_timestamp_mix(self):
        import pandas as pd
        from backtest_engine.bug_guard import check_data_period
        r = check_data_period(100, 60, start_ts=1609459200,
                              end_ts=pd.Timestamp("2024-01-01", tz="UTC"))
        assert r.passed and "36.0 months" in r.message


class TestDataPeriodFromTimestamps:
    """v0.6.1: run_all_checks/GateKeeper take BG-06's period from the bar
    times; without them the estimate is 24h × 22 days (528 1h bars a month)."""

    def _bg06(self, **kw):
        rep = run_all_checks(resolution_minutes=1, strict=False, _suppress_print=True, **kw)
        return [r for r in rep.results if r.check_id == "BG-06"]

    def test_estimate_is_528_bars_a_month(self):
        from backtest_engine.bug_guard import check_data_period
        assert "12.0 months" in check_data_period(528 * 12, 60).message
        assert check_data_period(528 * 12, 60).passed
        # 8.7 months of real FX 1h bars (~519/month) passed the old 374/month
        # estimate as 12.1 months
        assert not check_data_period(4520, 60).passed

    def test_timestamps_override_bar_count(self):
        ts = 1609459200 + 3600 * np.arange(6 * 520, dtype=np.int64)  # ~4.3 months
        (r,) = self._bg06(n_bars=10_000_000, bar_minutes=60, timestamps=ts)
        assert not r.passed and "first and last bar times" in r.message

    def test_timestamps_alone_run_bg06(self):
        ts = np.linspace(1609459200, 1609459200 + 400 * 86400, 1000).astype(np.int64)
        (r,) = self._bg06(timestamps=ts)
        assert r.passed and "13.1 months" in r.message

    def test_millisecond_timestamps_are_an_error(self):
        ts = (1609459200 + 3600 * np.arange(100, dtype=np.int64)) * 1000
        (r,) = self._bg06(timestamps=ts)
        assert not r.passed and r.severity == "ERROR" and "BAD TIMESTAMPS" in r.message

    def test_no_bars_no_timestamps_skips(self):
        assert self._bg06() == []


class TestFixedCostMessages:
    def test_gross_is_not_called_correct(self):
        r = check_fixed_cost_usage(np.array([1.0, 2.0]), cost_is_fixed=False, has_costs=False)
        assert r.passed and "GROSS" in r.message and "correct" not in r.message

    def test_unknown_costs_are_not_called_correct(self):
        r = check_fixed_cost_usage(np.array([1.0, 2.0]))
        assert r.passed and "not checked" in r.message and "correct" not in r.message

    def test_per_trade_costs(self):
        r = check_fixed_cost_usage(np.array([1.0, 2.0]), cost_is_fixed=False, has_costs=True)
        assert r.passed and "per-trade" in r.message
