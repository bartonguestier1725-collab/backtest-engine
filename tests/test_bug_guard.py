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
