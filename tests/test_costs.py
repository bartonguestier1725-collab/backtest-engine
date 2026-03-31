"""Tests for backtest_engine.costs."""

import numpy as np
import pytest

from backtest_engine.costs import BrokerCost


class TestBrokerCost:
    def test_cost_price_spread_only(self):
        bc = BrokerCost(spreads={"EURUSD": 0.0001})
        assert bc.cost_price("EURUSD") == pytest.approx(0.0001)

    def test_cost_price_with_commission(self):
        bc = BrokerCost(
            spreads={"EURUSD": 0.00002},
            commission_per_lot=5.0,
            pip_values={"EURUSD": 10.0},
            pip_sizes={"EURUSD": 0.0001},
        )
        # commission in price = 5.0 * 0.0001 / 10.0 = 0.00005
        expected = 0.00002 + 0.00005
        assert bc.cost_price("EURUSD") == pytest.approx(expected)

    def test_cost_unknown_instrument(self):
        bc = BrokerCost()
        assert bc.cost_price("UNKNOWN") == 0.0

    def test_as_r(self):
        bc = BrokerCost(spreads={"EURUSD": 0.0001})
        # Risk = 0.001 (10 pips SL)
        r_cost = bc.as_r("EURUSD", 0.001)
        assert r_cost == pytest.approx(0.1)  # 0.0001 / 0.001 = 0.1R

    def test_as_r_zero_risk(self):
        bc = BrokerCost(spreads={"EURUSD": 0.0001})
        assert bc.as_r("EURUSD", 0.0) == 0.0

    def test_as_r_array(self):
        bc = BrokerCost(spreads={"EURUSD": 0.0001, "GBPUSD": 0.0002})
        instruments = ["EURUSD", "GBPUSD"]
        risks = np.array([0.001, 0.002], dtype=np.float64)
        result = bc.as_r_array(instruments, risks)
        assert result[0] == pytest.approx(0.1)
        assert result[1] == pytest.approx(0.1)


class TestPresets:
    def test_tradeview_ilc(self):
        bc = BrokerCost.tradeview_ilc()
        assert "EURUSD" in bc.spreads
        assert bc.commission_per_lot == 5.0
        assert len(bc.spreads) == 29  # 27 FX pairs + EURCHF + XAUUSD
        # Cost should be positive
        cost = bc.cost_price("EURUSD")
        assert cost > 0.0

    def test_fundora(self):
        bc = BrokerCost.fundora()
        assert "EURUSD" in bc.spreads
        assert bc.commission_per_lot == 0.0
        assert len(bc.spreads) == 29  # 27 FX pairs + EURCHF + XAUUSD
        # Fundora spreads should be wider than Tradeview
        tv = BrokerCost.tradeview_ilc()
        assert bc.spreads["EURUSD"] > tv.spreads["EURUSD"]

    def test_fundora_eurchf_present(self):
        """EURCHF should be in fundora cost model."""
        bc = BrokerCost.fundora()
        assert "EURCHF" in bc.spreads
        assert bc.spreads["EURCHF"] == pytest.approx(0.00014)

    def test_jpy_pairs_have_correct_pip_size(self):
        bc = BrokerCost.tradeview_ilc()
        assert bc.pip_sizes["USDJPY"] == 0.01
        assert bc.pip_sizes["EURJPY"] == 0.01
        assert bc.pip_sizes["EURUSD"] == 0.0001


class TestCostPrices:
    def test_cost_prices_returns_all_instruments(self):
        bc = BrokerCost.tradeview_ilc()
        prices = bc.cost_prices()
        assert len(prices) == len(bc.spreads)
        for inst in bc.spreads:
            assert inst in prices
            assert prices[inst] == pytest.approx(bc.cost_price(inst))

    def test_cost_prices_spread_only(self):
        bc = BrokerCost(spreads={"EURUSD": 0.0001, "GBPUSD": 0.0002})
        prices = bc.cost_prices()
        assert prices["EURUSD"] == pytest.approx(0.0001)
        assert prices["GBPUSD"] == pytest.approx(0.0002)

    def test_cost_prices_empty(self):
        bc = BrokerCost()
        assert bc.cost_prices() == {}


class TestPipSize:
    def test_pip_size_eurusd(self):
        bc = BrokerCost.tradeview_ilc()
        assert bc.pip_size("EURUSD") == pytest.approx(0.0001)

    def test_pip_size_usdjpy(self):
        bc = BrokerCost.tradeview_ilc()
        assert bc.pip_size("USDJPY") == pytest.approx(0.01)

    def test_pip_size_xauusd(self):
        bc = BrokerCost.tradeview_ilc()
        assert bc.pip_size("XAUUSD") == pytest.approx(0.01)

    def test_pip_size_unknown_defaults(self):
        bc = BrokerCost()
        assert bc.pip_size("UNKNOWN") == pytest.approx(0.0001)

    def test_price_to_pips_eurusd(self):
        bc = BrokerCost.tradeview_ilc()
        # 10 pips = 0.001 price diff
        assert bc.price_to_pips("EURUSD", 0.001) == pytest.approx(10.0)

    def test_price_to_pips_xauusd(self):
        bc = BrokerCost.tradeview_ilc()
        # 250 pips = 2.50 price diff (pip_size=0.01)
        assert bc.price_to_pips("XAUUSD", 2.50) == pytest.approx(250.0)

    def test_price_to_pips_usdjpy(self):
        bc = BrokerCost.tradeview_ilc()
        # 15 pips = 0.15 price diff
        assert bc.price_to_pips("USDJPY", 0.15) == pytest.approx(15.0)

    def test_price_to_pips_zero_pip_size(self):
        bc = BrokerCost(pip_sizes={"TEST": 0.0})
        assert bc.price_to_pips("TEST", 1.0) == 0.0


class TestPerTradeCost:
    def test_per_trade_cost_equals_as_r_array(self):
        bc = BrokerCost(spreads={"EURUSD": 0.0001, "GBPUSD": 0.0002})
        instruments = ["EURUSD", "GBPUSD"]
        risks = np.array([0.001, 0.002], dtype=np.float64)
        result1 = bc.per_trade_cost(instruments, risks)
        result2 = bc.as_r_array(instruments, risks)
        np.testing.assert_array_equal(result1, result2)
