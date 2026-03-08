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
        assert len(bc.spreads) == 27
        # Cost should be positive
        cost = bc.cost_price("EURUSD")
        assert cost > 0.0

    def test_fundora(self):
        bc = BrokerCost.fundora()
        assert "EURUSD" in bc.spreads
        assert bc.commission_per_lot == 0.0
        assert len(bc.spreads) == 28  # 27 FX pairs + XAUUSD
        # Fundora spreads should be wider than Tradeview
        tv = BrokerCost.tradeview_ilc()
        assert bc.spreads["EURUSD"] > tv.spreads["EURUSD"]

    def test_jpy_pairs_have_correct_pip_size(self):
        bc = BrokerCost.tradeview_ilc()
        assert bc.pip_sizes["USDJPY"] == 0.01
        assert bc.pip_sizes["EURJPY"] == 0.01
        assert bc.pip_sizes["EURUSD"] == 0.0001
