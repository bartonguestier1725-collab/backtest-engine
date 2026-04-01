"""Broker cost model for backtesting."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BrokerCost:
    """Broker cost model with per-instrument spread and commission data.

    **IMPORTANT — one-way vs round-trip:**

    - ``spreads``: **One-way** bid-ask spread in price units. This is the
      spread you see on your platform (e.g. EURUSD 0.2 pips = 0.00002).
      You pay it once on entry. Do NOT double it.
    - ``commission_per_lot``: **Round-trip** commission per standard lot in
      account currency (e.g. $5 RT = $2.50 per side). Do NOT halve it.
    - ``cost_price()``: Returns total cost = spread + commission (all in
      price units, round-trip equivalent). This is what gets subtracted
      from each trade's PnL.

    If you get these backwards (doubling spread or halving commission),
    your cost estimate will be off by ~2x and PF will be wrong.

    Attributes
    ----------
    spreads : dict mapping instrument name → one-way bid-ask spread in price units.
    commission_per_lot : Round-trip commission per standard lot in account currency.
    pip_values : dict mapping instrument name → pip value per lot in account currency.
    pip_sizes : dict mapping instrument name → pip size (e.g. 0.0001 for EURUSD).
    """

    spreads: dict[str, float] = field(default_factory=dict)
    commission_per_lot: float = 0.0
    pip_values: dict[str, float] = field(default_factory=dict)
    pip_sizes: dict[str, float] = field(default_factory=dict)

    def pip_size(self, instrument: str) -> float:
        """Return the pip size for an instrument.

        Standard: 0.0001 (EURUSD etc.), 0.01 (JPY pairs, XAUUSD).
        Falls back to 0.0001 if instrument is unknown.
        """
        return self.pip_sizes.get(instrument, 0.0001)

    def price_to_pips(self, instrument: str, price_diff: float) -> float:
        """Convert a price difference to pips.

        Example: XAUUSD price_diff=2.50 → 250 pips (pip_size=0.01).
        """
        ps = self.pip_size(instrument)
        if ps == 0.0:
            return 0.0
        return price_diff / ps

    def cost_price(self, instrument: str) -> float:
        """Total round-trip cost in price units for an instrument."""
        spread = self.spreads.get(instrument, 0.0)
        if self.commission_per_lot == 0.0:
            return spread

        pip_size = self.pip_sizes.get(instrument, 0.0001)
        pip_value = self.pip_values.get(instrument, 10.0)
        if pip_value == 0.0:
            return spread

        # Convert commission to price units
        commission_price = self.commission_per_lot * pip_size / pip_value
        return spread + commission_price

    def cost_prices(self) -> dict[str, float]:
        """Return {instrument: total_RT_cost_in_price_units} for all instruments."""
        return {inst: self.cost_price(inst) for inst in self.spreads}

    def as_r(self, instrument: str, risk_price: float) -> float:
        """Cost as a fraction of risk (R-units).

        Parameters
        ----------
        instrument : Instrument name.
        risk_price : SL distance in price units.
        """
        if risk_price == 0.0:
            return 0.0
        return self.cost_price(instrument) / risk_price

    def as_r_array(
        self, instruments: list[str], risk_prices: np.ndarray,
    ) -> np.ndarray:
        """Vectorized cost in R-units for multiple trades."""
        costs = np.empty(len(instruments), dtype=np.float64)
        for i, inst in enumerate(instruments):
            costs[i] = self.as_r(inst, risk_prices[i])
        return costs

    def per_trade_cost(
        self,
        instruments: list[str],
        sl_distances: np.ndarray,
    ) -> np.ndarray:
        """Per-trade cost in R-units for an array of trades.

        Equivalent to as_r_array but named for clarity in the
        simulate_trades(entry_costs=...) workflow.

        Parameters
        ----------
        instruments : List of instrument names, one per trade.
        sl_distances : SL distances in price units, one per trade.

        Returns
        -------
        1-D float64 array of costs in R-units.
        """
        return self.as_r_array(instruments, sl_distances)

    # --- Presets ---

    @classmethod
    def tradeview_ilc(cls) -> BrokerCost:
        """Tradeview ILC account preset (ECN, $5 RT commission/lot).

        Spreads are one-way bid-ask (measured). Commission is $5 round-trip.
        """
        spreads = {
            "EURUSD": 0.00002, "GBPUSD": 0.00004, "USDJPY": 0.003,
            "USDCHF": 0.00004, "AUDUSD": 0.00003, "NZDUSD": 0.00004,
            "USDCAD": 0.00004, "EURGBP": 0.00004, "EURJPY": 0.005,
            "GBPJPY": 0.008, "AUDJPY": 0.005, "EURAUD": 0.00006,
            "EURNZD": 0.00008, "GBPAUD": 0.00007, "GBPNZD": 0.00010,
            "AUDNZD": 0.00005, "AUDCAD": 0.00005, "NZDCAD": 0.00006,
            "EURCAD": 0.00005, "GBPCAD": 0.00006, "CADCHF": 0.00005,
            "CADJPY": 0.005, "CHFJPY": 0.005, "NZDJPY": 0.005,
            "GBPCHF": 0.00006, "AUDCHF": 0.00005, "NZDCHF": 0.00006,
            "EURCHF": 0.00004,
            # Metals — pip_size=0.01, pip_value=$1.00/lot
            "XAUUSD": 0.20,
        }
        pip_sizes = {}
        pip_values = {}
        for inst in spreads:
            if inst == "XAUUSD":
                pip_sizes[inst] = 0.01
                pip_values[inst] = 1.0
            elif inst.endswith("JPY"):
                pip_sizes[inst] = 0.01
                pip_values[inst] = 6.67  # approx for JPY pairs (1000 JPY per lot)
            else:
                pip_sizes[inst] = 0.0001
                pip_values[inst] = 10.0
        return cls(
            spreads=spreads,
            commission_per_lot=5.0,
            pip_values=pip_values,
            pip_sizes=pip_sizes,
        )

    @classmethod
    def fundora(cls) -> BrokerCost:
        """Fundora prop firm preset (wider spreads, no commission)."""
        spreads = {
            "EURUSD": 0.00010, "GBPUSD": 0.00014, "USDJPY": 0.012,
            "USDCHF": 0.00014, "AUDUSD": 0.00012, "NZDUSD": 0.00014,
            "USDCAD": 0.00014, "EURGBP": 0.00014, "EURJPY": 0.016,
            "GBPJPY": 0.022, "AUDJPY": 0.016, "EURAUD": 0.00018,
            "EURNZD": 0.00024, "GBPAUD": 0.00022, "GBPNZD": 0.00028,
            "AUDNZD": 0.00016, "AUDCAD": 0.00016, "NZDCAD": 0.00018,
            "EURCAD": 0.00016, "GBPCAD": 0.00018, "CADCHF": 0.00016,
            "CADJPY": 0.016, "CHFJPY": 0.016, "NZDJPY": 0.016,
            "GBPCHF": 0.00018, "AUDCHF": 0.00016, "NZDCHF": 0.00018,
            "EURCHF": 0.00014,
            # Metals — pip_size=0.01, pip_value=$1.00/lot
            "XAUUSD": 0.40,
        }
        pip_sizes = {}
        pip_values = {}
        for inst in spreads:
            if inst == "XAUUSD":
                pip_sizes[inst] = 0.01
                pip_values[inst] = 1.0
            elif inst.endswith("JPY"):
                pip_sizes[inst] = 0.01
                pip_values[inst] = 6.67
            else:
                pip_sizes[inst] = 0.0001
                pip_values[inst] = 10.0
        return cls(
            spreads=spreads,
            commission_per_lot=0.0,
            pip_values=pip_values,
            pip_sizes=pip_sizes,
        )
