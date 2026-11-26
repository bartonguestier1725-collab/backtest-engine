"""1h-only vs 1m-fill simulation of the same signals (the README numbers).

EURUSD 2024, RSI(14) 30/70 reversal on 1h bars, SL 2xATR / TP 4xATR,
max hold 48h. Runs each simulation GROSS and NET (the sample ECN cost preset,
not any broker's quote).

  export AGGVAULT_KEY=tk_your_key
  uv run python examples/rsi_reversal/compare_1h_vs_hires.py

Fetches 1 year of 1m bars in 3 requests (under a minute).
"""

import numpy as np

from backtest_engine import (
    fetch_aggvault, simulate_trades, simulate_trades_hires,
    rsi, atr, BrokerCost, LONG, SHORT,
)

ts_1h, o_1h, h_1h, l_1h, c_1h, _ = fetch_aggvault("EURUSD", "1h", "2024-01-01", "2025-01-01")
ts_1m, o_1m, h_1m, l_1m, c_1m, _ = fetch_aggvault("EURUSD", "1m", "2024-01-01", "2025-01-01")

rsi_vals = rsi(c_1h, 14)
atr_vals = atr(h_1h, l_1h, c_1h, 14)
signal_bars, directions = [], []
for i in range(15, len(c_1h) - 1):
    if np.isnan(rsi_vals[i]) or np.isnan(atr_vals[i]):
        continue
    if rsi_vals[i] < 30:
        signal_bars.append(i); directions.append(LONG)
    elif rsi_vals[i] > 70:
        signal_bars.append(i); directions.append(SHORT)
signal_bars = np.array(signal_bars, dtype=np.int32)
directions = np.array(directions, dtype=np.int8)
sl = atr_vals[signal_bars] * 2.0
tp = atr_vals[signal_bars] * 4.0
net_costs = BrokerCost.sample_ecn().per_trade_cost(["EURUSD"] * len(signal_bars), sl)

print(f"{len(signal_bars)} signals")
for label, costs in (("GROSS", None), ("NET (sample ECN cost)", net_costs)):
    coarse = simulate_trades(
        h_1h, l_1h, c_1h, signal_bars, directions, sl, tp,
        max_hold=48, open_prices=o_1h, entry_costs=costs,
    )
    fine = simulate_trades_hires(
        signal_timestamps=ts_1h, signal_bars=signal_bars, directions=directions,
        sl_distances=sl, tp_distances=tp, max_hold=48, signal_bar_minutes=60,
        exec_timestamps=ts_1m, exec_opens=o_1m, exec_highs=h_1m,
        exec_lows=l_1m, exec_closes=c_1m, entry_costs=costs,
    )
    print(f"{label:<22} 1h fills: PF={coarse.profit_factor:.3f}  "
          f"1m fills: PF={fine.profit_factor:.3f}")
