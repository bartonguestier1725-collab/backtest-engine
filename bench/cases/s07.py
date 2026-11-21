import numpy as np
import pandas as pd
from backtest_engine import simulate_trades, atr, rsi, sma, LONG, SHORT


def signals(data):
    r = rsi(data["close"], 14)
    return np.where(r < 35, 1, 0)


def backtest(data):
    sig = signals(data)
    o, h, l, c = data["open"], data["high"], data["low"], data["close"]
    a = atr(h, l, c, 14)
    sb = np.flatnonzero((sig != 0) & ~np.isnan(a))
    sb = sb[sb < len(c) - 13].astype(np.int32)
    d = np.full(len(sb), LONG, dtype=np.int8)
    reach = np.array([h[i + 1:i + 13].max() for i in sb]) - o[sb + 1]
    tp = np.maximum(reach * 0.9, a[sb] * 0.5)
    return simulate_trades(h, l, c, sb, d, a[sb] * 2.0, tp, max_hold=12, open_prices=o)
